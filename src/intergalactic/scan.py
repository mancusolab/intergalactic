# pattern: Imperative Shell

"""Stream BED phenotypes through the regional CLI fit workflow."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import time

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import polars as pl

from .cli import _covariate_frame, _fit_logging, _fit_region, _validate_test_arguments, _write_payload


def add_scan_parser(subparsers: Any, add_model_arguments: Callable[[argparse.ArgumentParser], None]) -> None:
    parser = subparsers.add_parser("scan", help="fit expression BED rows sequentially")
    parser.add_argument("linear_arg_bundle", type=Path)
    parser.add_argument("--phenotype-bed", type=Path, required=True)
    parser.add_argument("--chr", dest="chromosome", help="restrict chromosome (1 and chr1 are equivalent)")
    parser.add_argument("--gene", help="restrict to one exact gene ID")
    parser.add_argument("--window-bp", type=int, default=1_000_000, help="half-width around BED start")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fail-fast",
        dest="continue_on_error",
        action="store_false",
        help="stop after the first failed or nonconverged gene (default: continue)",
    )
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--skip-existing", action="store_true", help="skip existing successful results")
    existing.add_argument("--overwrite", action="store_true", help="replace existing per-gene results")
    parser.add_argument("--log-file", type=Path, help="default: OUTPUT_DIR/scan.log")
    parser.add_argument("--verbose", action="store_true")
    add_model_arguments(parser)
    parser.set_defaults(func=run_scan, phenotype_id_column="IID", phenotype_separator=None, output=None)


def _bed_rows(path: Path, *, chromosome: str | None, gene: str | None) -> Iterator[tuple[list[str], list[str]]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None or len(header) < 5:
            raise ValueError("expression BED must have a header and at least one sample column")
        normalized = [name.lstrip("#").lower() for name in header[:4]]
        if (
            normalized[0] not in {"chr", "chrom", "chromosome"}
            or normalized[1] not in {"start", "chromstart"}
            or normalized[2] not in {"end", "chromend"}
            or normalized[3] not in {"gene", "gene_id", "phenotype_id", "id"}
        ):
            raise ValueError("BED header must begin with chrom, start, end, gene_id (or phenotype_id)")
        samples = header[4:]
        if any(not sample.strip() for sample in samples) or len(set(samples)) != len(samples):
            raise ValueError("BED sample IDs must be nonempty and unique")
        seen: set[str] = set()
        filenames: dict[str, str] = {}
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"BED line {line_number} has {len(row)} fields; expected {len(header)}")
            if chromosome is not None and row[0].removeprefix("chr") != chromosome.removeprefix("chr"):
                continue
            if gene is not None and row[3] != gene:
                continue
            if not row[0] or not row[3].strip():
                raise ValueError(f"BED line {line_number} has an empty chromosome or gene ID")
            if row[3] in seen:
                raise ValueError(f"duplicate selected BED gene ID: {row[3]}")
            seen.add(row[3])
            name = _result_name(row[3])
            if name in filenames:
                raise ValueError(f"gene filename collision: {row[3]} and {filenames[name]}")
            filenames[name] = row[3]
            yield samples, row


def _result_name(gene: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", gene) + ".json"


def _region(row: list[str], window: int) -> str:
    start, end = int(row[1]), int(row[2])
    if start < 0 or end <= start:
        raise ValueError("BED coordinates must satisfy 0 <= start < end")
    return f"{row[0]}:{max(0, start - window)}-{start + window}"


def _phenotype(samples: list[str], row: list[str]) -> pl.LazyFrame:
    values: list[float | None] = []
    for sample, value in zip(samples, row[4:], strict=True):
        if value.strip().lower() in {"", ".", "na", "nan", "null"}:
            values.append(None)
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"nonfinite phenotype for sample {sample}: {value}")
            values.append(number)
    return pl.DataFrame(
        {"sample_id": samples, "phenotype": values}, schema={"sample_id": pl.String, "phenotype": pl.Float64}
    ).lazy()


def run_scan(args: argparse.Namespace) -> int:
    if args.window_bp < 1:
        raise ValueError("--window-bp must be a positive integer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file is None:
        args.log_file = args.output_dir / "scan.log"
    protected = [args.phenotype_bed, args.linear_arg_bundle, args.output_dir / "summary.tsv"]
    if args.covariates is not None:
        protected.append(args.covariates)
    if args.log_file.resolve() in {path.resolve() for path in protected}:
        raise ValueError("--log-file must differ from input and summary paths")
    summary_path = (args.output_dir / "summary.tsv").resolve()
    if summary_path in {args.phenotype_bed.resolve(), args.linear_arg_bundle.resolve()} or (
        args.covariates is not None and summary_path == args.covariates.resolve()
    ):
        raise ValueError("summary.tsv must not overwrite an input file")
    with _fit_logging(args) as logger:
        started = time.perf_counter()
        logger.info("Starting sequential BED scan: %s", args.phenotype_bed)
        try:
            status = _run_scan(args, logger)
        except Exception as error:
            logger.error("Scan failed: %s", error, exc_info=args.verbose)
            return 1
        logger.info("Scan finished in %.1f seconds", time.perf_counter() - started)
        return status


def _run_scan(args: argparse.Namespace, logger: Any) -> int:
    _validate_test_arguments(args)
    covariate_data = None
    if args.covariates is not None:
        covariates, names = _covariate_frame(
            args.covariates,
            id_column=args.covariate_id_column or "IID",
            covariate_columns=args.covariate_columns,
            separator=args.covariate_separator,
        )
        covariate_data = (covariates.collect().lazy(), names)
    fields = [
        "gene_id",
        "region",
        "status",
        "n_individuals",
        "n_variants",
        "log_likelihood",
        "interaction_p_value",
        "joint_p_value",
        "test_method",
        "test_status",
        "test_message",
        "sigma_a2",
        "sigma_h2",
        "sigma_e2",
        "output",
        "message",
    ]
    count, failures = 0, 0
    summary_path = args.output_dir / "summary.tsv"
    if summary_path.resolve() == args.log_file.resolve():
        raise ValueError("--log-file must differ from the summary.tsv path")
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        handle.flush()
        for samples, row in _bed_rows(args.phenotype_bed, chromosome=args.chromosome, gene=args.gene):
            count += 1
            gene = row[3]
            output = args.output_dir / _result_name(gene)
            record: dict[str, Any] = {"gene_id": gene, "output": str(output)}
            logger.info("Gene %d: %s", count, gene)
            try:
                record["region"] = _region(row, args.window_bp)
                input_paths = {args.phenotype_bed.resolve(), args.linear_arg_bundle.resolve()}
                if args.covariates is not None:
                    input_paths.add(args.covariates.resolve())
                if output.resolve() in input_paths or output.resolve() == summary_path.resolve():
                    raise ValueError("gene result path must not overwrite an input or the summary")
                if output.resolve() == args.log_file.resolve():
                    raise ValueError("gene result path collides with --log-file")
                if output.exists() and not args.overwrite:
                    if not args.skip_existing:
                        raise ValueError(f"result exists: {output}; use --overwrite or --skip-existing")
                    payload = json.loads(output.read_text())
                    if (
                        payload.get("gene_id") != gene
                        or payload.get("region") != record["region"]
                        or not payload.get("success")
                    ):
                        raise ValueError(f"existing result does not match gene/region or did not converge: {output}")
                    record.update(status="skipped", message="existing successful result; settings not revalidated")
                    logger.info("Skipping existing result: %s", output)
                else:
                    gene_args = argparse.Namespace(**vars(args))
                    gene_args.region = record["region"]
                    gene_args.phenotype = args.phenotype_bed
                    gene_args.phenotype_column = gene
                    gene_args.output = output
                    payload = _fit_region(
                        gene_args, logger, phenotype_frame=_phenotype(samples, row), covariate_data=covariate_data
                    )
                    payload.update(
                        gene_id=gene, phenotype_bed=str(args.phenotype_bed), bed_start=int(row[1]), bed_end=int(row[2])
                    )
                    _write_payload(payload, output)
                    record.update(
                        status="success" if payload["success"] else "nonconverged", message=payload["message"]
                    )
                    logger.info("Wrote results to %s", output)
                record.update({key: payload[key] for key in ("n_individuals", "n_variants", "log_likelihood")})
                record.update(payload["variance_components"])
                tests = payload.get("tests", {})
                for target in ("interaction", "joint"):
                    record[f"{target}_p_value"] = tests.get(target, {}).get("p_value")
                record["test_method"] = ";".join(f"{name}:{test['method']}" for name, test in tests.items())
                record["test_status"] = ";".join(
                    f"{name}:{'success' if test['success'] else 'unavailable'}" for name, test in tests.items()
                )
                record["test_message"] = ";".join(
                    f"{name}:{test['reason']}" for name, test in tests.items() if test.get("reason")
                )
                if record["status"] == "nonconverged":
                    failures += 1
            except Exception as error:
                failures += 1
                record.update(status="failed", message=str(error))
                logger.error("Gene %s failed: %s", gene, error, exc_info=args.verbose)
            writer.writerow(record)
            handle.flush()
            if record["status"] in {"failed", "nonconverged"} and not args.continue_on_error:
                break
    if count == 0:
        raise ValueError("no BED genes matched the requested filters")
    logger.info("Scan processed %d genes; %d failures; summary: %s", count, failures, summary_path)
    return int(failures > 0)
