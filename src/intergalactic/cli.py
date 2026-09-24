# pattern: Imperative Shell

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from scipy.sparse.linalg import aslinearoperator, LinearOperator

from .inference import restricted_likelihood_ratio_test
from .kernels import validate_haplotype_count
from .likelihood import optimize_variance_components, VarianceComponents
from .operators import (
    _CallableLinearOperator,
    AdditiveHaplotypeKernel,
    ConcatenatedHaplotypeOperator,
    DiploidHaplotypeMap,
    InteractionMode,
    SameHaplotypeInteractionKernel,
)


@dataclass(frozen=True)
class LinearArgSelection:
    """Loaded LinearARG operator and selected regional metadata."""

    linear_arg: Any
    block_name: str | None
    region: str | None


@dataclass(frozen=True)
class ModelInputs:
    """Phenotype and optional covariate arrays aligned to LinearARG individual order."""

    sample_ids: list[str]
    phenotype: npt.NDArray[np.float64]
    covariates: npt.NDArray[np.float64] | None
    covariate_names: list[str]


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int
    end: int


def _normalized_separator(separator: str | None, path: Path) -> str:
    if separator is not None:
        return separator.encode("utf-8").decode("unicode_escape")
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _scan_table(path: Path, *, separator: str | None = None) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        return pl.scan_csv(path, separator=_normalized_separator(separator, path))
    raise ValueError(f"unsupported table format for {path}; expected .csv, .tsv, .txt, or .parquet")


def _require_columns(lazy_frame: pl.LazyFrame, columns: Sequence[str], *, table_name: str) -> None:
    available = set(lazy_frame.collect_schema().names())
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(missing)}")


def _duplicate_keys(lazy_frame: pl.LazyFrame, key: str) -> list[str]:
    duplicates = (
        lazy_frame.group_by(key)
        .agg(pl.len().alias("_count"))
        .filter(pl.col("_count") > 1)
        .select(pl.col(key).cast(pl.String))
        .collect()
    )
    return duplicates.get_column(key).to_list()


def _phenotype_frame(
    path: Path,
    *,
    id_column: str,
    phenotype_column: str,
    separator: str | None,
) -> pl.LazyFrame:
    raw = _scan_table(path, separator=separator)
    _require_columns(raw, [id_column, phenotype_column], table_name="phenotype table")
    duplicates = _duplicate_keys(raw, id_column)
    if duplicates:
        raise ValueError(f"phenotype table has duplicate sample IDs: {', '.join(duplicates[:5])}")
    return raw.select(
        pl.col(id_column).cast(pl.String).alias("sample_id"),
        pl.col(phenotype_column).cast(pl.Float64).alias("phenotype"),
    )


def _covariate_columns(raw: pl.LazyFrame, *, id_column: str, requested_columns: Sequence[str] | None) -> list[str]:
    if requested_columns is not None:
        return list(requested_columns)
    identifier_columns = {id_column, "FID", "IID"}
    return [column for column in raw.collect_schema().names() if column not in identifier_columns]


def _covariate_frame(
    path: Path,
    *,
    id_column: str,
    covariate_columns: Sequence[str] | None,
    separator: str | None,
) -> tuple[pl.LazyFrame, list[str]]:
    raw = _scan_table(path, separator=separator)
    columns = _covariate_columns(raw, id_column=id_column, requested_columns=covariate_columns)
    if not columns:
        raise ValueError("covariate table must contain at least one covariate column")
    _require_columns(raw, [id_column, *columns], table_name="covariate table")
    duplicates = _duplicate_keys(raw, id_column)
    if duplicates:
        raise ValueError(f"covariate table has duplicate sample IDs: {', '.join(duplicates[:5])}")
    return (
        raw.select(
            pl.col(id_column).cast(pl.String).alias("sample_id"),
            *(pl.col(column).cast(pl.Float64).alias(column) for column in columns),
        ),
        columns,
    )


def _missing_values(frame: pl.DataFrame, columns: Sequence[str]) -> list[str]:
    missing = frame.select(pl.any_horizontal(*(pl.col(column).is_null() for column in columns)).alias("_missing"))
    mask = missing.get_column("_missing")
    return frame.filter(mask).get_column("sample_id").to_list()


def load_model_inputs(
    *,
    phenotype_path: Path,
    phenotype_id_column: str,
    phenotype_column: str,
    sample_ids: Sequence[str],
    covariate_path: Path | None = None,
    covariate_id_column: str | None = None,
    covariate_columns: Sequence[str] | None = None,
    phenotype_separator: str | None = None,
    covariate_separator: str | None = None,
    allow_missing_samples: bool = False,
    phenotype_frame: pl.LazyFrame | None = None,
    covariate_data: tuple[pl.LazyFrame, list[str]] | None = None,
) -> ModelInputs:
    """Load and align phenotype/covariate tables to LinearARG individual IDs."""
    ordered_samples = pl.DataFrame(
        {"sample_id": [str(sample_id) for sample_id in sample_ids], "_row_order": np.arange(len(sample_ids))}
    ).lazy()
    phenotype = (
        phenotype_frame
        if phenotype_frame is not None
        else _phenotype_frame(
            phenotype_path,
            id_column=phenotype_id_column,
            phenotype_column=phenotype_column,
            separator=phenotype_separator,
        )
    )
    joined = ordered_samples.join(phenotype, on="sample_id", how="left", validate="1:1")
    covariate_names: list[str] = []
    if covariate_data is None and covariate_path is not None:
        covariate_data = _covariate_frame(
            covariate_path,
            id_column=covariate_id_column or phenotype_id_column,
            covariate_columns=covariate_columns,
            separator=covariate_separator,
        )
    if covariate_data is not None:
        covariates, covariate_names = covariate_data
        joined = joined.join(covariates, on="sample_id", how="left", validate="1:1")

    materialized = joined.sort("_row_order").collect()
    value_columns = ["phenotype", *covariate_names]
    if allow_missing_samples:
        materialized = materialized.filter(~pl.any_horizontal(*(pl.col(column).is_null() for column in value_columns)))
        if materialized.height == 0:
            raise ValueError("no LinearARG samples have both phenotype and covariate values")
    else:
        missing_phenotypes = _missing_values(materialized, ["phenotype"])
        if missing_phenotypes:
            raise ValueError(f"missing phenotype values for samples: {', '.join(missing_phenotypes[:5])}")
        if covariate_names:
            missing_covariates = _missing_values(materialized, covariate_names)
            if missing_covariates:
                raise ValueError(f"missing covariate values for samples: {', '.join(missing_covariates[:5])}")

    phenotype_values = materialized.get_column("phenotype").to_numpy().astype(np.float64, copy=False)
    covariate_values = (
        materialized.select(covariate_names).to_numpy().astype(np.float64, copy=False) if covariate_names else None
    )
    return ModelInputs(
        sample_ids=materialized.get_column("sample_id").to_list(),
        phenotype=phenotype_values,
        covariates=covariate_values,
        covariate_names=covariate_names,
    )


def parse_region(region: str) -> Region:
    try:
        chrom, interval = region.split(":", maxsplit=1)
        start, end = interval.split("-", maxsplit=1)
    except ValueError as error:
        raise ValueError("region must have the form chrom:start-end") from error
    parsed = Region(chrom=chrom.removeprefix("chr"), start=int(start), end=int(end))
    if parsed.start >= parsed.end:
        raise ValueError("region start must be less than region end")
    return parsed


def _block_interval(block: dict[str, Any]) -> Region | None:
    if {"chrom", "start", "end"}.issubset(block):
        return Region(str(block["chrom"]).removeprefix("chr"), int(block["start"]), int(block["end"]))
    block_name = block.get("block_name")
    if not isinstance(block_name, str) or ":" not in block_name:
        return None
    try:
        return parse_region(block_name)
    except ValueError:
        return None


def _overlaps(left: Region, right: Region) -> bool:
    return left.chrom == right.chrom and left.start < right.end and right.start < left.end


def select_block_name(blocks: pl.DataFrame | None, *, region: str | None) -> str | None:
    """Select one LinearARG block for a targeted regional fit."""
    if blocks is None or blocks.height == 0:
        return None
    if region is None:
        if blocks.height == 1:
            return str(blocks.get_column("block_name")[0])
        raise ValueError("--region is required when the LinearARG bundle contains multiple blocks")

    target = parse_region(region)
    rows = blocks.iter_rows(named=True)
    exact = [str(row["block_name"]) for row in rows if row.get("block_name") == region]
    if exact:
        return exact[0]

    matches = []
    for row in blocks.iter_rows(named=True):
        interval = _block_interval(row)
        if interval is not None and _overlaps(interval, target):
            matches.append(str(row["block_name"]))
    if not matches:
        raise ValueError(f"no LinearARG block overlaps region {region}")
    if len(matches) > 1:
        raise ValueError(f"region {region} overlaps multiple blocks; provide a narrower region or block-level bundle")
    return matches[0]


def _region_bed(region: str, *, variant_chromosomes: Sequence[str] | None = None) -> pl.DataFrame:
    parsed = parse_region(region)
    chromosome = parsed.chrom
    if variant_chromosomes is not None:
        available = {str(value) for value in variant_chromosomes}
        prefixed = f"chr{parsed.chrom}"
        if prefixed in available:
            chromosome = prefixed
        elif parsed.chrom not in available:
            chromosome = region.split(":", maxsplit=1)[0]
    return pl.DataFrame(
        {"chrom": [chromosome], "chromStart": [parsed.start], "chromEnd": [parsed.end]},
        schema={"chrom": pl.String, "chromStart": pl.Int64, "chromEnd": pl.Int64},
    )


def _import_linear_dag() -> tuple[Any, Any, Any | None]:
    try:
        from linear_dag import LinearARG, list_blocks
    except ImportError as error:
        raise RuntimeError("linear_dag must be importable to load a LinearARG bundle") from error
    linear_arg_zarr_reader: Any | None = None
    try:
        linear_arg_zarr_reader = getattr(importlib.import_module("linear_dag.core.zarr_io"), "LinearARGZarrReader")
    except ImportError:
        pass
    return LinearARG, list_blocks, linear_arg_zarr_reader


def load_linear_arg_selection(path: Path, *, region: str | None) -> LinearArgSelection:
    """Load and concatenate all storage blocks overlapping a regional window."""
    LinearARG, list_blocks, LinearARGZarrReader = _import_linear_dag()
    reader = None
    if path.is_dir():
        if LinearARGZarrReader is None:
            raise RuntimeError("linear_dag.core.zarr_io.LinearARGZarrReader is unavailable")
        reader = LinearARGZarrReader.open(path)
        blocks = reader.list_blocks()
    else:
        blocks = list_blocks(path)
    block_names: list[str | None] = []
    if region is not None and blocks is not None:
        target = parse_region(region)
        for row in blocks.iter_rows(named=True):
            interval = _block_interval(row)
            if interval is not None and _overlaps(interval, target):
                block_names.append(str(row["block_name"]))
    if not block_names:
        block_names = [select_block_name(blocks, region=region)]

    loaded = []
    individual_ids = None
    for block_name in block_names:
        if reader is not None:
            if block_name is None:
                raise ValueError("Zarr LinearARG bundles must contain at least one block")
            linear_arg = reader.read_block(block_name, load_metadata=region is not None)
        else:
            linear_arg = LinearARG.read(path, block=block_name, load_metadata=region is not None)
        if len(block_names) > 1:
            block_ids = individual_ids_from_linear_arg(linear_arg)
            if individual_ids is None:
                individual_ids = block_ids
            elif block_ids != individual_ids:
                raise ValueError("overlapping LinearARG blocks must have identical sample identities and order")
        if region is not None and getattr(linear_arg, "variants", None) is None and block_name != region:
            raise ValueError("region filtering requires variant metadata on the selected LinearARG")
        if region is not None and getattr(linear_arg, "variants", None) is not None:
            variants = linear_arg.variants
            variant_chromosomes = None
            columns = variants.collect_schema().names() if isinstance(variants, pl.LazyFrame) else variants.columns
            if "CHROM" in columns:
                chromosome_frame = variants.select(pl.col("CHROM").cast(pl.String).unique())
                if isinstance(chromosome_frame, pl.LazyFrame):
                    chromosome_frame = chromosome_frame.collect()
                variant_chromosomes = chromosome_frame.get_column("CHROM").to_list()
            linear_arg.filter_variants_by_bed(_region_bed(region, variant_chromosomes=variant_chromosomes))
        loaded.append(linear_arg)
    if len(loaded) == 1:
        return LinearArgSelection(linear_arg=loaded[0], block_name=block_names[0], region=region)
    nonempty = [block for block in loaded if block.shape[1] > 0]
    combined = ConcatenatedHaplotypeOperator(nonempty, iids=individual_ids or []) if nonempty else loaded[0]
    return LinearArgSelection(
        linear_arg=combined,
        block_name="|".join(str(name) for name in block_names),
        region=region,
    )


def individual_ids_from_linear_arg(linear_arg: Any) -> list[str]:
    """Return diploid individual IDs from adjacent haplotype-level LinearARG IDs."""
    n_individuals = validate_haplotype_count(linear_arg.shape[0])
    iids = getattr(linear_arg, "iids", None)
    if iids is None:
        raise ValueError("LinearARG bundle must contain iids for phenotype alignment")
    iid_values = (
        [str(value) for value in iids.to_list()] if hasattr(iids, "to_list") else [str(value) for value in iids]
    )
    if len(iid_values) == n_individuals:
        return iid_values
    if len(iid_values) != 2 * n_individuals:
        raise ValueError("LinearARG iid count must equal n_individuals or 2 * n_individuals")
    individual_ids = []
    for index in range(n_individuals):
        first, second = iid_values[2 * index : 2 * index + 2]
        if first != second:
            raise ValueError("haplotype-level LinearARG iids must be adjacent duplicated individual IDs")
        individual_ids.append(first)
    return individual_ids


def _linear_arg_for_samples(
    linear_arg: Any, sample_ids: Sequence[str], selected_sample_ids: Sequence[str]
) -> LinearOperator:
    """Return a haplotype operator containing only selected diploid individuals."""
    base = aslinearoperator(linear_arg)
    individual_indices = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    missing = [sample_id for sample_id in selected_sample_ids if sample_id not in individual_indices]
    if missing:
        raise ValueError(f"selected samples are absent from LinearARG: {', '.join(missing[:5])}")
    haplotype_indices = np.asarray(
        [
            hap_index
            for sample_id in selected_sample_ids
            for hap_index in (2 * individual_indices[sample_id], 2 * individual_indices[sample_id] + 1)
        ],
        dtype=np.intp,
    )

    def scatter(values: npt.ArrayLike) -> npt.NDArray[np.number]:
        values_array = np.asarray(values)
        shape = (base.shape[0],) if values_array.ndim == 1 else (base.shape[0], values_array.shape[1])
        expanded = np.zeros(shape, dtype=values_array.dtype)
        expanded[haplotype_indices] = values_array
        return expanded

    return _CallableLinearOperator(
        shape=(len(haplotype_indices), base.shape[1]),
        dtype=base.dtype,
        matvec=lambda values: np.asarray(base @ values)[haplotype_indices],
        matmat=lambda values: np.asarray(base @ values)[haplotype_indices],
        rmatvec=lambda values: base.T @ scatter(values),
        rmatmat=lambda values: base.T @ scatter(values),
    )


def _initial_components(args: argparse.Namespace) -> VarianceComponents | None:
    values = (args.initial_sigma_a2, args.initial_sigma_h2, args.initial_sigma_e2)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("initial variance components must be provided together")
    return VarianceComponents(float(values[0]), float(values[1]), float(values[2]))


def _fit_payload(
    *, args: argparse.Namespace, selection: LinearArgSelection, inputs: ModelInputs, fit: Any
) -> dict[str, Any]:
    return {
        "linear_arg_path": str(args.linear_arg_bundle),
        "region": selection.region,
        "block_name": selection.block_name,
        "n_individuals": len(inputs.sample_ids),
        "n_variants": int(selection.linear_arg.shape[1]),
        "phenotype_column": args.phenotype_column,
        "likelihood_method": args.likelihood_method,
        "residual_degrees_of_freedom": len(inputs.sample_ids) - len(inputs.covariate_names),
        "covariate_columns": inputs.covariate_names,
        "variance_components": {
            "sigma_a2": fit.variance_components.sigma_a2,
            "sigma_h2": fit.variance_components.sigma_h2,
            "sigma_e2": fit.variance_components.sigma_e2,
        },
        "fixed_effects": [float(value) for value in fit.fixed_effects],
        "log_likelihood": float(fit.log_likelihood),
        "negative_log_likelihood": float(fit.negative_log_likelihood),
        "success": bool(fit.success),
        "message": str(fit.message),
        "n_iterations": int(fit.n_iterations),
        "accepted_steps": int(fit.accepted_steps),
        "rejected_steps": int(fit.rejected_steps),
        "trust_radius": float(fit.trust_radius),
        "logdet_method": str(fit.logdet_method),
        "num_logdet_probes": int(fit.num_logdet_probes),
        "lanczos_rank": int(fit.lanczos_rank),
    }


def _write_payload(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.write_text(text)


@contextmanager
def _fit_logging(args: argparse.Namespace) -> Iterator[logging.Logger]:
    """Own the handlers for one CLI invocation without changing root logging."""
    logger = logging.Logger("intergalactic", logging.DEBUG if args.verbose else logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)
    try:
        log_path = args.log_file
        if log_path is None and args.output is not None:
            log_path = Path(f"{args.output}.log")
        if log_path is not None:
            if args.output is not None and log_path.resolve() == args.output.resolve():
                raise ValueError("--log-file and --output must use different paths")
            disk = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            disk.setFormatter(formatter)
            logger.addHandler(disk)
            logger.info("Log file: %s", log_path)
        yield logger
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def run_fit(args: argparse.Namespace) -> int:
    with _fit_logging(args) as logger:
        started = time.perf_counter()
        logger.info("Starting intergalactic fit")
        logger.debug("Arguments: %s", vars(args))
        try:
            result = _run_fit(args, logger)
        except Exception as error:
            logger.error(
                "Fit failed after %.1f seconds: %s", time.perf_counter() - started, error, exc_info=args.verbose
            )
            return 1
        logger.info("Finished in %.1f seconds", time.perf_counter() - started)
        return result


def _run_fit(args: argparse.Namespace, logger: logging.Logger) -> int:
    _write_payload(_fit_region(args, logger), args.output)
    logger.info("Wrote results to %s", args.output or "stdout")
    return 0


def _validate_test_arguments(args: argparse.Namespace) -> None:
    if args.test != "none":
        if args.likelihood_method != "reml":
            raise ValueError("--test requires --likelihood-method reml")
        if args.test in {"joint", "both"} and args.pvalue_method != "bootstrap":
            raise ValueError("joint testing requires --pvalue-method bootstrap")
        if args.pvalue_method == "bootstrap" and args.bootstrap_replicates < 1:
            raise ValueError("--bootstrap-replicates must be positive")


def _fit_region(
    args: argparse.Namespace,
    logger: logging.Logger,
    *,
    phenotype_frame: pl.LazyFrame | None = None,
    covariate_data: tuple[pl.LazyFrame, list[str]] | None = None,
) -> dict[str, Any]:
    _validate_test_arguments(args)
    logger.info("Loading LinearARG: %s; region: %s", args.linear_arg_bundle, args.region or "whole block")
    selection = load_linear_arg_selection(args.linear_arg_bundle, region=args.region)
    if selection.linear_arg.shape[1] == 0:
        region_description = args.region or selection.block_name or "selected LinearARG block"
        raise ValueError(f"no LinearARG variants found in {region_description}")
    sample_ids = individual_ids_from_linear_arg(selection.linear_arg)
    logger.info(
        "Loaded block %s: %d individuals, %d variants",
        selection.block_name or "root",
        len(sample_ids),
        selection.linear_arg.shape[1],
    )
    logger.info(
        "Reading phenotype %s (%s) and covariates %s", args.phenotype, args.phenotype_column, args.covariates or "none"
    )
    inputs = load_model_inputs(
        phenotype_path=args.phenotype,
        phenotype_id_column=args.phenotype_id_column,
        phenotype_column=args.phenotype_column,
        sample_ids=sample_ids,
        covariate_path=args.covariates,
        covariate_id_column=args.covariate_id_column,
        covariate_columns=args.covariate_columns,
        phenotype_separator=args.phenotype_separator,
        covariate_separator=args.covariate_separator,
        allow_missing_samples=args.allow_sample_subset,
        phenotype_frame=phenotype_frame,
        covariate_data=covariate_data,
    )
    if args.allow_sample_subset and len(inputs.sample_ids) != len(sample_ids):
        selection = LinearArgSelection(
            linear_arg=_linear_arg_for_samples(selection.linear_arg, sample_ids, inputs.sample_ids),
            block_name=selection.block_name,
            region=selection.region,
        )
    logger.info(
        "Sample alignment retained %d of %d individuals; %d covariates",
        len(inputs.sample_ids),
        len(sample_ids),
        len(inputs.covariate_names),
    )
    logger.debug("Covariates: %s", ", ".join(inputs.covariate_names) or "none")
    diploid_map = DiploidHaplotypeMap.from_haplotypes(selection.linear_arg.shape[0])
    normalization = not args.no_normalize
    logger.info(
        "Constructing additive GRM (center=%s, normalize=%s, batch_size=%d)",
        args.center,
        normalization,
        args.kernel_batch_size,
    )
    stage_started = time.perf_counter()
    additive = AdditiveHaplotypeKernel(
        selection.linear_arg,
        diploid_map,
        normalization=normalization,
        center=args.center,
        batch_size=args.kernel_batch_size,
    )
    logger.info("Additive GRM ready in %.1f seconds", time.perf_counter() - stage_started)
    logger.info("Preparing HxH kernel (mode=%s)", args.interaction_mode)
    stage_started = time.perf_counter()
    interaction = SameHaplotypeInteractionKernel(
        selection.linear_arg,
        diploid_map,
        interaction_mode=args.interaction_mode,
        normalization=normalization,
        center=args.center,
        batch_size=args.kernel_batch_size,
    )
    logger.info("HxH kernel initialized in %.1f seconds", time.perf_counter() - stage_started)
    logger.info(
        "Optimizing variance components (%s): maxiter=%d, probes=%d, Lanczos rank=%d, seed=%d",
        args.likelihood_method.upper(),
        args.maxiter,
        args.num_logdet_probes,
        args.lanczos_rank,
        args.seed,
    )
    fit = optimize_variance_components(
        inputs.phenotype,
        additive,
        interaction,
        initial=_initial_components(args),
        covariates=inputs.covariates,
        likelihood_method=args.likelihood_method,
        projection_batch_size=args.kernel_batch_size,
        maxiter=args.maxiter,
        logdet_probe_mode=args.logdet_probe_mode,
        num_logdet_probes=args.num_logdet_probes,
        lanczos_rank=args.lanczos_rank,
        seed=args.seed,
        cg_rtol=args.cg_rtol,
        cg_atol=args.cg_atol,
        cg_maxiter=args.cg_maxiter,
        logger=logger,
    )
    report = logger.info if fit.success else logger.warning
    report(
        "Fit %s: %s; iterations=%d, accepted=%d, rejected=%d, log_likelihood=%.9g",
        "converged" if fit.success else "did not converge",
        fit.message,
        fit.n_iterations,
        fit.accepted_steps,
        fit.rejected_steps,
        fit.log_likelihood,
    )
    components = fit.variance_components
    logger.info(
        "Variance components: additive=%.9g, HxH=%.9g, residual=%.9g",
        components.sigma_a2,
        components.sigma_h2,
        components.sigma_e2,
    )
    boundary = [
        name
        for name, value in zip(("additive", "HxH", "residual"), components.as_array())
        if value <= 1e-10 * (1 + 1e-6)
    ]
    if boundary:
        logger.warning("Variance components at the lower bound: %s", ", ".join(boundary))
    payload = _fit_payload(args=args, selection=selection, inputs=inputs, fit=fit)
    payload["tests"] = {}
    if args.test != "none":
        targets = ("interaction", "joint") if args.test == "both" else (args.test,)
        for target in targets:
            logger.info("Testing %s variance: %s calibration, exact likelihood refits", target, args.pvalue_method)
            result = restricted_likelihood_ratio_test(
                inputs.phenotype,
                additive,
                interaction,
                covariates=inputs.covariates,
                target=target,
                method=args.pvalue_method,
                num_bootstrap=args.bootstrap_replicates,
                seed=args.seed,
                maxiter=args.maxiter,
                projection_batch_size=args.kernel_batch_size,
                logger=logger,
            )
            payload["tests"][target] = result
            if result["success"]:
                logger.info(
                    "%s test: statistic=%.9g, p_value=%.9g (%s)",
                    target,
                    result["statistic"],
                    result["p_value"],
                    result["method"],
                )
            else:
                logger.warning("%s p-value unavailable: %s", target, result["reason"])
    return payload


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--test",
        choices=["none", "interaction", "joint", "both"],
        default="none",
        help="optional variance-component test; interaction tests HxH beyond additive",
    )
    parser.add_argument(
        "--pvalue-method",
        choices=["asymptotic", "bootstrap"],
        default="asymptotic",
        help="asymptotic is an approximate screening test; joint tests require bootstrap",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=199,
        help="number of null simulations/refits for bootstrap p-values (default: 199)",
    )
    parser.add_argument("--covariates", type=Path)
    parser.add_argument(
        "--allow-sample-subset",
        action="store_true",
        help="fit only LinearARG individuals with non-missing phenotype and covariate values",
    )
    parser.add_argument("--covariate-id-column")
    parser.add_argument("--covariate-columns", nargs="+")
    parser.add_argument("--covariate-separator")
    parser.add_argument(
        "--interaction-mode",
        choices=[mode.value for mode in InteractionMode],
        default=InteractionMode.ORDERED_SELF.value,
    )
    parser.add_argument(
        "--kernel-batch-size",
        type=int,
        default=32,
        help="maximum operator right-hand sides per kernel construction batch",
    )
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--initial-sigma-a2", type=float)
    parser.add_argument("--initial-sigma-h2", type=float)
    parser.add_argument("--initial-sigma-e2", type=float)
    parser.add_argument(
        "--likelihood-method",
        choices=["reml", "ml"],
        default="reml",
        help="likelihood criterion (default: reml; ml reproduces the previous criterion)",
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--logdet-probe-mode", choices=["rademacher", "normal", "basis"], default="rademacher")
    parser.add_argument("--num-logdet-probes", type=int, default=16)
    parser.add_argument("--lanczos-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cg-rtol", type=float, default=1e-6)
    parser.add_argument("--cg-atol", type=float, default=0.0)
    parser.add_argument("--cg-maxiter", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="intergalactic")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", help="fit variance components for one LinearARG region")
    fit.add_argument("linear_arg_bundle", type=Path)
    fit.add_argument("--region", help="target region of the form chrom:start-end")
    fit.add_argument("--phenotype", type=Path, required=True)
    fit.add_argument("--phenotype-id-column", default="IID")
    fit.add_argument("--phenotype-column", required=True)
    fit.add_argument("--phenotype-separator")
    _add_model_arguments(fit)
    fit.add_argument("--output", type=Path)
    fit.add_argument("--log-file", type=Path, help="log path (default: OUTPUT.log when --output is given)")
    fit.add_argument("--verbose", action="store_true", help="log every optimizer iteration and debug details")
    fit.set_defaults(func=run_fit)
    from .scan import add_scan_parser

    add_scan_parser(subparsers, _add_model_arguments)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
