# pattern: Imperative Shell

from __future__ import annotations

import argparse
import importlib
import json

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from .kernels import validate_haplotype_count
from .likelihood import optimize_variance_components, VarianceComponents
from .operators import AdditiveHaplotypeKernel, DiploidHaplotypeMap, InteractionMode, SameHaplotypeInteractionKernel


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
    return [column for column in raw.collect_schema().names() if column != id_column]


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
) -> ModelInputs:
    """Load and align phenotype/covariate tables to LinearARG individual IDs."""
    ordered_samples = pl.DataFrame(
        {"sample_id": [str(sample_id) for sample_id in sample_ids], "_row_order": np.arange(len(sample_ids))}
    ).lazy()
    phenotype = _phenotype_frame(
        phenotype_path,
        id_column=phenotype_id_column,
        phenotype_column=phenotype_column,
        separator=phenotype_separator,
    )
    joined = ordered_samples.join(phenotype, on="sample_id", how="left", validate="1:1")
    covariate_names: list[str] = []
    if covariate_path is not None:
        covariates, covariate_names = _covariate_frame(
            covariate_path,
            id_column=covariate_id_column or phenotype_id_column,
            covariate_columns=covariate_columns,
            separator=covariate_separator,
        )
        joined = joined.join(covariates, on="sample_id", how="left", validate="1:1")

    materialized = joined.sort("_row_order").collect()
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


def _region_bed(region: str) -> pl.DataFrame:
    parsed = parse_region(region)
    return pl.DataFrame(
        {"chrom": [parsed.chrom], "chromStart": [parsed.start], "chromEnd": [parsed.end]},
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
    """Load a LinearARG block and optionally filter variants to a region."""
    LinearARG, list_blocks, LinearARGZarrReader = _import_linear_dag()
    if path.is_dir():
        if LinearARGZarrReader is None:
            raise RuntimeError("linear_dag.core.zarr_io.LinearARGZarrReader is unavailable")
        reader = LinearARGZarrReader.open(path)
        blocks = reader.list_blocks()
        block_name = select_block_name(blocks, region=region)
        if block_name is None:
            raise ValueError("Zarr LinearARG bundles must contain at least one block")
        linear_arg = reader.read_block(block_name, load_metadata=region is not None)
    else:
        blocks = list_blocks(path)
        block_name = select_block_name(blocks, region=region)
        linear_arg = LinearARG.read(path, block=block_name, load_metadata=region is not None)

    if region is not None and getattr(linear_arg, "variants", None) is None and block_name != region:
        raise ValueError("region filtering requires variant metadata on the selected LinearARG")
    if region is not None and getattr(linear_arg, "variants", None) is not None:
        linear_arg.filter_variants_by_bed(_region_bed(region))
    return LinearArgSelection(linear_arg=linear_arg, block_name=block_name, region=region)


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


def run_fit(args: argparse.Namespace) -> int:
    selection = load_linear_arg_selection(args.linear_arg_bundle, region=args.region)
    sample_ids = individual_ids_from_linear_arg(selection.linear_arg)
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
    )
    diploid_map = DiploidHaplotypeMap.from_haplotypes(selection.linear_arg.shape[0])
    normalization = not args.no_normalize
    additive = AdditiveHaplotypeKernel(
        selection.linear_arg,
        diploid_map,
        normalization=normalization,
        center=args.center,
    )
    interaction = SameHaplotypeInteractionKernel(
        selection.linear_arg,
        diploid_map,
        interaction_mode=args.interaction_mode,
        normalization=normalization,
        center=args.center,
    )
    fit = optimize_variance_components(
        inputs.phenotype,
        additive,
        interaction,
        initial=_initial_components(args),
        covariates=inputs.covariates,
        maxiter=args.maxiter,
        logdet_probe_mode=args.logdet_probe_mode,
        num_logdet_probes=args.num_logdet_probes,
        lanczos_rank=args.lanczos_rank,
        seed=args.seed,
        cg_rtol=args.cg_rtol,
        cg_atol=args.cg_atol,
        cg_maxiter=args.cg_maxiter,
    )
    _write_payload(_fit_payload(args=args, selection=selection, inputs=inputs, fit=fit), args.output)
    return 0


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
    fit.add_argument("--covariates", type=Path)
    fit.add_argument("--covariate-id-column")
    fit.add_argument("--covariate-columns", nargs="+")
    fit.add_argument("--covariate-separator")
    fit.add_argument(
        "--interaction-mode",
        choices=[mode.value for mode in InteractionMode],
        default=InteractionMode.ORDERED_SELF.value,
    )
    fit.add_argument("--center", action="store_true")
    fit.add_argument("--no-normalize", action="store_true")
    fit.add_argument("--initial-sigma-a2", type=float)
    fit.add_argument("--initial-sigma-h2", type=float)
    fit.add_argument("--initial-sigma-e2", type=float)
    fit.add_argument("--maxiter", type=int, default=1000)
    fit.add_argument("--logdet-probe-mode", choices=["rademacher", "normal", "basis"], default="rademacher")
    fit.add_argument("--num-logdet-probes", type=int, default=16)
    fit.add_argument("--lanczos-rank", type=int, default=32)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--cg-rtol", type=float, default=1e-6)
    fit.add_argument("--cg-atol", type=float, default=0.0)
    fit.add_argument("--cg-maxiter", type=int)
    fit.add_argument("--output", type=Path)
    fit.set_defaults(func=run_fit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
