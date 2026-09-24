# pattern: Functional Core

import io
import json
import logging
import re

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scipy.sparse.linalg import aslinearoperator, LinearOperator

from intergalactic import cli
from intergalactic.likelihood import VarianceComponents


class _DenseLinearArg(LinearOperator):
    def __init__(self, haplotypes: np.ndarray, iids: list[str]) -> None:
        super().__init__(np.dtype(np.float64), haplotypes.shape)
        self.haplotypes = haplotypes
        self.iids = iids

    def _matvec(self, x):
        return self.haplotypes @ np.asarray(x)

    def _rmatvec(self, x):
        return self.haplotypes.T @ np.asarray(x)

    def _matmat(self, X):
        return self.haplotypes @ np.asarray(X)

    def _rmatmat(self, X):
        return self.haplotypes.T @ np.asarray(X)


@dataclass(frozen=True)
class _FakeFit:
    variance_components: VarianceComponents
    fixed_effects: np.ndarray
    log_likelihood: float
    negative_log_likelihood: float
    success: bool
    message: str
    n_iterations: int
    accepted_steps: int
    rejected_steps: int
    trust_radius: float
    logdet_method: str
    num_logdet_probes: int
    lanczos_rank: int


class _FakeLinearARG:
    shape = (4, 3)
    iids = ["sample_a", "sample_a", "sample_b", "sample_b"]
    variants: pl.DataFrame | pl.LazyFrame = pl.DataFrame({"CHROM": ["1"]}).lazy()

    def __init__(self) -> None:
        self.filtered_bed: pl.DataFrame | None = None

    def filter_variants_by_bed(self, bed):
        self.filtered_bed = bed


@pytest.mark.parametrize("subset", [False, True])
def test_cli_loads_phenotype_and_covariates_by_linear_arg_iids(tmp_path: Path, monkeypatch, subset):
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_path.write_text("iid\ty\nsample_b\t2.5\nsample_a\t1.5\n")
    covariate_path = tmp_path / "covariates.tsv"
    covariate_path.write_text("iid\tage\tpc1\nsample_a\t40\t0.1\nsample_b\t50\t-0.2\n")
    bundle_path = tmp_path / "bundle.h5"
    output_path = tmp_path / "fit.json"
    captured = {}

    def fake_load_linear_arg(path, *, region):
        captured["bundle_path"] = path
        captured["region"] = region
        return cli.LinearArgSelection(
            linear_arg=_DenseLinearArg(
                np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                ["sample_a", "sample_a", "sample_b", "sample_b"],
            )
            if not subset
            else _DenseLinearArg(
                np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0],
                    ]
                ),
                ["sample_a", "sample_a", "sample_b", "sample_b", "absent", "absent"],
            ),
            block_name="chr1:1-10",
            region="chr1:1-10",
        )

    def fake_optimize(y, additive, interaction, **kwargs):
        captured["y"] = y
        captured["covariates"] = kwargs["covariates"]
        captured["additive_shape"] = additive.shape
        captured["interaction_shape"] = interaction.shape
        return _FakeFit(
            variance_components=VarianceComponents(1e-10, 0.5, 0.25),
            fixed_effects=np.array([0.1, 0.2]),
            log_likelihood=-3.0,
            negative_log_likelihood=3.0,
            success=True,
            message="ok",
            n_iterations=2,
            accepted_steps=1,
            rejected_steps=0,
            trust_radius=1.0,
            logdet_method="lanczos_basis",
            num_logdet_probes=2,
            lanczos_rank=2,
        )

    monkeypatch.setattr(cli, "load_linear_arg_selection", fake_load_linear_arg)
    monkeypatch.setattr(cli, "optimize_variance_components", fake_optimize)

    exit_code = cli.main(
        [
            "fit",
            str(bundle_path),
            "--region",
            "chr1:1-10",
            "--phenotype",
            str(phenotype_path),
            "--phenotype-id-column",
            "iid",
            "--phenotype-column",
            "y",
            "--covariates",
            str(covariate_path),
            "--covariate-id-column",
            "iid",
            "--covariate-columns",
            "age",
            "pc1",
            "--logdet-probe-mode",
            "basis",
            "--output",
            str(output_path),
            *(["--allow-sample-subset"] if subset else []),
        ]
    )

    assert exit_code == 0
    assert captured["bundle_path"] == bundle_path
    assert captured["region"] == "chr1:1-10"
    np.testing.assert_allclose(captured["y"], np.array([1.5, 2.5]))
    np.testing.assert_allclose(captured["covariates"], np.array([[40.0, 0.1], [50.0, -0.2]]))
    assert captured["additive_shape"] == (2, 2)
    assert captured["interaction_shape"] == (2, 2)
    assert output_path.exists()
    log_text = Path(f"{output_path}.log").read_text()
    assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - INFO\] Starting intergalactic fit", log_text)
    for stage in (
        "Loading LinearARG",
        "Sample alignment retained",
        "Constructing additive GRM",
        "Preparing HxH kernel",
        "Optimizing variance components",
        "Fit converged",
        "Wrote results",
    ):
        assert stage in log_text
    assert " - DEBUG]" not in log_text
    assert "Variance components at the lower bound: additive" in log_text


def test_cli_rejects_missing_phenotype_rows(tmp_path: Path):
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_path.write_text("iid\ty\nsample_a\t1.5\n")

    sample_ids = ["sample_a", "sample_b"]

    try:
        cli.load_model_inputs(
            phenotype_path=phenotype_path,
            phenotype_id_column="iid",
            phenotype_column="y",
            sample_ids=sample_ids,
        )
    except ValueError as error:
        assert "missing phenotype values" in str(error)
    else:
        raise AssertionError("expected missing phenotype rows to fail")


def test_select_block_name_uses_region_overlap():
    blocks = pl.DataFrame(
        {
            "block_name": ["chr1:1-100", "chr1:100-200"],
            "chrom": ["1", "1"],
            "start": [1, 100],
            "end": [100, 200],
        }
    )

    assert cli.select_block_name(blocks, region="chr1:120-130") == "chr1:100-200"


def test_linear_arg_loader_filters_root_bundle_by_region(tmp_path: Path, monkeypatch):
    fake_linear_arg = _FakeLinearARG()
    captured = {}

    class FakeLinearARGReader:
        @staticmethod
        def read(path, *, block, load_metadata):
            captured["path"] = path
            captured["block"] = block
            captured["load_metadata"] = load_metadata
            return fake_linear_arg

    monkeypatch.setattr(cli, "_import_linear_dag", lambda: (FakeLinearARGReader, lambda _path: None, None))

    selection = cli.load_linear_arg_selection(tmp_path / "root.h5", region="chr1:10-20")

    assert selection.block_name is None
    assert selection.region == "chr1:10-20"
    assert captured["block"] is None
    assert captured["load_metadata"]
    assert fake_linear_arg.filtered_bed is not None
    assert fake_linear_arg.filtered_bed.to_dict(as_series=False) == {
        "chrom": ["1"],
        "chromStart": [10],
        "chromEnd": [20],
    }


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("chromosome", ["1", "chr1"])
def test_loader_matches_uppercase_chromosome_metadata(tmp_path, monkeypatch, lazy, chromosome):
    operator = _FakeLinearARG()
    operator.variants = pl.DataFrame({"CHROM": [chromosome]})
    if lazy:
        operator.variants = operator.variants.lazy()

    class Reader:
        @staticmethod
        def read(*args, **kwargs):
            return operator

    monkeypatch.setattr(cli, "_import_linear_dag", lambda: (Reader, lambda _: None, None))
    cli.load_linear_arg_selection(tmp_path / "test.h5", region="chr1:10-20")
    assert operator.filtered_bed is not None
    assert operator.filtered_bed["chrom"].to_list() == [chromosome]


def test_sample_subset_aligns_inputs_and_both_operator_directions(tmp_path):
    phenotype = tmp_path / "phenotype.tsv"
    phenotype.write_text("IID\ty\nc\t3\na\t1\nb\t2\n")
    covariates = tmp_path / "covariates.tsv"
    covariates.write_text("IID\tage\na\t20\nc\t40\n")
    inputs = cli.load_model_inputs(
        phenotype_path=phenotype,
        phenotype_id_column="IID",
        phenotype_column="y",
        sample_ids=["a", "b", "c", "d"],
        covariate_path=covariates,
        allow_missing_samples=True,
    )
    assert inputs.sample_ids == ["a", "c"]
    np.testing.assert_array_equal(inputs.phenotype, [1, 3])
    np.testing.assert_array_equal(inputs.covariates, [[20], [40]])
    h = np.arange(40.0).reshape(8, 5)
    operator = cli._linear_arg_for_samples(aslinearoperator(h), ["a", "b", "c", "d"], inputs.sample_ids)
    expected = h[[0, 1, 4, 5]]
    np.testing.assert_allclose(operator @ np.ones(5), expected @ np.ones(5))
    np.testing.assert_allclose(operator @ np.eye(5), expected)
    np.testing.assert_allclose(operator.T @ np.ones(4), expected.T @ np.ones(4))
    np.testing.assert_allclose(operator.T @ np.eye(4), expected.T)


def test_sample_subset_rejects_empty_overlap(tmp_path):
    phenotype = tmp_path / "phenotype.tsv"
    phenotype.write_text("IID\ty\nother\t1\n")
    with pytest.raises(ValueError, match="no LinearARG samples"):
        cli.load_model_inputs(
            phenotype_path=phenotype,
            phenotype_id_column="IID",
            phenotype_column="y",
            sample_ids=["a"],
            allow_missing_samples=True,
        )


@pytest.mark.parametrize("family_id", ["0", "family_a"])
def test_automatic_covariates_exclude_family_ids_after_sample_subset(tmp_path, family_id):
    phenotype = tmp_path / "phenotype.tsv"
    phenotype.write_text("IID\ty\na\t1\nb\t2\n")
    covariates = tmp_path / "covariates.tsv"
    covariates.write_text(f"FID\tIID\tPC1\tsex\n{family_id}\tb\t0.2\t0\n{family_id}\ta\t0.1\t1\n")
    inputs = cli.load_model_inputs(
        phenotype_path=phenotype,
        phenotype_id_column="IID",
        phenotype_column="y",
        sample_ids=["a", "b", "absent"],
        covariate_path=covariates,
        allow_missing_samples=True,
    )
    assert inputs.sample_ids == ["a", "b"]
    assert inputs.covariate_names == ["PC1", "sex"]
    assert inputs.covariates is not None
    np.testing.assert_allclose(inputs.covariates, [[0.1, 1.0], [0.2, 0.0]])
    assert np.linalg.matrix_rank(inputs.covariates) == 2


def test_covariate_identifier_defaults_and_explicit_selection():
    raw = pl.DataFrame({"sample": ["a"], "FID": [0], "IID": ["a"], "age": [20]}).lazy()
    assert cli._covariate_columns(raw, id_column="sample", requested_columns=None) == ["age"]
    assert cli._covariate_columns(raw, id_column="sample", requested_columns=["FID", "age"]) == ["FID", "age"]


@pytest.mark.parametrize("verbose", [False, True])
def test_cli_log_destinations_and_repeated_invocations(tmp_path, monkeypatch, verbose):
    root_handlers = list(logging.getLogger().handlers)
    seen_loggers = []

    def fake_run(args, logger):
        seen_loggers.append(logger)
        logger.debug("Debug progress")
        logger.info("Fit progress")
        cli._write_payload({"success": True}, args.output)
        return 0

    monkeypatch.setattr(cli, "_run_fit", fake_run)
    for invocation in range(2):
        path = tmp_path / f"run{invocation}.log"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = cli.main(
                [
                    "fit",
                    "bundle.h5",
                    "--phenotype",
                    "y.tsv",
                    "--phenotype-column",
                    "y",
                    "--log-file",
                    str(path),
                    *(["--verbose"] if verbose else []),
                ]
            )
        assert result == 0
        assert json.loads(stdout.getvalue()) == {"success": True}
        assert stderr.getvalue().count("Starting intergalactic fit") == 1
        assert path.read_text() == stderr.getvalue()
        assert ("Debug progress" in path.read_text()) == verbose
        assert not seen_loggers[-1].handlers
    assert logging.getLogger().handlers == root_handlers


@pytest.mark.parametrize("verbose", [False, True])
def test_cli_logs_failures_and_returns_nonzero_without_result(tmp_path, monkeypatch, verbose):
    def fail(*args, **kwargs):
        raise ValueError("bad covariate design")

    monkeypatch.setattr(cli, "load_linear_arg_selection", fail)
    output = tmp_path / "fit.json"
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = cli.main(
            [
                "fit",
                "bundle.h5",
                "--phenotype",
                "y.tsv",
                "--phenotype-column",
                "y",
                "--output",
                str(output),
                *(["--verbose"] if verbose else []),
            ]
        )
    assert status == 1
    assert stdout.getvalue() == ""
    assert not output.exists()
    log = Path(f"{output}.log").read_text()
    assert " - ERROR] Fit failed" in log
    assert "bad covariate design" in log
    assert ("Traceback" in log) == verbose
    assert "Finished in" not in log


def test_cli_rejects_result_log_path_collision(tmp_path):
    path = tmp_path / "fit.json"
    path.write_text("existing result")
    with pytest.raises(ValueError, match="different paths"):
        cli.main(
            [
                "fit",
                "bundle.h5",
                "--phenotype",
                "y.tsv",
                "--phenotype-column",
                "y",
                "--output",
                str(path),
                "--log-file",
                str(path),
            ]
        )
    assert path.read_text() == "existing result"
