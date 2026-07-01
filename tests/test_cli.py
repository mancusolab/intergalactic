# pattern: Functional Core

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from scipy.sparse.linalg import LinearOperator

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
    variants = object()

    def __init__(self) -> None:
        self.filtered_bed = None

    def filter_variants_by_bed(self, bed):
        self.filtered_bed = bed


def test_cli_loads_phenotype_and_covariates_by_linear_arg_iids(tmp_path: Path, monkeypatch):
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
            variance_components=VarianceComponents(1.0, 0.5, 0.25),
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
