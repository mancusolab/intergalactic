# pattern: Functional Core

import numpy as np
import pytest

from scipy.sparse.linalg import aslinearoperator

from intergalactic import (
    AdditiveHaplotypeKernel,
    DiagonalNormalizer,
    DiploidHaplotypeMap,
    InteractionMode,
    SameHaplotypeInteractionKernel,
    TraceNormalizer,
    VarianceComponentKernel,
)


def _haplotypes() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _pairing_matrix(n_individuals: int) -> np.ndarray:
    pairing = np.zeros((n_individuals, 2 * n_individuals), dtype=np.float64)
    for individual in range(n_individuals):
        pairing[individual, 2 * individual : 2 * individual + 2] = 1.0
    return pairing


def _ordered_same_haplotype_kernel(haplotypes: np.ndarray) -> np.ndarray:
    pairing = _pairing_matrix(haplotypes.shape[0] // 2)
    haplotype_gram = haplotypes @ haplotypes.T
    return pairing @ (haplotype_gram * haplotype_gram) @ pairing.T


def _offdiag_same_haplotype_kernel(haplotypes: np.ndarray) -> np.ndarray:
    pairing = _pairing_matrix(haplotypes.shape[0] // 2)
    haplotype_gram = haplotypes @ haplotypes.T
    return pairing @ (0.5 * (haplotype_gram * haplotype_gram - haplotype_gram)) @ pairing.T


def test_diploid_haplotype_map_uses_adjacent_maternal_paternal_ordering():
    diploid_map = DiploidHaplotypeMap(3)

    np.testing.assert_array_equal(
        diploid_map.haploid_from_diploid(np.array([2.0, -1.0, 3.5])), [2.0, 2.0, -1.0, -1.0, 3.5, 3.5]
    )
    np.testing.assert_array_equal(
        diploid_map.diploid_from_haploid(np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])), [3.0, 12.0, 48.0]
    )


def test_additive_haplotype_kernel_matches_dense_construction():
    haplotypes = _haplotypes()
    diploid_map = DiploidHaplotypeMap.from_haplotypes(haplotypes.shape[0])
    kernel = AdditiveHaplotypeKernel(aslinearoperator(haplotypes), diploid_map, normalization=None)

    expected = _pairing_matrix(3) @ haplotypes @ haplotypes.T @ _pairing_matrix(3).T
    vector = np.array([0.25, -1.0, 2.0])
    matrix = np.column_stack([vector, np.array([1.0, 0.0, -0.5])])

    np.testing.assert_allclose(kernel.matvec(vector), expected @ vector)
    np.testing.assert_allclose(kernel.matmat(matrix), expected @ matrix)
    assert kernel.as_linear_operator().shape == (3, 3)


@pytest.mark.parametrize(
    ("mode", "dense_builder"),
    [
        (InteractionMode.ORDERED_SELF, _ordered_same_haplotype_kernel),
        (InteractionMode.UNORDERED_OFFDIAG, _offdiag_same_haplotype_kernel),
    ],
)
def test_same_haplotype_interaction_kernel_matches_dense_construction(mode, dense_builder):
    haplotypes = _haplotypes()
    diploid_map = DiploidHaplotypeMap.from_haplotypes(haplotypes.shape[0])
    kernel = SameHaplotypeInteractionKernel(
        aslinearoperator(haplotypes),
        diploid_map,
        interaction_mode=mode,
        normalization=None,
    )

    vector = np.array([1.5, -0.5, 0.75])
    matrix = np.column_stack([vector, np.array([-1.0, 2.0, 0.25])])
    expected = dense_builder(haplotypes)

    np.testing.assert_allclose(kernel.matvec(vector), expected @ vector)
    np.testing.assert_allclose(kernel.matmat(matrix), expected @ matrix)


def test_same_haplotype_interaction_kernel_differs_from_genotype_gxg():
    haplotypes = _haplotypes()
    same_haplotype = _ordered_same_haplotype_kernel(haplotypes)
    genotypes = _pairing_matrix(3) @ haplotypes
    genotype_gxg = (genotypes @ genotypes.T) ** 2

    assert not np.allclose(same_haplotype, genotype_gxg)


def test_block_local_kernel_equals_sum_of_dense_block_kernels():
    haplotypes = _haplotypes()
    blocks = [np.array([0, 2]), np.array([1, 3])]
    diploid_map = DiploidHaplotypeMap.from_haplotypes(haplotypes.shape[0])
    kernel = SameHaplotypeInteractionKernel(
        aslinearoperator(haplotypes),
        diploid_map,
        blocks=blocks,
        interaction_mode=InteractionMode.ORDERED_SELF,
        normalization=None,
    )

    expected = sum(_ordered_same_haplotype_kernel(haplotypes[:, block]) for block in blocks)
    vector = np.array([0.5, 1.25, -2.0])

    np.testing.assert_allclose(kernel.matvec(vector), expected @ vector)


def test_trace_and_diagonal_normalization_match_dense_reference():
    haplotypes = _haplotypes()
    diploid_map = DiploidHaplotypeMap.from_haplotypes(haplotypes.shape[0])
    vector = np.array([1.0, -2.0, 0.5])

    trace_kernel = SameHaplotypeInteractionKernel(
        aslinearoperator(haplotypes),
        diploid_map,
        interaction_mode=InteractionMode.ORDERED_SELF,
        normalization=TraceNormalizer(),
    )
    dense = _ordered_same_haplotype_kernel(haplotypes)
    expected_trace = dense / (np.trace(dense) / dense.shape[0])
    np.testing.assert_allclose(trace_kernel.matvec(vector), expected_trace @ vector)

    diagonal_kernel = SameHaplotypeInteractionKernel(
        aslinearoperator(haplotypes),
        diploid_map,
        interaction_mode=InteractionMode.ORDERED_SELF,
        normalization=DiagonalNormalizer(),
    )
    scale = np.sqrt(np.diag(dense))
    expected_diagonal = dense / np.outer(scale, scale)
    np.testing.assert_allclose(diagonal_kernel.matvec(vector), expected_diagonal @ vector)


def test_variance_component_kernel_combines_components_and_residual_noise():
    haplotypes = _haplotypes()
    diploid_map = DiploidHaplotypeMap.from_haplotypes(haplotypes.shape[0])
    additive = AdditiveHaplotypeKernel(aslinearoperator(haplotypes), diploid_map, normalization=None)
    interaction = SameHaplotypeInteractionKernel(aslinearoperator(haplotypes), diploid_map, normalization=None)
    combined = VarianceComponentKernel(additive, interaction)

    vector = np.array([0.2, -0.4, 0.6])
    expected = 1.5 * additive.matvec(vector) + 0.25 * interaction.matvec(vector) + 2.0 * vector

    np.testing.assert_allclose(combined.matvec(vector, sigma_a2=1.5, sigma_h2=0.25, sigma_e2=2.0), expected)


def test_unnormalized_kernels_preserve_float32_dtype():
    haplotypes = _haplotypes().astype(np.float32)
    vector = np.array([0.2, -0.4, 0.6], dtype=np.float32)
    diploid_map = DiploidHaplotypeMap.from_haplotypes(haplotypes.shape[0])

    additive = AdditiveHaplotypeKernel(aslinearoperator(haplotypes), diploid_map, normalization=None)
    interaction = SameHaplotypeInteractionKernel(aslinearoperator(haplotypes), diploid_map, normalization=None)

    assert additive.matvec(vector).dtype == np.float32
    assert interaction.matvec(vector).dtype == np.float32
