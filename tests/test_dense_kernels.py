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


@pytest.mark.parametrize("mode", list(InteractionMode))
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("normalization", [None, True, DiagonalNormalizer()])
def test_batched_interaction_matches_explicit_pair_features(mode, center, normalization):
    from intergalactic.operators import _CallableLinearOperator

    rng = np.random.default_rng(123)
    haplotypes = rng.integers(0, 2, size=(10, 17)).astype(float)
    calls = []

    def forward(values):
        assert values.shape[1] <= 2
        calls.append(values.shape)
        return haplotypes @ values

    def reverse(values):
        assert values.shape[1] <= 2
        calls.append(values.shape)
        return haplotypes.T @ values

    operator = _CallableLinearOperator(
        shape=haplotypes.shape,
        dtype=float,
        matvec=lambda v: haplotypes @ v,
        matmat=forward,
        rmatvec=lambda v: haplotypes.T @ v,
        rmatmat=reverse,
    )
    blocks = [np.array([0, 3, 6, 10, 16]), np.array([2, 4, 5, 9])]
    pairing = _pairing_matrix(5)
    expected = np.zeros((5, 5))
    for indices in blocks:
        block = haplotypes[:, indices]
        if center:
            block = block - block.mean(axis=0)
        pairs = [
            (a, b)
            for a in range(block.shape[1])
            for b in range(block.shape[1])
            if mode is InteractionMode.ORDERED_SELF or a < b
        ]
        features = np.column_stack([block[:, a] * block[:, b] for a, b in pairs])
        features = pairing @ features
        expected += features @ features.T
    if normalization is True:
        expected /= np.trace(expected) / 5
    elif isinstance(normalization, DiagonalNormalizer):
        expected /= np.sqrt(np.outer(expected.diagonal(), expected.diagonal()))
    kernel = SameHaplotypeInteractionKernel(
        operator,
        DiploidHaplotypeMap(5),
        blocks=blocks,
        interaction_mode=mode,
        center=center,
        normalization=normalization,
        batch_size=2,
    )
    np.testing.assert_allclose(kernel.matmat(np.eye(5)), expected, atol=1e-12)
    count = len(calls)
    np.testing.assert_allclose(kernel.matvec(np.ones(5)), expected @ np.ones(5), atol=1e-12)
    assert len(calls) == count  # Repeated likelihood products reuse the cached kernel.


@pytest.mark.parametrize("center", [False, True])
def test_additive_normalization_uses_bounded_operator_batches(center):
    from intergalactic.operators import _CallableLinearOperator

    haplotypes = np.random.default_rng(7).normal(size=(10, 101))

    def reverse(values):
        assert values.shape[1] <= 2
        return haplotypes.T @ values

    operator = _CallableLinearOperator(
        shape=haplotypes.shape,
        dtype=float,
        matvec=lambda v: haplotypes @ v,
        matmat=lambda v: haplotypes @ v,
        rmatvec=lambda v: haplotypes.T @ v,
        rmatmat=reverse,
    )
    kernel = AdditiveHaplotypeKernel(operator, DiploidHaplotypeMap(5), center=center, batch_size=2)
    h = haplotypes - haplotypes.mean(axis=0) if center else haplotypes
    genotypes = _pairing_matrix(5) @ h
    expected = genotypes @ genotypes.T
    expected /= np.trace(expected) / 5
    np.testing.assert_allclose(kernel.matvec(np.arange(5.0)), expected @ np.arange(5.0), atol=1e-12)


@pytest.mark.parametrize("mode", list(InteractionMode))
def test_dense_kernel_utility_matches_explicit_centered_features(mode):
    from intergalactic.kernels import same_haplotype_apply

    h = _haplotypes()
    h -= h.mean(axis=0)
    pairs = [(a, b) for a in range(4) for b in range(4) if mode is InteractionMode.ORDERED_SELF or a < b]
    features = np.column_stack([h[:, a] * h[:, b] for a, b in pairs])
    weights = np.random.default_rng(42).normal(size=(6, 2))
    np.testing.assert_allclose(
        same_haplotype_apply(h, weights, interaction_mode=mode),
        features @ features.T @ weights,
        atol=1e-12,
    )


@pytest.mark.parametrize("kernel_type", [AdditiveHaplotypeKernel, SameHaplotypeInteractionKernel])
@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_invalid_kernel_batch_size_is_rejected(kernel_type, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        kernel_type(aslinearoperator(_haplotypes()), DiploidHaplotypeMap(3), batch_size=batch_size)


@pytest.mark.parametrize("center", [False, True])
def test_integer_haplotypes_and_repeated_variant_indices(center):
    h = _haplotypes().astype(np.int8)
    indices = np.array([2, 0, 2, 3])
    kernel = SameHaplotypeInteractionKernel(
        aslinearoperator(h),
        DiploidHaplotypeMap(3),
        variant_indices=indices,
        normalization=None,
        center=center,
        batch_size=2,
    )
    selected = h[:, indices].astype(float)
    if center:
        selected -= selected.mean(axis=0)
    expected = _ordered_same_haplotype_kernel(selected)
    np.testing.assert_allclose(kernel.matmat(np.eye(3)), expected, atol=1e-6)
