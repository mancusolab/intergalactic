# pattern: Functional Core

from __future__ import annotations

from enum import StrEnum

import numpy as np
import numpy.typing as npt


class InteractionMode(StrEnum):
    """Supported same-haplotype interaction feature conventions."""

    ORDERED_SELF = "ordered_self"
    UNORDERED_OFFDIAG = "unordered_offdiag"


def normalize_interaction_mode(mode: InteractionMode | str) -> InteractionMode:
    """Return a validated [`intergalactic.kernels.InteractionMode`][] value.

    **Arguments:**

    - `mode`: Interaction mode enum or string.

    **Returns:**

    - Normalized interaction mode.

    **Raises:**

    - `ValueError`: If `mode` is unknown.
    """
    try:
        return InteractionMode(mode)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in InteractionMode)
        raise ValueError(f"interaction_mode must be one of {allowed}; got {mode!r}") from exc


def validate_haplotype_count(two_n: int) -> int:
    """Validate an adjacent-pair haplotype count and return the individual count.

    **Arguments:**

    - `two_n`: Number of stacked haplotype rows.

    **Returns:**

    - Number of diploid individuals.

    **Raises:**

    - `ValueError`: If `two_n` is not positive and even.
    """
    if two_n <= 0:
        raise ValueError("number of haplotypes must be positive")
    if two_n % 2 != 0:
        raise ValueError("number of haplotypes must be even for adjacent diploid pairing")
    return two_n // 2


def as_column_matrix(values: npt.ArrayLike, *, expected_rows: int, name: str) -> tuple[npt.NDArray[np.floating], bool]:
    """Convert a vector or column-major matrix input into a two-dimensional array.

    **Arguments:**

    - `values`: Input vector or matrix.
    - `expected_rows`: Required leading dimension.
    - `name`: Name used in error messages.

    **Returns:**

    - Tuple `(matrix, was_vector)`.

    **Raises:**

    - `ValueError`: If `values` is not one- or two-dimensional with the required row count.
    """
    array = np.asarray(values)
    if array.ndim == 1:
        if array.shape[0] != expected_rows:
            raise ValueError(f"{name} must have length {expected_rows}; got {array.shape[0]}")
        return array.reshape(expected_rows, 1), True
    if array.ndim == 2:
        if array.shape[0] != expected_rows:
            raise ValueError(f"{name} must have {expected_rows} rows; got {array.shape[0]}")
        return array, False
    raise ValueError(f"{name} must be one- or two-dimensional")


def adjacent_haplotype_from_individual(values: npt.ArrayLike, *, n_individuals: int) -> npt.NDArray[np.number]:
    """Lift diploid rows to adjacent maternal/paternal haplotype rows.

    **Arguments:**

    - `values`: Individual-level vector or matrix with `n_individuals` rows.
    - `n_individuals`: Number of diploid individuals.

    **Returns:**

    - Haplotype-level values with adjacent duplicate rows for each individual.
    """
    matrix, was_vector = as_column_matrix(values, expected_rows=n_individuals, name="individual values")
    lifted = np.repeat(matrix, 2, axis=0)
    return lifted.ravel() if was_vector else lifted


def adjacent_individual_from_haplotype(values: npt.ArrayLike, *, n_individuals: int) -> npt.NDArray[np.number]:
    """Collapse adjacent maternal/paternal haplotype rows to diploid rows.

    **Arguments:**

    - `values`: Haplotype-level vector or matrix with `2 * n_individuals` rows.
    - `n_individuals`: Number of diploid individuals.

    **Returns:**

    - Individual-level values formed by summing adjacent haplotype rows.
    """
    matrix, was_vector = as_column_matrix(values, expected_rows=2 * n_individuals, name="haplotype values")
    collapsed = matrix.reshape(n_individuals, 2, matrix.shape[1]).sum(axis=1)
    return collapsed.ravel() if was_vector else collapsed


def ordered_same_haplotype_apply(
    haplotypes: npt.NDArray[np.floating],
    haplotype_weights: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Apply $\\Phi(H)\\Phi(H)^T$ without materializing $\\Phi(H)$.

    **Arguments:**

    - `haplotypes`: Materialized local haplotype block $H_b$ with shape `(2n, p_b)`.
    - `haplotype_weights`: Haplotype-level weights with shape `(2n, k)`.

    **Returns:**

    - Haplotype-level products with shape `(2n, k)`.
    """
    result = np.empty_like(haplotype_weights, dtype=np.result_type(haplotypes.dtype, haplotype_weights.dtype))
    for column_index in range(haplotype_weights.shape[1]):
        weights = haplotype_weights[:, column_index]
        # B = H.T @ diag(w) @ H, formed for one weight vector and local block.
        co_carriage = haplotypes.T @ (haplotypes * weights[:, None])
        result[:, column_index] = np.einsum("ij,jk,ik->i", haplotypes, co_carriage, haplotypes, optimize=True)
    return result


def offdiag_same_haplotype_apply(
    haplotypes: npt.NDArray[np.floating],
    haplotype_weights: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Apply the unordered off-diagonal same-haplotype interaction kernel.

    This computes $0.5[(HH^T)^2 - HH^T]w$ in feature space without materializing
    either the kernel matrix or the pair-feature matrix.

    **Arguments:**

    - `haplotypes`: Materialized local haplotype block $H_b$ with shape `(2n, p_b)`.
    - `haplotype_weights`: Haplotype-level weights with shape `(2n, k)`.

    **Returns:**

    - Haplotype-level products with shape `(2n, k)`.
    """
    result = np.empty_like(haplotype_weights, dtype=np.result_type(haplotypes.dtype, haplotype_weights.dtype))
    for column_index in range(haplotype_weights.shape[1]):
        weights = haplotype_weights[:, column_index]
        co_carriage = haplotypes.T @ (haplotypes * weights[:, None])
        ordered = np.einsum("ij,jk,ik->i", haplotypes, co_carriage, haplotypes, optimize=True)
        self_pairs = haplotypes @ co_carriage.diagonal()
        result[:, column_index] = 0.5 * (ordered - self_pairs)
    return result


def same_haplotype_apply(
    haplotypes: npt.NDArray[np.floating],
    haplotype_weights: npt.NDArray[np.floating],
    *,
    interaction_mode: InteractionMode | str,
) -> npt.NDArray[np.floating]:
    """Apply a same-haplotype interaction kernel for one materialized window.

    **Arguments:**

    - `haplotypes`: Materialized local haplotype block.
    - `haplotype_weights`: Haplotype-level weights.
    - `interaction_mode`: Ordered/self or unordered/off-diagonal interaction convention.

    **Returns:**

    - Haplotype-level products.
    """
    mode = normalize_interaction_mode(interaction_mode)
    if mode is InteractionMode.ORDERED_SELF:
        return ordered_same_haplotype_apply(haplotypes, haplotype_weights)
    return offdiag_same_haplotype_apply(haplotypes, haplotype_weights)
