# pattern: Functional Core

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt

from scipy.sparse.linalg import aslinearoperator, LinearOperator

from .kernels import (
    adjacent_haplotype_from_individual,
    adjacent_individual_from_haplotype,
    as_column_matrix,
    InteractionMode,
    normalize_interaction_mode,
    validate_haplotype_count,
)


class KernelOperator(Protocol):
    """Protocol for kernels used by [`intergalactic.operators.VarianceComponentKernel`][]."""

    shape: tuple[int, int]

    def matvec(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        """Apply the kernel to a vector."""

    def matmat(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        """Apply the kernel to one or more vectors."""


@dataclass(frozen=True)
class TraceNormalizer:
    """Normalize a kernel so $\\operatorname{tr}(K) / n = 1$."""


@dataclass(frozen=True)
class DiagonalNormalizer:
    """Normalize a kernel to unit diagonal with $D^{-1/2}KD^{-1/2}$."""


Normalization = TraceNormalizer | DiagonalNormalizer | bool | None


class _CallableLinearOperator(LinearOperator):
    def __init__(
        self,
        *,
        shape: tuple[int, int],
        dtype: npt.DTypeLike = np.float64,
        matvec: Callable[[npt.ArrayLike], npt.NDArray[np.number]],
        matmat: Callable[[npt.ArrayLike], npt.NDArray[np.number]],
        rmatvec: Callable[[npt.ArrayLike], npt.NDArray[np.number]] | None = None,
        rmatmat: Callable[[npt.ArrayLike], npt.NDArray[np.number]] | None = None,
    ) -> None:
        super().__init__(np.dtype(dtype), shape)
        self._matvec_callable = matvec
        self._matmat_callable = matmat
        self._rmatvec_callable = rmatvec
        self._rmatmat_callable = rmatmat

    def _matvec(self, x: npt.ArrayLike) -> npt.NDArray[np.number]:
        return self._matvec_callable(x)

    def _matmat(self, X: npt.ArrayLike) -> npt.NDArray[np.number]:
        return self._matmat_callable(X)

    def _rmatvec(self, x: npt.ArrayLike) -> npt.NDArray[np.number]:
        if self._rmatvec_callable is None:
            return super()._rmatvec(x)
        return self._rmatvec_callable(x)

    def _rmatmat(self, X: npt.ArrayLike) -> npt.NDArray[np.number]:
        if self._rmatmat_callable is None:
            return super()._rmatmat(X)
        return self._rmatmat_callable(X)


@dataclass(frozen=True)
class DiploidHaplotypeMap:
    """Map between adjacent haplotype rows and diploid individual rows.

    The ordering is explicit: rows `(0, 1)` are the two haplotypes for
    individual 0, rows `(2, 3)` are the two haplotypes for individual 1, and so
    on. The first row in each pair may be interpreted as maternal and the second
    as paternal when the upstream haplotype source uses that convention.

    !!! Example

        ```python
        diploid_map = DiploidHaplotypeMap(2)
        diploid_map.haploid_from_diploid([3.0, 5.0])
        # array([3., 3., 5., 5.])
        ```
    """

    n_individuals: int
    haplotype_order: str = "adjacent_maternal_paternal"

    def __post_init__(self) -> None:
        if self.n_individuals <= 0:
            raise ValueError("n_individuals must be positive")
        if self.haplotype_order != "adjacent_maternal_paternal":
            raise ValueError("only adjacent_maternal_paternal haplotype ordering is supported")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the combiner shape `(n_individuals, 2 * n_individuals)`."""
        return (self.n_individuals, 2 * self.n_individuals)

    @property
    def T(self) -> TransposedDiploidHaplotypeMap:
        """Return the transpose map for `C.T @ v` lifting."""
        return TransposedDiploidHaplotypeMap(self)

    @classmethod
    def from_haplotypes(cls, two_n: int) -> DiploidHaplotypeMap:
        """Construct a map from a stacked haplotype row count.

        **Arguments:**

        - `two_n`: Number of stacked haplotype rows.

        **Returns:**

        - A diploid combiner for adjacent haplotype pairs.
        """
        return cls(validate_haplotype_count(two_n))

    def haploid_from_diploid(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        """Lift individual-level values with $C^T$.

        **Arguments:**

        - `values`: Vector or matrix with one row per diploid individual.

        **Returns:**

        - Haplotype-level vector or matrix with duplicated adjacent rows.
        """
        return adjacent_haplotype_from_individual(values, n_individuals=self.n_individuals)

    def diploid_from_haploid(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        """Collapse haplotype-level values with $C$.

        **Arguments:**

        - `values`: Vector or matrix with one row per haplotype.

        **Returns:**

        - Individual-level vector or matrix formed by summing adjacent pairs.
        """
        return adjacent_individual_from_haplotype(values, n_individuals=self.n_individuals)

    def as_linear_operator(self) -> LinearOperator:
        """Return `C` as a SciPy [`scipy.sparse.linalg.LinearOperator`][].

        **Returns:**

        - Linear operator mapping haplotype rows to diploid rows.
        """
        return _CallableLinearOperator(
            shape=self.shape,
            matvec=self.diploid_from_haploid,
            matmat=self.diploid_from_haploid,
            rmatvec=self.haploid_from_diploid,
            rmatmat=self.haploid_from_diploid,
        )

    def __matmul__(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        return self.diploid_from_haploid(values)


@dataclass(frozen=True)
class TransposedDiploidHaplotypeMap:
    """Transpose view of [`intergalactic.operators.DiploidHaplotypeMap`][]."""

    diploid_map: DiploidHaplotypeMap

    @property
    def shape(self) -> tuple[int, int]:
        """Return the transpose-combiner shape `(2 * n_individuals, n_individuals)`."""
        n_individuals, two_n = self.diploid_map.shape
        return (two_n, n_individuals)

    def __matmul__(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        return self.diploid_map.haploid_from_diploid(values)


class _BaseKernel:
    linear_arg: LinearOperator
    diploid_map: DiploidHaplotypeMap
    shape: tuple[int, int]
    _trace_scale: float
    _diagonal_scale: npt.NDArray[np.float64] | None

    def _raw_matmat(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        raise NotImplementedError

    def _raw_diagonal(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    def _normalization_vectors(self, normalization: Normalization) -> tuple[float, npt.NDArray[np.float64] | None]:
        if normalization is True:
            normalization = TraceNormalizer()
        if normalization in (None, False):
            return 1.0, None
        diagonal = np.asarray(self._raw_diagonal(), dtype=np.float64)
        if isinstance(normalization, TraceNormalizer):
            trace_scale = float(diagonal.sum() / diagonal.shape[0])
            if trace_scale <= 0.0:
                raise ValueError("cannot trace-normalize a kernel with non-positive trace")
            return trace_scale, None
        if isinstance(normalization, DiagonalNormalizer):
            if np.any(diagonal <= 0.0):
                raise ValueError("cannot diagonal-normalize a kernel with non-positive diagonal entries")
            return 1.0, 1.0 / np.sqrt(diagonal)
        raise TypeError(f"unsupported normalization: {normalization!r}")

    def _apply_normalized(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        matrix, was_vector = as_column_matrix(values, expected_rows=self.shape[1], name="kernel input")
        working = matrix if self._diagonal_scale is None else self._diagonal_scale[:, None] * matrix
        result = self._raw_matmat(working) / self._trace_scale
        if self._diagonal_scale is not None:
            result = self._diagonal_scale[:, None] * result
        return result.ravel() if was_vector else result

    def matvec(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        """Apply the kernel to a vector.

        **Arguments:**

        - `values`: Individual-level vector.

        **Returns:**

        - Individual-level kernel-vector product.
        """
        result = self._apply_normalized(values)
        if np.asarray(result).ndim != 1:
            return np.asarray(result).reshape(self.shape[0])
        return result

    def matmat(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        """Apply the kernel to one or more vectors.

        **Arguments:**

        - `values`: Individual-level vector or matrix.

        **Returns:**

        - Individual-level kernel product with matching matrix shape.
        """
        result = self._apply_normalized(values)
        if np.asarray(values).ndim == 1:
            return np.asarray(result).reshape(self.shape[0], 1)
        return result

    def as_linear_operator(self) -> LinearOperator:
        """Return this kernel as a SciPy [`scipy.sparse.linalg.LinearOperator`][].

        **Returns:**

        - Linear operator over diploid individual space.
        """
        return _CallableLinearOperator(shape=self.shape, matvec=self.matvec, matmat=self.matmat)


class AdditiveHaplotypeKernel(_BaseKernel):
    """Additive same-window haplotype kernel $K_A = C H H^T C^T$.

    This operator composes a `LinearARG`-compatible haplotype operator with an
    adjacent diploid combiner. It does not materialize the dense haplotype
    matrix for unnormalized products.
    """

    def __init__(
        self,
        linear_arg: LinearOperator,
        diploid_map: DiploidHaplotypeMap,
        *,
        normalization: Normalization = True,
        center: bool = False,
        batch_size: int = 32,
    ) -> None:
        self.linear_arg = aslinearoperator(linear_arg)
        self.diploid_map = diploid_map
        self.center = center
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.shape = (diploid_map.n_individuals, diploid_map.n_individuals)
        if self.linear_arg.shape[0] != 2 * diploid_map.n_individuals:
            raise ValueError("linear_arg row count must equal 2 * diploid_map.n_individuals")
        self._column_means: npt.NDArray[np.float64] | None = None
        self._trace_scale, self._diagonal_scale = self._normalization_vectors(normalization)

    def _means(self) -> npt.NDArray[np.float64]:
        if self._column_means is None:
            ones = np.ones(self.linear_arg.shape[0], dtype=np.float64)
            self._column_means = np.asarray(self.linear_arg.T @ ones, dtype=np.float64) / self.linear_arg.shape[0]
        return self._column_means

    def _haplotype_matmat(self, values: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        result = np.asarray(self.linear_arg @ values)
        if not self.center:
            return result
        centered_means = (self._means() @ values).reshape(1, -1)
        return result - np.ones((self.linear_arg.shape[0], 1), dtype=result.dtype) @ centered_means

    def _haplotype_rmatmat(self, values: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        result = np.asarray(self.linear_arg.T @ values)
        if not self.center:
            return result
        return result - self._means()[:, None] @ values.sum(axis=0, keepdims=True)

    def _raw_matmat(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        matrix, _ = as_column_matrix(values, expected_rows=self.shape[1], name="kernel input")
        haplotype_weights = self.diploid_map.haploid_from_diploid(matrix)
        variant_weights = self._haplotype_rmatmat(haplotype_weights)
        haplotype_result = self._haplotype_matmat(variant_weights)
        return self.diploid_map.diploid_from_haploid(haplotype_result)

    def _raw_diagonal(self) -> npt.NDArray[np.float64]:
        # diag(C H H.T C.T) is the squared norm of each column of H.T C.T.
        n = self.shape[0]
        diagonal = np.empty(n, dtype=np.float64)
        dtype = np.dtype(np.result_type(self.linear_arg.dtype or np.float64, np.float32))
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            basis = np.zeros((n, stop - start), dtype=dtype)
            basis[np.arange(start, stop), np.arange(stop - start)] = 1
            weights = self._haplotype_rmatmat(self.diploid_map.haploid_from_diploid(basis))
            diagonal[start:stop] = np.einsum("ij,ij->j", weights, weights)
        return diagonal


class SameHaplotypeInteractionKernel(_BaseKernel):
    """Same-haplotype cis-regulatory interaction kernel.

    This operator computes $C\\Phi(H)\\Phi(H)^TC^Tv$ where $\\Phi(h)=h\\otimes h$
    is applied to each physical haplotype row before diploid collapse. This is
    intentionally not the genotype-level GxG kernel $\\Phi(CH)\\Phi(CH)^T$.

    !!! info

        The exact backend constructs haplotype Gram columns in batches and
        caches the individual kernel. It never constructs a variant-square
        matrix or a pair-feature matrix. ``batch_size`` bounds the number of
        right-hand sides passed to the haplotype operator at construction.
    """

    def __init__(
        self,
        linear_arg: LinearOperator,
        diploid_map: DiploidHaplotypeMap,
        *,
        variant_indices: object | None = None,
        blocks: object | None = None,
        interaction_mode: InteractionMode | str = InteractionMode.ORDERED_SELF,
        backend: str = "dense_window",
        normalization: Normalization = True,
        center: bool = False,
        batch_size: int = 32,
    ) -> None:
        if backend != "dense_window":
            raise ValueError("only the dense_window backend is currently implemented")
        if variant_indices is not None and blocks is not None:
            raise ValueError("variant_indices and blocks are mutually exclusive")
        self.linear_arg = aslinearoperator(linear_arg)
        self.diploid_map = diploid_map
        self.interaction_mode = normalize_interaction_mode(interaction_mode)
        self.backend = backend
        self.center = center
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.shape = (diploid_map.n_individuals, diploid_map.n_individuals)
        if self.linear_arg.shape[0] != 2 * diploid_map.n_individuals:
            raise ValueError("linear_arg row count must equal 2 * diploid_map.n_individuals")
        self.blocks = self._normalize_blocks(variant_indices=variant_indices, blocks=blocks)
        self._interaction_matrix: npt.NDArray[np.floating] | None = None
        self._trace_scale, self._diagonal_scale = self._normalization_vectors(normalization)

    def _normalize_blocks(
        self,
        *,
        variant_indices: object | None,
        blocks: object | None,
    ) -> list[npt.NDArray[np.int64]]:
        p = self.linear_arg.shape[1]
        if blocks is None:
            if variant_indices is None:
                return [np.arange(p, dtype=np.int64)]
            blocks = [variant_indices]
        if not isinstance(blocks, Iterable):
            raise TypeError("blocks must be an iterable of one-dimensional variant index arrays")
        normalized = []
        for block in blocks:
            block_array = np.asarray(block, dtype=np.int64)
            if block_array.ndim != 1:
                raise ValueError("each block must be one-dimensional")
            if block_array.size == 0:
                raise ValueError("blocks must not be empty")
            if np.any((block_array < 0) | (block_array >= p)):
                raise ValueError("block variant indices are out of bounds")
            normalized.append(block_array)
        return normalized

    def _block_gram(self, block: npt.NDArray[np.int64]) -> npt.NDArray[np.floating]:
        """Construct H_block H_block.T using bounded batches of haplotype basis vectors."""
        rows, variants = self.linear_arg.shape
        dtype = np.dtype(np.result_type(self.linear_arg.dtype or np.float64, np.float32))
        gram = np.empty((rows, rows), dtype=dtype)
        # Counts preserve repeated indices, as explicit H[:, block] would.
        counts = np.bincount(block, minlength=variants).astype(dtype)
        for start in range(0, rows, self.batch_size):
            stop = min(start + self.batch_size, rows)
            basis = np.zeros((rows, stop - start), dtype=dtype)
            basis[np.arange(start, stop), np.arange(stop - start)] = 1
            if self.center:
                basis -= basis.mean(axis=0, keepdims=True)
            weights = np.asarray(self.linear_arg.T @ basis) * counts[:, None]
            columns = np.asarray(self.linear_arg @ weights)
            if self.center:
                columns = columns - columns.mean(axis=0, keepdims=True)
            gram[:, start:stop] = columns
        return (gram + gram.T) * 0.5

    def _squared_feature_gram(self, block: npt.NDArray[np.int64]) -> npt.NDArray[np.floating]:
        """Sum squared-column outer products to remove self pairs, including after centering."""
        rows, variants = self.linear_arg.shape
        dtype = np.dtype(np.result_type(self.linear_arg.dtype or np.float64, np.float32))
        gram = np.zeros((rows, rows), dtype=dtype)
        for start in range(0, len(block), self.batch_size):
            indices = block[start : start + self.batch_size]
            selector = np.zeros((variants, len(indices)), dtype=dtype)
            selector[indices, np.arange(len(indices))] = 1
            columns = np.asarray(self.linear_arg @ selector)
            if self.center:
                columns = columns - columns.mean(axis=0, keepdims=True)
            squared = np.square(columns)
            gram += squared @ squared.T
        return gram

    def _cached_matrix(self) -> npt.NDArray[np.floating]:
        if self._interaction_matrix is None:
            n = self.shape[0]
            dtype = np.dtype(np.result_type(self.linear_arg.dtype or np.float64, np.float32))
            result = np.zeros((n, n), dtype=dtype)
            for block in self.blocks:
                gram = self._block_gram(block)
                np.square(gram, out=gram)
                if self.interaction_mode is InteractionMode.UNORDERED_OFFDIAG:
                    gram -= self._squared_feature_gram(block)
                    gram *= 0.5
                # Square before diploid aggregation; retain cross-variant pairs
                # across all construction batches within this regulatory block.
                result += gram.reshape(n, 2, n, 2).sum(axis=(1, 3))
            self._interaction_matrix = result
        return self._interaction_matrix

    def _raw_matmat(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        matrix, _ = as_column_matrix(values, expected_rows=self.shape[1], name="kernel input")
        return self._cached_matrix() @ matrix

    def _raw_diagonal(self) -> npt.NDArray[np.float64]:
        return np.asarray(self._cached_matrix().diagonal(), dtype=np.float64)


@dataclass(frozen=True)
class VarianceComponentKernel:
    """Variance-component covariance operator.

    This combines additive and same-haplotype interaction components as
    $K = \\sigma_A^2K_A + \\sigma_H^2K_H + \\sigma_e^2I$.
    """

    additive_kernel: KernelOperator
    interaction_kernel: KernelOperator

    def __post_init__(self) -> None:
        if self.additive_kernel.shape != self.interaction_kernel.shape:
            raise ValueError("additive and interaction kernels must have the same shape")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the diploid covariance shape."""
        return self.additive_kernel.shape

    def matvec(
        self,
        values: npt.ArrayLike,
        *,
        sigma_a2: float,
        sigma_h2: float,
        sigma_e2: float,
    ) -> npt.NDArray[np.number]:
        """Apply $\\sigma_A^2K_A + \\sigma_H^2K_H + \\sigma_e^2I$ to a vector.

        **Arguments:**

        - `values`: Individual-level vector.
        - `sigma_a2`: Additive variance component.
        - `sigma_h2`: Same-haplotype interaction variance component.
        - `sigma_e2`: Residual variance component.

        **Returns:**

        - Individual-level covariance product.
        """
        vector = np.asarray(values)
        return (
            sigma_a2 * self.additive_kernel.matvec(vector)
            + sigma_h2 * self.interaction_kernel.matvec(vector)
            + sigma_e2 * vector
        )
