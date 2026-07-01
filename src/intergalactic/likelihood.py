# pattern: Functional Core

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, Protocol

import numpy as np
import numpy.typing as npt

from scipy.optimize import minimize


class MatrixKernel(Protocol):
    """Protocol for kernels that can be materialized by applying identity columns."""

    shape: tuple[int, int]

    def matmat(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        """Apply the kernel to one or more vectors."""


@dataclass(frozen=True)
class VarianceComponents:
    """Variance-component parameterization.

    The covariance model is
    $K = \\sigma_A^2K_A + \\sigma_H^2K_H + \\sigma_e^2I$.
    """

    sigma_a2: float
    sigma_h2: float
    sigma_e2: float

    def as_array(self) -> npt.NDArray[np.float64]:
        """Return variance components in `(sigma_a2, sigma_h2, sigma_e2)` order."""
        return np.array([self.sigma_a2, self.sigma_h2, self.sigma_e2], dtype=np.float64)


@dataclass(frozen=True)
class GaussianLogLikelihood:
    """Exact Gaussian log-likelihood evaluation."""

    log_likelihood: float
    quadratic_form: float
    log_determinant: float
    variance_components: VarianceComponents


@dataclass(frozen=True)
class VarianceComponentFit:
    """Result from exact variance-component likelihood optimization."""

    variance_components: VarianceComponents
    log_likelihood: float
    negative_log_likelihood: float
    success: bool
    message: str
    n_iterations: int


def _validate_variance_components(variance_components: VarianceComponents) -> npt.NDArray[np.float64]:
    values = variance_components.as_array()
    if not np.all(np.isfinite(values)):
        raise ValueError("variance components must be finite")
    if np.any(values < 0.0):
        raise ValueError("variance components must be nonnegative")
    return values


def _validate_response(y: npt.ArrayLike) -> npt.NDArray[np.float64]:
    response = np.asarray(y, dtype=np.float64)
    if response.ndim != 1:
        raise ValueError("y must be a one-dimensional response vector")
    if response.size == 0:
        raise ValueError("y must not be empty")
    if not np.all(np.isfinite(response)):
        raise ValueError("y must contain only finite values")
    return response


def _validate_component_matrix(matrix: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    component = np.asarray(matrix, dtype=np.float64)
    if component.ndim != 2 or component.shape[0] != component.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(component)):
        raise ValueError(f"{name} must contain only finite values")
    return component


def dense_kernel_matrix(kernel: MatrixKernel) -> npt.NDArray[np.float64]:
    """Materialize a square kernel operator by applying identity columns.

    This is the exact likelihood boundary. It is appropriate for small or
    moderate sample counts where an $n \\times n$ covariance matrix is acceptable.

    **Arguments:**

    - `kernel`: Square kernel exposing `shape` and `matmat`.

    **Returns:**

    - Dense kernel matrix.

    **Raises:**

    - `ValueError`: If the kernel is not square.
    """
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square to materialize a covariance matrix")
    identity = np.eye(kernel.shape[1], dtype=np.float64)
    return _validate_component_matrix(kernel.matmat(identity), name="kernel")


def _as_component_matrix(component: MatrixKernel | npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    if hasattr(component, "matmat") and hasattr(component, "shape"):
        return dense_kernel_matrix(cast(MatrixKernel, component))
    return _validate_component_matrix(component, name=name)


def covariance_matrix(
    additive: MatrixKernel | npt.ArrayLike,
    interaction: MatrixKernel | npt.ArrayLike,
    variance_components: VarianceComponents,
) -> npt.NDArray[np.float64]:
    """Construct $\\sigma_A^2K_A + \\sigma_H^2K_H + \\sigma_e^2I$.

    **Arguments:**

    - `additive`: Additive component matrix or square kernel.
    - `interaction`: Same-haplotype interaction component matrix or square kernel.
    - `variance_components`: Nonnegative variance components.

    **Returns:**

    - Dense covariance matrix.
    """
    sigma_a2, sigma_h2, sigma_e2 = _validate_variance_components(variance_components)
    additive_matrix = _as_component_matrix(additive, name="additive")
    interaction_matrix = _as_component_matrix(interaction, name="interaction")
    if additive_matrix.shape != interaction_matrix.shape:
        raise ValueError("additive and interaction components must have the same shape")
    return sigma_a2 * additive_matrix + sigma_h2 * interaction_matrix + sigma_e2 * np.eye(additive_matrix.shape[0])


def gaussian_log_likelihood(
    y: npt.ArrayLike,
    additive: MatrixKernel | npt.ArrayLike,
    interaction: MatrixKernel | npt.ArrayLike,
    variance_components: VarianceComponents,
) -> GaussianLogLikelihood:
    """Evaluate the exact marginal Gaussian log likelihood.

    The evaluated model is
    $y \\sim N(0, \\sigma_A^2K_A + \\sigma_H^2K_H + \\sigma_e^2I)$.

    **Arguments:**

    - `y`: One-dimensional phenotype or molecular phenotype vector.
    - `additive`: Additive component matrix or square kernel.
    - `interaction`: Same-haplotype interaction component matrix or square kernel.
    - `variance_components`: Nonnegative variance components.

    **Returns:**

    - Log-likelihood result containing the likelihood and Cholesky-derived
      diagnostic terms.

    **Raises:**

    - `ValueError`: If dimensions are inconsistent or the covariance is not
      positive definite.
    """
    response = _validate_response(y)
    covariance = covariance_matrix(additive, interaction, variance_components)
    if covariance.shape[0] != response.shape[0]:
        raise ValueError("covariance dimension must match y length")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance matrix is not positive definite") from exc
    alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, response))
    quadratic_form = float(response @ alpha)
    log_determinant = float(2.0 * np.log(np.diag(cholesky)).sum())
    log_likelihood = -0.5 * (quadratic_form + log_determinant + response.shape[0] * np.log(2.0 * np.pi))
    return GaussianLogLikelihood(
        log_likelihood=float(log_likelihood),
        quadratic_form=quadratic_form,
        log_determinant=log_determinant,
        variance_components=variance_components,
    )


def _variance_components_from_log(log_values: npt.NDArray[np.float64]) -> VarianceComponents:
    sigma_a2, sigma_h2, sigma_e2 = np.exp(log_values)
    return VarianceComponents(float(sigma_a2), float(sigma_h2), float(sigma_e2))


def _default_initial_components(y: npt.NDArray[np.float64]) -> VarianceComponents:
    empirical_second_moment = max(float(y @ y / y.shape[0]), 1e-6)
    share = empirical_second_moment / 3.0
    return VarianceComponents(share, share, share)


def optimize_variance_components(
    y: npt.ArrayLike,
    additive: MatrixKernel | npt.ArrayLike,
    interaction: MatrixKernel | npt.ArrayLike,
    *,
    initial: VarianceComponents | None = None,
    lower_bound: float = 1e-10,
    upper_bound: float | None = None,
    maxiter: int = 1000,
) -> VarianceComponentFit:
    """Optimize variance components by exact Gaussian maximum likelihood.

    Optimization is performed over log variance components with L-BFGS-B. This
    keeps fitted variance components positive while still allowing an explicit
    lower bound close to zero.

    **Arguments:**

    - `y`: One-dimensional phenotype or molecular phenotype vector.
    - `additive`: Additive component matrix or square kernel.
    - `interaction`: Same-haplotype interaction component matrix or square kernel.
    - `initial`: Optional positive starting variance components.
    - `lower_bound`: Positive lower bound for each variance component.
    - `upper_bound`: Optional finite upper bound for each variance component.
    - `maxiter`: Maximum optimizer iterations.

    **Returns:**

    - Fitted variance components and optimizer status.

    **Raises:**

    - `ValueError`: If bounds or dimensions are invalid.
    """
    response = _validate_response(y)
    if lower_bound <= 0.0 or not np.isfinite(lower_bound):
        raise ValueError("lower_bound must be finite and positive")
    if upper_bound is not None and (upper_bound <= lower_bound or not np.isfinite(upper_bound)):
        raise ValueError("upper_bound must be finite and greater than lower_bound")
    additive_matrix = _as_component_matrix(additive, name="additive")
    interaction_matrix = _as_component_matrix(interaction, name="interaction")
    if additive_matrix.shape != interaction_matrix.shape:
        raise ValueError("additive and interaction components must have the same shape")
    if additive_matrix.shape[0] != response.shape[0]:
        raise ValueError("component dimensions must match y length")

    starting = initial or _default_initial_components(response)
    initial_values = np.maximum(_validate_variance_components(starting), lower_bound)
    bounds = [(np.log(lower_bound), None if upper_bound is None else np.log(upper_bound))] * 3

    def objective(log_values: npt.NDArray[np.float64]) -> float:
        variance_components = _variance_components_from_log(log_values)
        try:
            return -gaussian_log_likelihood(
                response,
                additive_matrix,
                interaction_matrix,
                variance_components,
            ).log_likelihood
        except ValueError:
            return float("inf")

    result = minimize(
        objective,
        np.log(initial_values),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )
    fitted_components = _variance_components_from_log(np.asarray(result.x, dtype=np.float64))
    log_likelihood = gaussian_log_likelihood(response, additive_matrix, interaction_matrix, fitted_components)
    return VarianceComponentFit(
        variance_components=fitted_components,
        log_likelihood=log_likelihood.log_likelihood,
        negative_log_likelihood=-log_likelihood.log_likelihood,
        success=bool(result.success),
        message=str(result.message),
        n_iterations=int(result.nit),
    )
