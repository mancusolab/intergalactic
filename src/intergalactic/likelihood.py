# pattern: Functional Core

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from scipy.optimize import minimize
from scipy.sparse.linalg import cg, LinearOperator


class MatvecKernel(Protocol):
    """Protocol for square kernels used through matrix-vector products only."""

    shape: tuple[int, int]

    def matvec(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        """Apply the kernel to one vector."""


LogdetProbeMode = Literal["rademacher", "normal", "basis"]


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
    """Matvec-only Gaussian log-likelihood evaluation.

    The quadratic form is computed by conjugate gradients. The log determinant
    is estimated by Lanczos quadrature; `logdet_probe_mode="basis"` with
    `lanczos_rank >= n` gives the deterministic full-basis result for small
    tests while still using only matvecs.
    """

    log_likelihood: float
    quadratic_form: float
    log_determinant: float
    score: npt.NDArray[np.float64]
    average_information: npt.NDArray[np.float64]
    log_score: npt.NDArray[np.float64]
    log_average_information: npt.NDArray[np.float64]
    variance_components: VarianceComponents
    logdet_method: str
    logdet_standard_error: float
    num_logdet_probes: int
    lanczos_rank: int
    cg_info: int


@dataclass(frozen=True)
class VarianceComponentFit:
    """Result from matvec-only variance-component likelihood optimization."""

    variance_components: VarianceComponents
    log_likelihood: float
    negative_log_likelihood: float
    success: bool
    message: str
    n_iterations: int
    log_gradient: npt.NDArray[np.float64]
    log_average_information: npt.NDArray[np.float64]
    logdet_method: str
    num_logdet_probes: int
    lanczos_rank: int


class VarianceComponentOperator(LinearOperator):
    """Linear operator for $\\sigma_A^2K_A + \\sigma_H^2K_H + \\sigma_e^2I$."""

    def __init__(
        self,
        additive: MatvecKernel,
        interaction: MatvecKernel,
        variance_components: VarianceComponents,
    ) -> None:
        if additive.shape[0] != additive.shape[1]:
            raise ValueError("additive kernel must be square")
        if interaction.shape[0] != interaction.shape[1]:
            raise ValueError("interaction kernel must be square")
        if additive.shape != interaction.shape:
            raise ValueError("additive and interaction kernels must have the same shape")
        _validate_variance_components(variance_components)
        super().__init__(np.dtype(np.float64), additive.shape)
        self.additive = additive
        self.interaction = interaction
        self.variance_components = variance_components

    def _matvec(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        vector = np.asarray(x, dtype=np.float64)
        if vector.shape != (self.shape[1],):
            raise ValueError(f"covariance matvec expected shape {(self.shape[1],)}; got {vector.shape}")
        return (
            self.variance_components.sigma_a2 * np.asarray(self.additive.matvec(vector), dtype=np.float64)
            + self.variance_components.sigma_h2 * np.asarray(self.interaction.matvec(vector), dtype=np.float64)
            + self.variance_components.sigma_e2 * vector
        )

    def _matmat(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != self.shape[1]:
            raise ValueError(f"covariance matmat expected leading dimension {self.shape[1]}; got {matrix.shape}")
        return np.column_stack([self._matvec(matrix[:, column]) for column in range(matrix.shape[1])])


@dataclass(frozen=True)
class _LogdetEstimate:
    value: float
    standard_error: float
    num_probes: int
    lanczos_rank: int
    method: str


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


def _solve_covariance(
    operator: LinearOperator,
    values: npt.ArrayLike,
    *,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
) -> tuple[npt.NDArray[np.float64], int]:
    solution, info = cg(operator, np.asarray(values, dtype=np.float64), rtol=cg_rtol, atol=cg_atol, maxiter=cg_maxiter)
    return np.asarray(solution, dtype=np.float64), int(info)


def _component_products(
    additive: MatvecKernel,
    interaction: MatvecKernel,
    values: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    vector = np.asarray(values, dtype=np.float64)
    return (
        np.asarray(additive.matvec(vector), dtype=np.float64),
        np.asarray(interaction.matvec(vector), dtype=np.float64),
        vector,
    )


def covariance_operator(
    additive: MatvecKernel,
    interaction: MatvecKernel,
    variance_components: VarianceComponents,
) -> VarianceComponentOperator:
    """Build the variance-component covariance as a matvec-only operator.

    **Arguments:**

    - `additive`: Square additive kernel exposing `matvec`.
    - `interaction`: Square same-haplotype interaction kernel exposing `matvec`.
    - `variance_components`: Nonnegative variance components.

    **Returns:**

    - Linear operator for $\\sigma_A^2K_A + \\sigma_H^2K_H + \\sigma_e^2I$.
    """
    return VarianceComponentOperator(additive, interaction, variance_components)


def _basis_probe(index: int, size: int) -> npt.NDArray[np.float64]:
    probe = np.zeros(size, dtype=np.float64)
    probe[index] = 1.0
    return probe


def _probe_vectors(
    *,
    size: int,
    mode: LogdetProbeMode,
    num_probes: int,
    rng: np.random.Generator,
) -> Iterator[npt.NDArray[np.float64]]:
    if mode == "basis":
        for index in range(size):
            yield _basis_probe(index, size)
        return
    if mode == "rademacher":
        for _ in range(num_probes):
            yield rng.choice(np.array([-1.0, 1.0]), size=size)
        return
    if mode == "normal":
        for _ in range(num_probes):
            yield rng.normal(size=size)
        return
    raise ValueError("logdet_probe_mode must be 'rademacher', 'normal', or 'basis'")


def _lanczos_log_quadrature(
    operator: LinearOperator,
    probe: npt.NDArray[np.float64],
    *,
    lanczos_rank: int,
    breakdown_tol: float = 1e-12,
) -> tuple[float, int]:
    probe_norm = float(np.linalg.norm(probe))
    if probe_norm == 0.0:
        raise ValueError("log-determinant probe vectors must be nonzero")
    max_rank = min(lanczos_rank, operator.shape[0])
    if max_rank <= 0:
        raise ValueError("lanczos_rank must be positive")

    q = probe / probe_norm
    previous_q = np.zeros_like(q)
    previous_beta = 0.0
    basis: list[npt.NDArray[np.float64]] = []
    alphas: list[float] = []
    betas: list[float] = []

    for step in range(max_rank):
        basis.append(q.copy())
        residual = np.asarray(operator.matvec(q), dtype=np.float64)
        alpha = float(q @ residual)
        residual = residual - alpha * q
        if step > 0:
            residual = residual - previous_beta * previous_q
        # Full reorthogonalization keeps the tiny Lanczos tridiagonal stable
        # enough for deterministic full-basis tests.
        for basis_vector in basis:
            residual = residual - float(basis_vector @ residual) * basis_vector
        beta = float(np.linalg.norm(residual))
        alphas.append(alpha)
        if step == max_rank - 1 or beta <= breakdown_tol:
            break
        betas.append(beta)
        previous_q = q
        previous_beta = beta
        q = residual / beta

    tridiagonal = np.diag(np.asarray(alphas, dtype=np.float64))
    if betas:
        off_diagonal = np.asarray(betas, dtype=np.float64)
        tridiagonal += np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)
    eigenvalues, eigenvectors = np.linalg.eigh(tridiagonal)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Lanczos tridiagonal is not positive definite")
    weights = eigenvectors[0, :] ** 2
    return float(probe_norm**2 * np.sum(weights * np.log(eigenvalues))), len(alphas)


def _estimate_log_determinant(
    operator: LinearOperator,
    *,
    mode: LogdetProbeMode,
    num_logdet_probes: int,
    lanczos_rank: int,
    seed: int | None,
) -> _LogdetEstimate:
    size = operator.shape[0]
    if num_logdet_probes <= 0:
        raise ValueError("num_logdet_probes must be positive")
    rng = np.random.default_rng(seed)
    estimates = []
    ranks = []
    for probe in _probe_vectors(size=size, mode=mode, num_probes=num_logdet_probes, rng=rng):
        estimate, rank_used = _lanczos_log_quadrature(operator, probe, lanczos_rank=lanczos_rank)
        estimates.append(estimate)
        ranks.append(rank_used)
    estimate_array = np.asarray(estimates, dtype=np.float64)
    if mode == "basis":
        value = float(estimate_array.sum())
        standard_error = 0.0
        probe_count = size
    else:
        value = float(estimate_array.mean())
        standard_error = (
            float(estimate_array.std(ddof=1) / np.sqrt(estimate_array.shape[0])) if num_logdet_probes > 1 else 0.0
        )
        probe_count = num_logdet_probes
    return _LogdetEstimate(
        value=value,
        standard_error=standard_error,
        num_probes=probe_count,
        lanczos_rank=max(ranks),
        method=f"lanczos_{mode}",
    )


def _estimate_trace_terms(
    operator: LinearOperator,
    additive: MatvecKernel,
    interaction: MatvecKernel,
    *,
    mode: LogdetProbeMode,
    num_trace_probes: int,
    seed: int | None,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
) -> npt.NDArray[np.float64]:
    size = operator.shape[0]
    if num_trace_probes <= 0:
        raise ValueError("num_trace_probes must be positive")
    rng = np.random.default_rng(seed)
    trace_terms = np.zeros(3, dtype=np.float64)
    probe_count = 0
    for probe in _probe_vectors(size=size, mode=mode, num_probes=num_trace_probes, rng=rng):
        precision_probe, cg_info = _solve_covariance(
            operator,
            probe,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_maxiter=cg_maxiter,
        )
        if cg_info != 0:
            raise ValueError(f"conjugate gradients did not converge while estimating traces; info={cg_info}")
        trace_terms += np.array(
            [component @ precision_probe for component in _component_products(additive, interaction, probe)],
            dtype=np.float64,
        )
        probe_count += 1
    return trace_terms if mode == "basis" else trace_terms / probe_count


def _score_and_average_information(
    operator: LinearOperator,
    additive: MatvecKernel,
    interaction: MatvecKernel,
    alpha: npt.NDArray[np.float64],
    *,
    trace_mode: LogdetProbeMode,
    num_trace_probes: int,
    seed: int | None,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    component_alpha = _component_products(additive, interaction, alpha)
    quadratic_terms = np.array([alpha @ values for values in component_alpha], dtype=np.float64)
    trace_terms = _estimate_trace_terms(
        operator,
        additive,
        interaction,
        mode=trace_mode,
        num_trace_probes=num_trace_probes,
        seed=seed,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
    )
    score = 0.5 * (quadratic_terms - trace_terms)

    precision_component_alpha = []
    for values in component_alpha:
        solved, cg_info = _solve_covariance(
            operator,
            values,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_maxiter=cg_maxiter,
        )
        if cg_info != 0:
            raise ValueError(
                f"conjugate gradients did not converge while computing average information; info={cg_info}"
            )
        precision_component_alpha.append(solved)

    average_information = np.array(
        [[0.5 * component_i @ precision_component_alpha[j] for j in range(3)] for component_i in component_alpha],
        dtype=np.float64,
    )
    return score, 0.5 * (average_information + average_information.T)


def gaussian_log_likelihood(
    y: npt.ArrayLike,
    additive: MatvecKernel,
    interaction: MatvecKernel,
    variance_components: VarianceComponents,
    *,
    logdet_probe_mode: LogdetProbeMode = "rademacher",
    num_logdet_probes: int = 16,
    lanczos_rank: int = 32,
    seed: int | None = 0,
    cg_rtol: float = 1e-6,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
) -> GaussianLogLikelihood:
    """Evaluate a matvec-only marginal Gaussian log likelihood.

    The evaluated model is
    $y \\sim N(0, \\sigma_A^2K_A + \\sigma_H^2K_H + \\sigma_e^2I)$.
    The solve uses conjugate gradients. The log determinant uses stochastic
    Lanczos quadrature unless `logdet_probe_mode="basis"` is selected.

    **Arguments:**

    - `y`: One-dimensional phenotype or molecular phenotype vector.
    - `additive`: Additive kernel exposing `matvec`.
    - `interaction`: Same-haplotype interaction kernel exposing `matvec`.
    - `variance_components`: Nonnegative variance components.
    - `logdet_probe_mode`: `"rademacher"`, `"normal"`, or `"basis"`.
    - `num_logdet_probes`: Number of random probes for stochastic modes.
    - `lanczos_rank`: Maximum Lanczos rank for each probe.
    - `seed`: Random seed for stochastic probes.
    - `cg_rtol`: Relative tolerance for conjugate gradients.
    - `cg_atol`: Absolute tolerance for conjugate gradients.
    - `cg_maxiter`: Optional maximum conjugate-gradient iterations.

    **Returns:**

    - Log-likelihood result containing quadratic and log-determinant terms.
    """
    response = _validate_response(y)
    operator = covariance_operator(additive, interaction, variance_components)
    if operator.shape[0] != response.shape[0]:
        raise ValueError("covariance dimension must match y length")
    solution, cg_info = _solve_covariance(
        operator,
        response,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
    )
    if cg_info != 0:
        raise ValueError(f"conjugate gradients did not converge; info={cg_info}")
    quadratic_form = float(response @ solution)
    logdet = _estimate_log_determinant(
        operator,
        mode=logdet_probe_mode,
        num_logdet_probes=num_logdet_probes,
        lanczos_rank=lanczos_rank,
        seed=seed,
    )
    score, average_information = _score_and_average_information(
        operator,
        additive,
        interaction,
        solution,
        trace_mode=logdet_probe_mode,
        num_trace_probes=num_logdet_probes,
        seed=seed,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
    )
    variance_vector = variance_components.as_array()
    log_likelihood = -0.5 * (quadratic_form + logdet.value + response.shape[0] * np.log(2.0 * np.pi))
    return GaussianLogLikelihood(
        log_likelihood=float(log_likelihood),
        quadratic_form=quadratic_form,
        log_determinant=logdet.value,
        score=score,
        average_information=average_information,
        log_score=variance_vector * score,
        log_average_information=np.outer(variance_vector, variance_vector) * average_information,
        variance_components=variance_components,
        logdet_method=logdet.method,
        logdet_standard_error=logdet.standard_error,
        num_logdet_probes=logdet.num_probes,
        lanczos_rank=logdet.lanczos_rank,
        cg_info=int(cg_info),
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
    additive: MatvecKernel,
    interaction: MatvecKernel,
    *,
    initial: VarianceComponents | None = None,
    lower_bound: float = 1e-10,
    upper_bound: float | None = None,
    maxiter: int = 1000,
    logdet_probe_mode: LogdetProbeMode = "rademacher",
    num_logdet_probes: int = 16,
    lanczos_rank: int = 32,
    seed: int | None = 0,
    cg_rtol: float = 1e-6,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
) -> VarianceComponentFit:
    """Optimize variance components with a matvec-only likelihood objective.

    Optimization is performed over log variance components with L-BFGS-B. The
    objective uses conjugate gradients and Lanczos log-determinant estimates;
    no component kernel or covariance matrix is materialized.

    **Arguments:**

    - `y`: One-dimensional phenotype or molecular phenotype vector.
    - `additive`: Additive kernel exposing `matvec`.
    - `interaction`: Same-haplotype interaction kernel exposing `matvec`.
    - `initial`: Optional positive starting variance components.
    - `lower_bound`: Positive lower bound for each variance component.
    - `upper_bound`: Optional finite upper bound for each variance component.
    - `maxiter`: Maximum optimizer iterations.
    - `logdet_probe_mode`: `"rademacher"`, `"normal"`, or `"basis"`.
    - `num_logdet_probes`: Number of random probes for stochastic modes.
    - `lanczos_rank`: Maximum Lanczos rank for each probe.
    - `seed`: Random seed for stochastic probes.
    - `cg_rtol`: Relative tolerance for conjugate gradients.
    - `cg_atol`: Absolute tolerance for conjugate gradients.
    - `cg_maxiter`: Optional maximum conjugate-gradient iterations.

    **Returns:**

    - Fitted variance components and optimizer status.
    """
    response = _validate_response(y)
    if lower_bound <= 0.0 or not np.isfinite(lower_bound):
        raise ValueError("lower_bound must be finite and positive")
    if upper_bound is not None and (upper_bound <= lower_bound or not np.isfinite(upper_bound)):
        raise ValueError("upper_bound must be finite and greater than lower_bound")
    if additive.shape != interaction.shape:
        raise ValueError("additive and interaction kernels must have the same shape")
    if additive.shape[0] != response.shape[0]:
        raise ValueError("component dimensions must match y length")

    starting = initial or _default_initial_components(response)
    initial_values = np.maximum(_validate_variance_components(starting), lower_bound)
    bounds = [(np.log(lower_bound), None if upper_bound is None else np.log(upper_bound))] * 3

    def objective(log_values: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]:
        variance_components = _variance_components_from_log(log_values)
        try:
            likelihood = gaussian_log_likelihood(
                response,
                additive,
                interaction,
                variance_components,
                logdet_probe_mode=logdet_probe_mode,
                num_logdet_probes=num_logdet_probes,
                lanczos_rank=lanczos_rank,
                seed=seed,
                cg_rtol=cg_rtol,
                cg_atol=cg_atol,
                cg_maxiter=cg_maxiter,
            )
            return -likelihood.log_likelihood, -likelihood.log_score
        except ValueError:
            return float("inf"), np.zeros(3, dtype=np.float64)

    result = minimize(
        objective,
        np.log(initial_values),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter},
    )
    fitted_components = _variance_components_from_log(np.asarray(result.x, dtype=np.float64))
    log_likelihood = gaussian_log_likelihood(
        response,
        additive,
        interaction,
        fitted_components,
        logdet_probe_mode=logdet_probe_mode,
        num_logdet_probes=num_logdet_probes,
        lanczos_rank=lanczos_rank,
        seed=seed,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
    )
    return VarianceComponentFit(
        variance_components=fitted_components,
        log_likelihood=log_likelihood.log_likelihood,
        negative_log_likelihood=-log_likelihood.log_likelihood,
        success=bool(result.success),
        message=str(result.message),
        n_iterations=int(result.nit),
        log_gradient=-log_likelihood.log_score,
        log_average_information=log_likelihood.log_average_information,
        logdet_method=log_likelihood.logdet_method,
        num_logdet_probes=log_likelihood.num_logdet_probes,
        lanczos_rank=log_likelihood.lanczos_rank,
    )
