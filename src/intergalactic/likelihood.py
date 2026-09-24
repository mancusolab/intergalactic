# pattern: Functional Core

from __future__ import annotations

import logging

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from scipy.sparse.linalg import cg, LinearOperator


class MatvecKernel(Protocol):
    """Protocol for square kernels used through matrix-vector products only."""

    shape: tuple[int, int]

    def matvec(self, values: npt.ArrayLike) -> npt.NDArray[np.number]:
        """Apply the kernel to one vector."""


LogdetProbeMode = Literal["rademacher", "normal", "basis"]
_AI_RIDGE = 1e-8
_TRUST_SHRINK = 0.25
_TRUST_GROW = 2.0
_MIN_GAIN_RATIO = 1e-4
_LOW_GAIN_RATIO = 0.25
_HIGH_GAIN_RATIO = 0.75
_BOUNDARY_STEP_FRACTION = 0.8
_MESSAGE_SCORE_CONVERGED = "log-scale score converged"
_MESSAGE_STEP_CONVERGED = "projected trust-region step converged"
_MESSAGE_RADIUS_CONVERGED = "trust-region radius converged"
_MESSAGE_MAXITER = "maximum iterations reached"
_CONVERGED_MESSAGES = frozenset(
    {
        _MESSAGE_SCORE_CONVERGED,
        _MESSAGE_STEP_CONVERGED,
        _MESSAGE_RADIUS_CONVERGED,
    }
)


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
    fixed_effects: npt.NDArray[np.float64]
    residual: npt.NDArray[np.float64]
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

    optimizer: str
    variance_components: VarianceComponents
    log_likelihood: float
    negative_log_likelihood: float
    success: bool
    message: str
    n_iterations: int
    accepted_steps: int
    rejected_steps: int
    trust_radius: float
    fixed_effects: npt.NDArray[np.float64]
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


@dataclass(frozen=True)
class _ProfiledMean:
    fixed_effects: npt.NDArray[np.float64]
    residual: npt.NDArray[np.float64]
    alpha: npt.NDArray[np.float64]
    quadratic_form: float


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


def _validate_covariates(covariates: npt.ArrayLike | None, *, n_observations: int) -> npt.NDArray[np.float64]:
    if covariates is None:
        return np.zeros((n_observations, 0), dtype=np.float64)
    design = np.asarray(covariates, dtype=np.float64)
    if design.ndim == 1:
        design = design.reshape(-1, 1)
    if design.ndim != 2:
        raise ValueError("covariates must be a one- or two-dimensional array")
    if design.shape[0] != n_observations:
        raise ValueError("covariates must have one row per response value")
    if not np.all(np.isfinite(design)):
        raise ValueError("covariates must contain only finite values")
    return design


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


def _solve_covariates(
    operator: LinearOperator,
    covariates: npt.NDArray[np.float64],
    *,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
) -> npt.NDArray[np.float64]:
    if covariates.shape[1] == 0:
        return np.zeros_like(covariates)
    solved_columns = []
    for column in range(covariates.shape[1]):
        solved, cg_info = _solve_covariance(
            operator,
            covariates[:, column],
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_maxiter=cg_maxiter,
        )
        if cg_info != 0:
            raise ValueError(f"conjugate gradients did not converge while solving covariates; info={cg_info}")
        solved_columns.append(solved)
    return np.column_stack(solved_columns)


def _profile_mean(
    response: npt.NDArray[np.float64],
    covariates: npt.NDArray[np.float64],
    precision_response: npt.NDArray[np.float64],
    operator: LinearOperator,
    *,
    cg_rtol: float,
    cg_atol: float,
    cg_maxiter: int | None,
) -> _ProfiledMean:
    if covariates.shape[1] == 0:
        return _ProfiledMean(
            fixed_effects=np.zeros(0, dtype=np.float64),
            residual=response,
            alpha=precision_response,
            quadratic_form=float(response @ precision_response),
        )

    precision_covariates = _solve_covariates(
        operator,
        covariates,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
    )
    normal_matrix = covariates.T @ precision_covariates
    rhs = covariates.T @ precision_response
    try:
        fixed_effects = np.linalg.solve(normal_matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariates must be full rank under the covariance precision") from error
    residual = response - covariates @ fixed_effects
    alpha = precision_response - precision_covariates @ fixed_effects
    return _ProfiledMean(
        fixed_effects=np.asarray(fixed_effects, dtype=np.float64),
        residual=np.asarray(residual, dtype=np.float64),
        alpha=np.asarray(alpha, dtype=np.float64),
        quadratic_form=float(residual @ alpha),
    )


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
    covariates: npt.ArrayLike | None = None,
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
    $y \\sim N(X\\beta, \\sigma_A^2K_A + \\sigma_H^2K_H + \\sigma_e^2I)$.
    Fixed effects are profiled by generalized least squares. The covariance
    solves use conjugate gradients. The log determinant uses stochastic Lanczos
    quadrature unless `logdet_probe_mode="basis"` is selected.

    **Arguments:**

    - `y`: One-dimensional phenotype or molecular phenotype vector.
    - `additive`: Additive kernel exposing `matvec`.
    - `interaction`: Same-haplotype interaction kernel exposing `matvec`.
    - `variance_components`: Nonnegative variance components.
    - `covariates`: Optional fixed-effect design matrix `X`.
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
    covariate_matrix = _validate_covariates(covariates, n_observations=response.shape[0])
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
    profiled_mean = _profile_mean(
        response,
        covariate_matrix,
        solution,
        operator,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
    )
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
        profiled_mean.alpha,
        trace_mode=logdet_probe_mode,
        num_trace_probes=num_logdet_probes,
        seed=seed,
        cg_rtol=cg_rtol,
        cg_atol=cg_atol,
        cg_maxiter=cg_maxiter,
    )
    variance_vector = variance_components.as_array()
    log_likelihood = -0.5 * (profiled_mean.quadratic_form + logdet.value + response.shape[0] * np.log(2.0 * np.pi))
    return GaussianLogLikelihood(
        log_likelihood=float(log_likelihood),
        quadratic_form=profiled_mean.quadratic_form,
        log_determinant=logdet.value,
        fixed_effects=profiled_mean.fixed_effects,
        residual=profiled_mean.residual,
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


def _clip_log_variances(
    log_values: npt.NDArray[np.float64],
    *,
    lower_log: float,
    upper_log: float | None,
) -> npt.NDArray[np.float64]:
    upper = np.inf if upper_log is None else upper_log
    return np.clip(np.asarray(log_values, dtype=np.float64), lower_log, upper)


def _symmetrize(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 0.5 * (values + values.T)


def _solve_ai_step(
    log_score: npt.NDArray[np.float64],
    log_average_information: npt.NDArray[np.float64],
    *,
    trust_radius: float,
) -> npt.NDArray[np.float64]:
    if trust_radius <= 0.0 or not np.isfinite(trust_radius):
        raise ValueError("trust_radius must be finite and positive")
    information = _symmetrize(log_average_information)
    stabilized = information + _AI_RIDGE * np.eye(information.shape[0])
    try:
        step = np.linalg.solve(stabilized, log_score)
    except np.linalg.LinAlgError:
        step = np.linalg.lstsq(stabilized, log_score, rcond=None)[0]

    step_norm = float(np.linalg.norm(step))
    if not np.all(np.isfinite(step)) or step_norm == 0.0:
        score_norm = float(np.linalg.norm(log_score))
        if score_norm == 0.0 or not np.isfinite(score_norm):
            return np.zeros_like(log_score)
        step = log_score / score_norm
        step_norm = 1.0
    if step_norm > trust_radius:
        step = step * (trust_radius / step_norm)
    return np.asarray(step, dtype=np.float64)


def _predicted_loglikelihood_gain(
    log_score: npt.NDArray[np.float64],
    log_average_information: npt.NDArray[np.float64],
    step: npt.NDArray[np.float64],
) -> float:
    information = _symmetrize(log_average_information)
    return float(log_score @ step - 0.5 * step @ information @ step)


def _default_initial_components(y: npt.NDArray[np.float64]) -> VarianceComponents:
    empirical_second_moment = max(float(y @ y / y.shape[0]), 1e-6)
    share = empirical_second_moment / 3.0
    return VarianceComponents(share, share, share)


def _validate_optimizer_controls(
    *,
    lower_bound: float,
    upper_bound: float | None,
    maxiter: int,
    initial_trust_radius: float,
    max_trust_radius: float,
    gradient_tol: float,
    step_tol: float,
) -> tuple[float, float | None]:
    if lower_bound <= 0.0 or not np.isfinite(lower_bound):
        raise ValueError("lower_bound must be finite and positive")
    if upper_bound is not None and (upper_bound <= lower_bound or not np.isfinite(upper_bound)):
        raise ValueError("upper_bound must be finite and greater than lower_bound")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if initial_trust_radius <= 0.0 or not np.isfinite(initial_trust_radius):
        raise ValueError("initial_trust_radius must be finite and positive")
    if max_trust_radius < initial_trust_radius or not np.isfinite(max_trust_radius):
        raise ValueError("max_trust_radius must be finite and at least initial_trust_radius")
    if gradient_tol <= 0.0 or not np.isfinite(gradient_tol):
        raise ValueError("gradient_tol must be finite and positive")
    if step_tol <= 0.0 or not np.isfinite(step_tol):
        raise ValueError("step_tol must be finite and positive")
    return float(np.log(lower_bound)), None if upper_bound is None else float(np.log(upper_bound))


def optimize_variance_components(
    y: npt.ArrayLike,
    additive: MatvecKernel,
    interaction: MatvecKernel,
    *,
    initial: VarianceComponents | None = None,
    covariates: npt.ArrayLike | None = None,
    lower_bound: float = 1e-10,
    upper_bound: float | None = None,
    maxiter: int = 1000,
    initial_trust_radius: float = 1.0,
    max_trust_radius: float = 4.0,
    gradient_tol: float = 1e-5,
    step_tol: float = 1e-8,
    logdet_probe_mode: LogdetProbeMode = "rademacher",
    num_logdet_probes: int = 16,
    lanczos_rank: int = 32,
    seed: int | None = 0,
    cg_rtol: float = 1e-6,
    cg_atol: float = 0.0,
    cg_maxiter: int | None = None,
    logger: logging.Logger | None = None,
) -> VarianceComponentFit:
    """Optimize variance components with a matvec-only likelihood objective.

    Optimization is performed over log variance components with a bounded
    trust-region AI-REML update. Each step uses the analytic log-scale score
    and average-information matrix from the likelihood evaluation. The optimizer
    does not materialize the covariance matrix; component kernels may cache
    their own matrices.

    **Arguments:**

    - `y`: One-dimensional phenotype or molecular phenotype vector.
    - `additive`: Additive kernel exposing `matvec`.
    - `interaction`: Same-haplotype interaction kernel exposing `matvec`.
    - `initial`: Optional positive starting variance components.
    - `covariates`: Optional fixed-effect design matrix `X`.
    - `lower_bound`: Positive lower bound for each variance component.
    - `upper_bound`: Optional finite upper bound for each variance component.
    - `maxiter`: Maximum optimizer iterations.
    - `initial_trust_radius`: Initial Euclidean trust-region radius in log variance space.
    - `max_trust_radius`: Maximum Euclidean trust-region radius in log variance space.
    - `gradient_tol`: Infinity-norm convergence tolerance for the log-scale score.
    - `step_tol`: Euclidean-norm stopping tolerance for projected log-scale steps.
    - `logdet_probe_mode`: `"rademacher"`, `"normal"`, or `"basis"`.
    - `num_logdet_probes`: Number of random probes for stochastic modes.
    - `lanczos_rank`: Maximum Lanczos rank for each probe.
    - `seed`: Random seed for stochastic probes.
    - `cg_rtol`: Relative tolerance for conjugate gradients.
    - `cg_atol`: Absolute tolerance for conjugate gradients.
    - `cg_maxiter`: Optional maximum conjugate-gradient iterations.
    - `logger`: Optional progress logger; omitted for silent library use.

    **Returns:**

    - Fitted variance components and optimizer status.
    """
    response = _validate_response(y)
    covariate_matrix = _validate_covariates(covariates, n_observations=response.shape[0])
    lower_log, upper_log = _validate_optimizer_controls(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        maxiter=maxiter,
        initial_trust_radius=initial_trust_radius,
        max_trust_radius=max_trust_radius,
        gradient_tol=gradient_tol,
        step_tol=step_tol,
    )
    if additive.shape != interaction.shape:
        raise ValueError("additive and interaction kernels must have the same shape")
    if additive.shape[0] != response.shape[0]:
        raise ValueError("component dimensions must match y length")

    starting = initial or _default_initial_components(response)
    initial_values = np.maximum(_validate_variance_components(starting), lower_bound)

    def evaluate(log_values: npt.NDArray[np.float64]) -> GaussianLogLikelihood:
        variance_components = _variance_components_from_log(log_values)
        return gaussian_log_likelihood(
            response,
            additive,
            interaction,
            variance_components,
            covariates=covariate_matrix,
            logdet_probe_mode=logdet_probe_mode,
            num_logdet_probes=num_logdet_probes,
            lanczos_rank=lanczos_rank,
            seed=seed,
            cg_rtol=cg_rtol,
            cg_atol=cg_atol,
            cg_maxiter=cg_maxiter,
        )

    log_values = _clip_log_variances(np.log(initial_values), lower_log=lower_log, upper_log=upper_log)
    if logger is not None:
        logger.info("Evaluating initial likelihood")
    current = evaluate(log_values)
    trust_radius = initial_trust_radius
    accepted_steps = 0
    rejected_steps = 0
    message = _MESSAGE_MAXITER
    n_iterations = 0

    for iteration in range(1, maxiter + 1):
        n_iterations = iteration
        score_norm = float(np.linalg.norm(current.log_score, ord=np.inf))
        if logger is not None:
            report = logger.info if iteration == 1 or iteration % 10 == 0 else logger.debug
            report(
                "Iteration %d: log_likelihood=%.9g, score=%.3g, additive=%.6g, HxH=%.6g, "
                "residual=%.6g, accepted=%d, rejected=%d, trust_radius=%.3g",
                iteration,
                current.log_likelihood,
                score_norm,
                current.variance_components.sigma_a2,
                current.variance_components.sigma_h2,
                current.variance_components.sigma_e2,
                accepted_steps,
                rejected_steps,
                trust_radius,
            )
        if score_norm <= gradient_tol:
            message = _MESSAGE_SCORE_CONVERGED
            break

        proposed_step = _solve_ai_step(
            current.log_score,
            current.log_average_information,
            trust_radius=trust_radius,
        )
        candidate_log_values = _clip_log_variances(
            log_values + proposed_step,
            lower_log=lower_log,
            upper_log=upper_log,
        )
        actual_step = candidate_log_values - log_values
        step_norm = float(np.linalg.norm(actual_step))
        if step_norm <= step_tol:
            message = _MESSAGE_STEP_CONVERGED
            break

        predicted_gain = _predicted_loglikelihood_gain(
            current.log_score,
            current.log_average_information,
            actual_step,
        )
        if predicted_gain <= 0.0 or not np.isfinite(predicted_gain):
            trust_radius *= _TRUST_SHRINK
            rejected_steps += 1
            continue

        try:
            candidate = evaluate(candidate_log_values)
        except ValueError as error:
            if logger is not None:
                logger.debug("Iteration %d: rejected candidate: %s", iteration, error)
            trust_radius *= _TRUST_SHRINK
            rejected_steps += 1
            continue

        actual_gain = candidate.log_likelihood - current.log_likelihood
        gain_ratio = actual_gain / predicted_gain
        if actual_gain > 0.0 and gain_ratio >= _MIN_GAIN_RATIO:
            current = candidate
            log_values = candidate_log_values
            accepted_steps += 1
            if gain_ratio > _HIGH_GAIN_RATIO and step_norm >= _BOUNDARY_STEP_FRACTION * trust_radius:
                trust_radius = min(max_trust_radius, _TRUST_GROW * trust_radius)
        else:
            rejected_steps += 1

        if gain_ratio < _LOW_GAIN_RATIO:
            trust_radius *= _TRUST_SHRINK
        if trust_radius <= step_tol:
            message = _MESSAGE_RADIUS_CONVERGED
            break
    else:
        score_norm = float(np.linalg.norm(current.log_score, ord=np.inf))
        if score_norm <= gradient_tol:
            message = _MESSAGE_SCORE_CONVERGED

    return VarianceComponentFit(
        optimizer="ai_trust_region",
        variance_components=current.variance_components,
        log_likelihood=current.log_likelihood,
        negative_log_likelihood=-current.log_likelihood,
        success=message in _CONVERGED_MESSAGES,
        message=message,
        n_iterations=n_iterations,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        trust_radius=trust_radius,
        fixed_effects=current.fixed_effects,
        log_gradient=-current.log_score,
        log_average_information=current.log_average_information,
        logdet_method=current.logdet_method,
        num_logdet_probes=current.num_logdet_probes,
        lanczos_rank=current.lanczos_rank,
    )
