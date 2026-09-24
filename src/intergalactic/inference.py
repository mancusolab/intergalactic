# pattern: Functional Core
"""Restricted likelihood-ratio inference with exact dense likelihood evaluations."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import chi2

from intergalactic.likelihood import (
    _prepare_restricted_model,
    _validate_covariates,
    _validate_response,
    MatvecKernel,
)

Array = npt.NDArray[np.float64]
_RESIDUAL_FLOOR = 1e-10
_KKT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class _Fit:
    components: Array
    log_likelihood: float
    success: bool
    message: str
    covariance: Array
    n_iterations: int


def _fit_covariance(response: Array, kernels: tuple[Array, ...], *, maxiter: int, initial: Array | None = None) -> _Fit:
    """Fit nonnegative components after scaling response and kernel traces."""
    n = response.size
    response_scale = float(response @ response / n)
    if len(kernels) == 1:
        # The joint-test null contains only the identity residual kernel.
        value = -0.5 * n * (np.log(2 * np.pi * response_scale) + 1)
        return _Fit(
            np.array([response_scale]), float(value), True, "analytic residual-only fit", response_scale * kernels[0], 0
        )
    scales = np.array([np.trace(kernel) / n for kernel in kernels])
    matrices = tuple(kernel / scale for kernel, scale in zip(kernels, scales, strict=True))
    y = response / np.sqrt(response_scale)
    identity = np.eye(n)

    def objective(values: Array) -> tuple[float, Array]:
        covariance = sum((value * matrix for value, matrix in zip(values, matrices, strict=True)), np.zeros((n, n)))
        try:
            factor = cho_factor(covariance, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return float("inf"), np.zeros_like(values)
        alpha = cho_solve(factor, y, check_finite=False)
        inverse = cho_solve(factor, identity, check_finite=False)
        logdet = 2 * np.log(np.diag(factor[0])).sum()
        value = 0.5 * (logdet + y @ alpha) / n
        gradient = np.array([0.5 * (np.sum(inverse * matrix.T) - alpha @ matrix @ alpha) / n for matrix in matrices])
        return float(value), gradient

    starts = [np.full(len(kernels), 1 / len(kernels))]
    boundary = np.zeros(len(kernels))
    boundary[-1] = 1.0
    starts.append(boundary)
    if initial is not None:
        starts.append(initial * scales / response_scale)
    candidates: list[tuple[float, Array, bool, str, int]] = []
    bounds = [(0.0, None)] * (len(kernels) - 1) + [(_RESIDUAL_FLOOR, None)]
    for start in starts:
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40},
        )
        value, gradient = objective(result.x)
        projected = gradient.copy()
        projected[(result.x <= 1e-8) & (gradient > 0)] = 0.0
        success = bool(
            result.success
            and np.isfinite(value)
            and np.max(np.abs(projected)) <= _KKT_TOLERANCE
            and result.x[-1] > 10 * _RESIDUAL_FLOOR
        )
        message = str(result.message) if success else "optimizer failed convergence/KKT or positive-residual checks"
        candidates.append((value, result.x, success, message, int(result.nit)))
    # Never replace a better unconverged candidate with an inferior converged fit.
    value, values, success, message, n_iterations = min(candidates, key=lambda item: item[0])
    components = values * response_scale / scales
    covariance = sum((value * matrix for value, matrix in zip(components, kernels, strict=True)), np.zeros((n, n)))
    log_likelihood = -n * value - 0.5 * n * (np.log(2 * np.pi) + np.log(response_scale))
    return _Fit(components, float(log_likelihood), success, message, covariance, n_iterations)


def _nested_fits(response: Array, kernels: tuple[Array, ...], target: str, maxiter: int) -> tuple[_Fit, _Fit, float]:
    null_kernels = (kernels[0], kernels[2]) if target == "interaction" else (kernels[2],)
    null = _fit_covariance(response, null_kernels, maxiter=maxiter)
    initial = (
        np.array([null.components[0], 0.0, null.components[-1]])
        if target == "interaction"
        else np.array([0.0, 0.0, null.components[0]])
    )
    full = _fit_covariance(response, kernels, maxiter=maxiter, initial=initial)
    statistic = 2 * (full.log_likelihood - null.log_likelihood)
    return null, full, float(max(0.0, statistic))


def _fit_payload(fit: _Fit, names: tuple[str, ...], correction: float) -> dict[str, Any]:
    return {
        "variance_components": dict(zip(names, fit.components.tolist(), strict=True)),
        "log_likelihood": fit.log_likelihood - correction,
        "success": fit.success,
        "message": fit.message,
        "n_iterations": fit.n_iterations,
    }


def restricted_likelihood_ratio_test(
    y: npt.ArrayLike,
    additive: MatvecKernel,
    interaction: MatvecKernel,
    *,
    covariates: npt.ArrayLike | None = None,
    target: Literal["interaction", "joint"] = "interaction",
    method: Literal["bootstrap", "asymptotic"] = "bootstrap",
    num_bootstrap: int = 199,
    seed: int = 0,
    maxiter: int = 500,
    projection_batch_size: int = 32,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Test HxH conditional on additive variance, or test both genetic components.

    Bootstrap calibration simulates under the fitted composite null and refits
    both models. It is a plug-in parametric bootstrap, not an exact finite-sample
    test. The asymptotic option is a screening approximation for one component
    with an interior additive nuisance estimate. No p-value is reported if
    identifiability, optimization, or bootstrap fitting checks fail.
    """
    if target not in {"interaction", "joint"} or method not in {"bootstrap", "asymptotic"}:
        raise ValueError("target must be interaction/joint and method must be bootstrap/asymptotic")
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if method == "bootstrap" and (not isinstance(num_bootstrap, int) or num_bootstrap < 1):
        raise ValueError("num_bootstrap must be a positive integer")
    response = _validate_response(y)
    design = _validate_covariates(covariates, n_observations=response.size)
    model = _prepare_restricted_model(response, design, additive, interaction, projection_batch_size)
    z = model.response
    kernels = (model.additive.matrix, model.interaction.matrix, np.eye(z.size))
    output: dict[str, Any] = {
        "target": target,
        "method": "parametric_bootstrap" if method == "bootstrap" else "asymptotic_50_50_mixture",
        "likelihood_method": "reml",
        "likelihood_evaluation": "dense_cholesky",
        "statistic": None,
        "p_value": None,
        "success": False,
        "reason": None,
        "residual_degrees_of_freedom": z.size,
        "seed": seed,
        "null_fit": None,
        "alternative_fit": None,
        "bootstrap_requested": num_bootstrap if method == "bootstrap" else 0,
        "bootstrap_completed": 0,
        "monte_carlo_standard_error": None,
        "minimum_p_value": 1 / (num_bootstrap + 1) if method == "bootstrap" else None,
    }
    if z @ z <= np.finfo(float).eps * max(float(response @ response), np.finfo(float).tiny):
        output["reason"] = "no residual phenotype variation after covariate projection"
        return output
    norms = np.array([np.linalg.norm(kernel) for kernel in kernels])
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        output["reason"] = "zero or nonfinite projected kernel"
        return output
    gram = np.array([[np.sum(left * right) for right in kernels] for left in kernels]) / np.outer(norms, norms)
    if np.linalg.eigvalsh(gram)[0] <= 1e-10:
        output["reason"] = "projected covariance components are not numerically identifiable"
        return output
    for kernel in kernels[:2]:
        eigenvalues = np.linalg.eigvalsh(kernel)
        if eigenvalues[0] < -1e-8 * max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny):
            output["reason"] = "projected kernel is not positive semidefinite"
            return output
    if logger is not None:
        logger.info("Fitting exact REML null and alternative for %s test", target)
    null, full, statistic = _nested_fits(z, kernels, target, maxiter)
    correction = 0.5 * model.log_design_determinant
    null_names = ("sigma_a2", "sigma_e2") if target == "interaction" else ("sigma_e2",)
    output.update(
        statistic=statistic,
        null_fit=_fit_payload(null, null_names, correction),
        alternative_fit=_fit_payload(full, ("sigma_a2", "sigma_h2", "sigma_e2"), correction),
    )
    if not null.success or not full.success or full.log_likelihood < null.log_likelihood - 1e-7:
        output["reason"] = "null/alternative optimization or nested-likelihood check failed"
        return output
    if method == "asymptotic":
        if target != "interaction":
            output["reason"] = "the 50:50 mixture is only available for the single interaction component"
            return output
        additive_fraction = null.components[0] * np.trace(kernels[0]) / np.trace(null.covariance)
        if additive_fraction <= 1e-6:
            output["reason"] = "additive nuisance estimate is on/near its boundary; use parametric bootstrap"
            return output
        output.update(
            p_value=1.0 if statistic <= 1e-10 else float(0.5 * chi2.sf(statistic, 1)),
            success=True,
            calibration="asymptotic screening approximation; not finite-sample calibration",
        )
        return output
    if logger is not None:
        logger.info(
            "Parametric bootstrap: %d replicates, seed=%d; each replicate refits both models", num_bootstrap, seed
        )
    rng = np.random.default_rng(seed)
    root = np.linalg.cholesky(null.covariance)
    exceedances = 0
    for replicate in range(num_bootstrap):
        simulated = root @ rng.standard_normal(z.size)
        boot_null, boot_full, boot_statistic = _nested_fits(simulated, kernels, target, maxiter)
        if not boot_null.success or not boot_full.success or boot_full.log_likelihood < boot_null.log_likelihood - 1e-7:
            output["reason"] = f"bootstrap replicate {replicate + 1} failed; no replicates were silently discarded"
            return output
        exceedances += boot_statistic >= statistic - 1e-10
        output["bootstrap_completed"] = replicate + 1
        if logger is not None and ((replicate + 1) % 10 == 0 or replicate + 1 == num_bootstrap):
            logger.info("Bootstrap %s: %d/%d replicates completed", target, replicate + 1, num_bootstrap)
    probability = (exceedances + 1) / (num_bootstrap + 1)
    output.update(
        p_value=probability,
        success=True,
        bootstrap_exceedances=int(exceedances),
        monte_carlo_standard_error=float(np.sqrt(probability * (1 - probability) / num_bootstrap)),
        calibration="plug-in parametric bootstrap under the fitted null",
    )
    return output
