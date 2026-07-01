# pattern: Functional Core

import numpy as np

from intergalactic import (
    covariance_operator,
    gaussian_log_likelihood,
    optimize_variance_components,
    VarianceComponents,
)


class _MatvecOnlyKernel:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.matvec_calls = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def matvec(self, values):
        self.matvec_calls += 1
        return self.matrix @ np.asarray(values)

    def matmat(self, values):
        raise AssertionError("likelihood code must not call matmat")

    def __array__(self):
        raise AssertionError("likelihood code must not convert kernels to dense arrays")


def _optimizer_example():
    y = np.array([1.2, -0.3, 0.7, -1.5, 0.4])
    additive = _MatvecOnlyKernel(np.diag([1.0, 0.8, 1.2, 0.5, 1.5]))
    interaction = _MatvecOnlyKernel(
        np.array(
            [
                [0.7, 0.1, 0.0, 0.0, 0.1],
                [0.1, 0.9, 0.2, 0.0, 0.0],
                [0.0, 0.2, 0.6, 0.1, 0.0],
                [0.0, 0.0, 0.1, 1.1, 0.2],
                [0.1, 0.0, 0.0, 0.2, 0.8],
            ]
        )
    )
    return y, additive, interaction


def test_covariance_operator_uses_component_matvecs_only():
    additive_matrix = np.array([[2.0, 0.5], [0.5, 3.0]])
    interaction_matrix = np.array([[1.5, 0.2], [0.2, 0.8]])
    additive = _MatvecOnlyKernel(additive_matrix)
    interaction = _MatvecOnlyKernel(interaction_matrix)
    variances = VarianceComponents(sigma_a2=0.4, sigma_h2=0.25, sigma_e2=0.8)

    operator = covariance_operator(additive, interaction, variances)
    vector = np.array([1.25, -0.5])
    expected = (
        variances.sigma_a2 * (additive_matrix @ vector)
        + variances.sigma_h2 * (interaction_matrix @ vector)
        + variances.sigma_e2 * vector
    )

    np.testing.assert_allclose(operator.matvec(vector), expected)
    assert additive.matvec_calls == 1
    assert interaction.matvec_calls == 1


def test_gaussian_log_likelihood_matches_cholesky_reference_without_dense_kernel_access():
    y = np.array([0.3, -1.2, 0.7])
    additive_matrix = np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.8]])
    interaction_matrix = np.array([[0.5, 0.1, 0.0], [0.1, 0.7, 0.2], [0.0, 0.2, 0.9]])
    additive = _MatvecOnlyKernel(additive_matrix)
    interaction = _MatvecOnlyKernel(interaction_matrix)
    variances = VarianceComponents(sigma_a2=0.4, sigma_h2=0.25, sigma_e2=0.8)

    covariance = (
        variances.sigma_a2 * additive_matrix
        + variances.sigma_h2 * interaction_matrix
        + variances.sigma_e2 * np.eye(y.shape[0])
    )
    cholesky = np.linalg.cholesky(covariance)
    alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, y))
    expected = -0.5 * (y @ alpha + 2.0 * np.log(np.diag(cholesky)).sum() + y.shape[0] * np.log(2.0 * np.pi))

    result = gaussian_log_likelihood(
        y,
        additive,
        interaction,
        variances,
        logdet_probe_mode="basis",
        lanczos_rank=y.shape[0],
        cg_rtol=1e-12,
        cg_atol=0.0,
    )

    np.testing.assert_allclose(result.log_likelihood, expected, rtol=1e-9, atol=1e-9)
    assert result.num_logdet_probes == y.shape[0]
    assert additive.matvec_calls > 0
    assert interaction.matvec_calls > 0


def test_gaussian_log_likelihood_reports_score_and_average_information():
    y = np.array([0.3, -1.2, 0.7])
    additive_matrix = np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.8]])
    interaction_matrix = np.array([[0.5, 0.1, 0.0], [0.1, 0.7, 0.2], [0.0, 0.2, 0.9]])
    additive = _MatvecOnlyKernel(additive_matrix)
    interaction = _MatvecOnlyKernel(interaction_matrix)
    variances = VarianceComponents(sigma_a2=0.4, sigma_h2=0.25, sigma_e2=0.8)
    components = [additive_matrix, interaction_matrix, np.eye(y.shape[0])]

    covariance = (
        variances.sigma_a2 * additive_matrix
        + variances.sigma_h2 * interaction_matrix
        + variances.sigma_e2 * np.eye(y.shape[0])
    )
    precision = np.linalg.inv(covariance)
    alpha = precision @ y
    expected_score = 0.5 * np.array(
        [alpha @ component @ alpha - np.trace(precision @ component) for component in components]
    )
    expected_ai = np.array(
        [
            [0.5 * (component_i @ alpha) @ precision @ (component_j @ alpha) for component_j in components]
            for component_i in components
        ]
    )
    variance_vector = variances.as_array()

    result = gaussian_log_likelihood(
        y,
        additive,
        interaction,
        variances,
        logdet_probe_mode="basis",
        lanczos_rank=y.shape[0],
        cg_rtol=1e-12,
        cg_atol=0.0,
    )

    np.testing.assert_allclose(result.score, expected_score, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(result.average_information, expected_ai, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(result.log_score, variance_vector * expected_score, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        result.log_average_information,
        np.outer(variance_vector, variance_vector) * expected_ai,
        rtol=1e-9,
        atol=1e-9,
    )


def test_optimizer_improves_matvec_only_negative_log_likelihood_from_initial_values():
    y, additive, interaction = _optimizer_example()
    initial = VarianceComponents(sigma_a2=0.05, sigma_h2=0.05, sigma_e2=0.05)

    fit = optimize_variance_components(
        y,
        additive,
        interaction,
        initial=initial,
        logdet_probe_mode="basis",
        lanczos_rank=y.shape[0],
        cg_rtol=1e-10,
        cg_atol=0.0,
        maxiter=100,
    )
    initial_nll = -gaussian_log_likelihood(
        y,
        additive,
        interaction,
        initial,
        logdet_probe_mode="basis",
        lanczos_rank=y.shape[0],
        cg_rtol=1e-10,
        cg_atol=0.0,
    ).log_likelihood

    assert fit.success
    assert fit.optimizer == "ai_trust_region"
    assert fit.accepted_steps > 0
    assert fit.rejected_steps >= 0
    assert fit.trust_radius > 0.0
    assert fit.negative_log_likelihood < initial_nll
    assert fit.log_gradient.shape == (3,)
    assert fit.log_average_information.shape == (3, 3)
    assert fit.variance_components.sigma_a2 > 0.0
    assert fit.variance_components.sigma_h2 > 0.0
    assert fit.variance_components.sigma_e2 > 0.0


def test_optimizer_reports_maximum_iterations_as_unsuccessful():
    y, additive, interaction = _optimizer_example()

    fit = optimize_variance_components(
        y,
        additive,
        interaction,
        initial=VarianceComponents(sigma_a2=0.05, sigma_h2=0.05, sigma_e2=0.05),
        logdet_probe_mode="basis",
        lanczos_rank=y.shape[0],
        cg_rtol=1e-10,
        cg_atol=0.0,
        maxiter=1,
    )

    assert not fit.success
    assert fit.message == "maximum iterations reached"
    assert fit.n_iterations == 1
