# pattern: Functional Core

from dataclasses import dataclass

import numpy as np

from intergalactic import (
    covariance_matrix,
    dense_kernel_matrix,
    gaussian_log_likelihood,
    optimize_variance_components,
    VarianceComponents,
)


@dataclass(frozen=True)
class _DenseKernel:
    matrix: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def matvec(self, values):
        return self.matrix @ values

    def matmat(self, values):
        return self.matrix @ values


def test_dense_kernel_matrix_materializes_operator_columns():
    kernel = _DenseKernel(np.array([[2.0, 0.5], [0.5, 3.0]]))

    np.testing.assert_allclose(dense_kernel_matrix(kernel), kernel.matrix)


def test_gaussian_log_likelihood_matches_manual_cholesky_formula():
    y = np.array([0.3, -1.2, 0.7])
    additive = np.array([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.8]])
    interaction = np.array([[0.5, 0.1, 0.0], [0.1, 0.7, 0.2], [0.0, 0.2, 0.9]])
    variances = VarianceComponents(sigma_a2=0.4, sigma_h2=0.25, sigma_e2=0.8)

    covariance = covariance_matrix(additive, interaction, variances)
    cholesky = np.linalg.cholesky(covariance)
    alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, y))
    expected = -0.5 * (y @ alpha + 2.0 * np.log(np.diag(cholesky)).sum() + y.shape[0] * np.log(2.0 * np.pi))

    assert gaussian_log_likelihood(y, additive, interaction, variances).log_likelihood == expected


def test_optimizer_improves_negative_log_likelihood_from_initial_values():
    y = np.array([1.2, -0.3, 0.7, -1.5, 0.4])
    additive = _DenseKernel(np.diag([1.0, 0.8, 1.2, 0.5, 1.5]))
    interaction = _DenseKernel(
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
    initial = VarianceComponents(sigma_a2=0.05, sigma_h2=0.05, sigma_e2=0.05)

    fit = optimize_variance_components(y, additive, interaction, initial=initial)
    initial_nll = -gaussian_log_likelihood(y, additive.matrix, interaction.matrix, initial).log_likelihood

    assert fit.success
    assert fit.negative_log_likelihood < initial_nll
    assert fit.variance_components.sigma_a2 > 0.0
    assert fit.variance_components.sigma_h2 > 0.0
    assert fit.variance_components.sigma_e2 > 0.0
