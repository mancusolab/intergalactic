# intergalactic

`intergalactic` implements matrix-free operators for the same-haplotype cis-regulatory interaction kernel.

The package targets variance-component models over stacked phased haplotypes. If `H` contains adjacent maternal and paternal haplotype rows and `C` sums each pair back to one diploid individual, the interaction component is

```text
K_H = C Phi(H) Phi(H)^T C^T
```

where `Phi(h) = h ⊗ h` is applied to each physical haplotype. This is not genotype-level GxG:

```text
Phi(C H) Phi(C H)^T
```

because genotype-level GxG includes cross-haplotype maternal-by-paternal terms. Those terms are not the target for cis-regulatory grammar.

See [docs/covariance-model.md](docs/covariance-model.md) for the covariance model and operator identities.

The likelihood API evaluates and optimizes the marginal Gaussian model:

```text
y ~ N(X beta, sigma_A^2 K_A + sigma_H^2 K_H + sigma_e^2 I)
```

`gaussian_log_likelihood` profiles optional fixed effects by generalized least squares, so covariates enter as the mean term `X @ beta`. `optimize_variance_components` builds a covariance `LinearOperator` and uses only `matvec` calls. The quadratic form is computed with conjugate gradients, and the log determinant is estimated with Lanczos quadrature. The optimizer uses bounded trust-region AI updates from the analytic log-scale score and AI-REML average-information matrix. `logdet_probe_mode="basis"` is deterministic for small tests; stochastic probe modes are intended for larger cohorts.
