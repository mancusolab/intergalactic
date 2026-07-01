# Same-Haplotype Cis-Regulatory Interaction Kernel

Let `H` be the stacked haplotype matrix with adjacent rows for the two haplotypes of each diploid individual. Let `C` be the diploid combiner that sums each adjacent pair.

The additive component is:

```text
K_A = C H H^T C^T
```

The same-haplotype cis-regulatory interaction kernel is:

```text
K_H = C Phi(H) Phi(H)^T C^T
```

where `Phi(h) = h ⊗ h` is applied row-wise to haplotypes. For ordered pairs including self-interactions:

```text
Phi(h_i)^T Phi(h_j) = (h_i^T h_j)^2
```

For unordered off-diagonal interactions, self-pair terms are removed:

```text
0.5 * ((H H^T)^2 - H H^T)
```

The implemented exact backend avoids materializing `Phi(H)`. For a vector `v`, it lifts to haplotype weights `w = C^T v`, forms a local weighted co-carriage matrix:

```text
B = H^T diag(w) H
```

and evaluates each haplotype row with:

```text
u_i = h_i^T B h_i
```

The diploid result is `C u`. Block-local kernels apply the same identity per regulatory block and sum the products.

This differs from genotype-level GxG:

```text
Phi(C H) Phi(C H)^T
```

`Phi(C H)` expands dosage genotypes and includes maternal-by-paternal cross terms. The same-haplotype formulation applies interactions before diploid collapse, so interaction features stay on one physical haplotype.

The variance-component covariance operator is:

```text
K = sigma_A^2 K_A + sigma_H^2 K_H + sigma_e^2 I
```

Trace normalization scales a component so `trace(K) / n = 1`. Diagonal normalization applies `D^-1/2 K D^-1/2`, where `D = diag(K)`.

## Likelihood evaluation

The exact maximum-likelihood path evaluates:

```text
log p(y | sigma_A^2, sigma_H^2, sigma_e^2)
```

under the zero-mean Gaussian covariance above. It materializes `K_A` and `K_H` as dense `n x n` matrices, builds the covariance matrix, and computes the log determinant and quadratic form by Cholesky decomposition.

Variance-component optimization uses a log-parameterized L-BFGS-B objective so fitted components remain positive:

```text
theta = log([sigma_A^2, sigma_H^2, sigma_e^2])
```

This is an exact dense likelihood backend. It is appropriate for testing, small cohorts, or local windows where dense individual-level covariance matrices are acceptable. Matrix-free REML, stochastic trace estimation, and average-information updates are separate scalable backends.
