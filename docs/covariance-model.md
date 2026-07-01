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

The likelihood path evaluates:

```text
log p(y | sigma_A^2, sigma_H^2, sigma_e^2)
```

under the zero-mean Gaussian covariance above, but the implementation does not materialize `K_A`, `K_H`, or the combined covariance matrix. It builds a covariance `LinearOperator`:

```text
v -> sigma_A^2 K_A v + sigma_H^2 K_H v + sigma_e^2 v
```

The quadratic form `y^T K^-1 y` is computed by conjugate gradients. The log determinant is estimated with Lanczos quadrature using covariance matvecs. For small deterministic tests, `logdet_probe_mode="basis"` and `lanczos_rank >= n` produce the full-basis Lanczos result without constructing the dense covariance matrix.

For variance component `i` with component matrix `K_i`, the likelihood score is:

```text
0.5 * (y^T P K_i P y - tr(P K_i))
```

where `P = K^-1` in the current zero-mean ML implementation. Trace terms are estimated with the same matvec-only probe machinery used by the likelihood path. The reported Hessian-like matrix is the AI-REML average-information matrix:

```text
AI_ij = 0.5 * y^T P K_i P K_j P y
```

The optimizer works on the log-variance scale so fitted components remain positive:

```text
theta = log([sigma_A^2, sigma_H^2, sigma_e^2])
```

At each iteration it evaluates the analytic log-scale score and the chain-rule AI approximation, then proposes a bounded trust-region Newton-like step:

```text
AI_log(theta) delta ~= score_log(theta)
```

The step is accepted only when the matvec-only likelihood evaluation improves enough relative to the local quadratic model. Stochastic probe modes (`"rademacher"` and `"normal"`) trade log-determinant accuracy for fewer matvecs. REML fixed-effect adjustments are a separate future extension.
