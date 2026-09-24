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

For unordered off-diagonal interactions, remove the self-pair features:

```text
S = H ⊙ H
0.5 * ((H H^T) ⊙ (H H^T) - S S^T)
```

Here `⊙` denotes elementwise multiplication. For binary, uncentered haplotypes, `S = H`. After centering, the squared-feature correction is required; subtracting `H H^T` would give a different kernel.

The exact backend constructs columns of the haplotype Gram matrix in batches:

```text
G[:, B] = H (H^T E_B)
K_H = C (G ⊙ G) C^T
```

`E_B` selects a small batch of haplotype basis vectors. For centered haplotypes, centering projections are applied before and after the operator products. Variant masks restrict each regulatory block. Construction batches do not partition the interaction model: all cross-variant pairs within a block are retained. Block-local kernels sum the resulting individual kernels.

The individual kernel is cached once and reused throughout likelihood evaluation. There is no variant-square identity, weighted co-carriage matrix, full haplotype matrix, or pair-feature matrix. With `m = 2n` haplotypes, `p` variants, and batch size `b`, construction arrays scale as `O(m² + pb + mb)` plus the LinearARG operator's own workspace. The cached interaction matrix is `O(n²)`. The unordered mode constructs the squared-feature correction using batches of variant columns.

The additive GRM remains `C H H^T C^T`. Its normalization diagonal is computed from squared column norms of `H^T C^T`, also in batches. Trace and diagonal normalization therefore retain their definitions without constructing a variant-sized identity.

The existing `backend="dense_window"` name remains supported. Both kernel constructors accept `batch_size` (default 32); the CLI exposes it as `--kernel-batch-size`.

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
log p(y | sigma_A^2, sigma_H^2, sigma_e^2, beta_hat)
```

under the Gaussian covariance above. The interaction component caches `K_H`; the additive component and combined covariance use operator products. The likelihood builds a covariance `LinearOperator`:

```text
v -> sigma_A^2 K_A v + sigma_H^2 K_H v + sigma_e^2 v
```

With no covariates, the quadratic form `y^T K^-1 y` is computed by conjugate gradients. With a fixed-effect design matrix `X`, the implementation profiles `beta` by generalized least squares:

```text
beta_hat = (X^T K^-1 X)^-1 X^T K^-1 y
```

The residual quadratic form `(y - X beta_hat)^T K^-1 (y - X beta_hat)` is then used in the profiled ML likelihood. The solves for `K^-1 y` and each column of `K^-1 X` use covariance matvecs. The log determinant is estimated with Lanczos quadrature using covariance matvecs. For small deterministic tests, `logdet_probe_mode="basis"` and `lanczos_rank >= n` produce the full-basis Lanczos result without constructing the dense covariance matrix.

For variance component `i` with component matrix `K_i`, the likelihood score is:

```text
0.5 * (alpha^T K_i alpha - tr(P K_i))
```

where `P = K^-1` and `alpha = P (y - X beta_hat)`. Trace terms are estimated with the same matvec-only probe machinery used by the likelihood path. The reported Hessian-like matrix is the AI-REML average-information matrix:

```text
AI_ij = 0.5 * (K_i alpha)^T P (K_j alpha)
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
