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

The CLI defaults to restricted maximum likelihood (REML). Python calls to
`gaussian_log_likelihood` and `optimize_variance_components` select it with
`likelihood_method="reml"`; their default remains `"ml"` for compatibility.
`--likelihood-method ml` selects the earlier profiled ML criterion in the CLI.
No intercept is added implicitly. Include a constant covariate if the model
requires an intercept. REML requires a full-column-rank design `X` with `p < n`.

### Restricted likelihood and error contrasts

Following the error-contrast formulation in Searle, Casella, and McCulloch,
[*Variance Components*, Chapter 6](https://onlinelibrary.wiley.com/doi/10.1002/9780470316856.ch6),
construct an orthonormal basis `Q` with `Q.T X = 0` and `Q.T Q = I`.
A full SVD of the column-scaled design supplies this basis and a numerical rank
check. Define:

```text
z = Q.T y
A = Q.T K_A Q
H = Q.T K_H Q
W = sigma_A^2 A + sigma_H^2 H + sigma_e^2 I
P = Q W^-1 Q.T
```

The optimizer builds and caches `A` and `H` once, using bounded batches of
operator products (CLI `--kernel-batch-size`). Normalization and centering are
applied to the original component kernels before projection; projected kernels
are not renormalized. The model retains the full additive and same-haplotype
interaction components. Storage is quadratic in sample count, including the
`n` by `n-p` basis and two `(n-p)` by `(n-p)` projected kernels; no variant-square
matrix is constructed. The additive ARG is no longer traversed during REML
iterations.

The reported restricted log likelihood uses the conventional determinant term:

```text
ell_R = -0.5 [(n-p) log(2 pi) + log|V| + log|X.T V^-1 X| + y.T P y]
      = -0.5 [(n-p) log(2 pi) + log|W| + log|X.T X| + z.T W^-1 z]
```

Thus the reported value differs from the orthonormal-contrast log density by
`-0.5 log|X.T X|`, a variance-parameter-independent constant. The result field
`log_determinant` includes this correction. Rescaling `X` changes that constant,
not the REML estimates. With no covariates, the correction is zero and REML
reduces to zero-mean ML. Do not compare REML likelihoods across different
fixed-effect designs.

For `alpha = P y`, the score and average-information matrix are:

```text
score_i = 0.5 [alpha.T K_i alpha - tr(P K_i)]
AI_ij = 0.5 (K_i alpha).T P (K_j alpha)
```

These are evaluated equivalently in contrast space by the existing numerical
routines: CG for solves and shared random probes for the trace terms. Lanczos
quadrature estimates `log|W|`. `logdet_probe_mode="basis"` with
`lanczos_rank >= n-p` gives deterministic full-basis evaluations for small
problems. Stochastic modes remain approximations; the sampled score is not
necessarily the exact derivative of a finite-rank sampled log determinant.
With 16 probes, each evaluation uses 20 CG solves (response, 16 probes, and
three average-information right-hand sides), independent of the covariate count.

Fixed effects are recovered only after optimization: solve `W u = z`, set
`alpha = Q u`, and solve `X beta_hat = y - V alpha` by scaled least squares.
This is the GLS estimate at the fitted covariance, without separate solves for
all columns of `V^-1 X`. The standalone likelihood API also returns GLS fixed
effects and an observation-space residual.

### Optimization and ML compatibility

The optimizer works in log variance components, using bounded trust-region
average-information steps and accepting candidates according to likelihood gain.
The component lower bounds, CG tolerances, probes, Lanczos rank, and seed retain
their previous meanings. JSON output identifies `likelihood_method` and
`residual_degrees_of_freedom`; old ML and new REML likelihood values are different
criteria.

ML continues to profile fixed effects with
`beta_hat = (X.T V^-1 X)^-1 X.T V^-1 y`, solving for the covariates at each
evaluation. Its likelihood uses `n log(2 pi) + log|V|` and its score trace uses
`V^-1`, without the restricted-likelihood correction.
