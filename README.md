# intergalactic

`intergalactic` implements operators for the same-haplotype cis-regulatory interaction kernel. The additive GRM uses LinearARG products; the interaction component caches an individual-level kernel constructed with bounded batches.

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

The targeted CLI wraps one regional LinearARG fit:

```bash
intergalactic fit path/to/linear_arg.h5 \
  --region chr1:100000-200000 \
  --phenotype phenotypes.tsv \
  --phenotype-id-column IID \
  --phenotype-column trait \
  --covariates covariates.tsv \
  --covariate-id-column IID \
  --covariate-columns age sex pc1 pc2 \
  --output fit.json
```

Phenotype and covariate tables are loaded with Polars, joined to LinearARG individual IDs by sample key, and exported to arrays only after row order and null checks. Automatic covariate selection excludes the sample ID column and the identifier columns `FID` and `IID`; use `--covariate-columns` to select columns explicitly. When `--region` is supplied, the selected LinearARG block or root bundle is filtered to that interval before kernel construction.

Use `--allow-sample-subset` when the phenotype or covariate table covers only a subset of ARG individuals. The fit retains individuals with values in both tables and selects their two adjacent haplotype rows in ARG order. Without this flag, missing values raise an error.

Kernel construction uses up to 32 operator right-hand sides at a time. Set `--kernel-batch-size` to a smaller positive integer if the ARG's internal workspace exceeds your memory allocation. The H×H component caches an `n × n` individual kernel; construction uses haplotype Gram matrices and batched variant weights rather than variant-square matrices. See the covariance-model documentation for centering and interaction conventions.
