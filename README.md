# intergalactic

`intergalactic` implements operators for the same-haplotype cis-regulatory interaction kernel. Kernel construction uses bounded LinearARG products; REML caches both kernels in the space orthogonal to the covariates.

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

The CLI defaults to REML (`--likelihood-method reml`). It projects the phenotype and both kernels into orthonormal error contrasts once, then fits variance components with the existing CG solves, Lanczos log determinants, and bounded trust-region average-information updates. Covariates are eliminated during optimization; GLS fixed effects are recovered at the fitted covariance. No intercept is added automatically. Use `--likelihood-method ml` for the previous profiled ML criterion. Python APIs retain their ML default for compatibility; pass `likelihood_method="reml"` explicitly. `logdet_probe_mode="basis"` with sufficient Lanczos rank gives deterministic evaluations for small problems.

### Allele-frequency scaling

`fit` and `scan` now default to standardized haplotypes. Frequencies `p` are
estimated from the retained individuals after phenotype/covariate alignment;
variants with `p=0` or `p=1` in that cohort are excluded. An implicit operator
applies `H* = (H-p)/sqrt(p(1-p))` without materializing the haplotype matrix.
The additive GRM is `K_A = C H* H*.T C.T / (2m) = Z Z.T / m`, where
`Z = (G-2p)/sqrt(2p(1-p))` and `m` counts retained polymorphic variants.
It receives no further trace normalization.

H×H uses the same standardized haplotypes. Its individual-level interaction
features are centered after haplotype combination, then the kernel is scaled to
mean diagonal one. This weighting does not standardize each pair feature to unit
variance or remove overlap with additive effects. The default still includes
ordered pairs and self-pairs.

Use `--no-standardize` to reproduce the previous kernel construction; `--center`
then controls haplotype centering. `--no-normalize` disables H×H trace scaling
(and additive trace scaling in legacy mode); the standardized additive GRM
always retains its conventional `1/m` factor. JSON reports `kernel_scaling`,
`n_variants_before_standardization`, `n_monomorphic_excluded`, and the final
`n_variants`. Use a new output directory or `--overwrite` when changing scaling;
`--skip-existing` does not validate these settings.

Python kernel constructors retain their prior defaults. To build the same model:

```python
h = StandardizedHaplotypeOperator(retained_haplotype_operator)
a = AdditiveHaplotypeKernel(h, diploid_map, normalization=False, divisor=2 * h.shape[1])
k = SameHaplotypeInteractionKernel(h, diploid_map, center_features=True, normalization=True)
```

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

The CLI logs timestamped progress to stderr in the same format as jaxqtl. With `--output fit.json`, it also writes `fit.json.log` (overwritten for each run). Use `--log-file PATH` to choose a different log file; without `--output` or `--log-file`, logging is console-only and stdout contains the result JSON. Logs report loading, retained samples, kernel setup, optimizer progress, convergence, variance estimates, and elapsed time. Boundary variance estimates and unsuccessful convergence are reported as warnings. The first and every tenth optimizer iteration appear at INFO level; `--verbose` includes every iteration, arguments, covariate names, and failure tracebacks. Errors during fitting are logged and return exit status 1.

### Scan an expression BED file

```bash
intergalactic scan path/to/linear_arg.h5 \
  --phenotype-bed Whole_Blood.v8.normalized_expression.bed.gz \
  --covariates Whole_Blood.v8.covariates.tsv \
  --covariate-id-column IID \
  --allow-sample-subset \
  --chr chr1 \
  --output-dir results/Whole_Blood_chr1
```

`scan` streams a plain or gzip-compressed, tab-delimited expression BED in file order and fits each selected gene sequentially. Omit `--chr` to analyze all chromosomes; `1` and `chr1` are equivalent filters. Use `--gene GENE_ID` to select one exact identifier. The header must contain chromosome, start, end, gene ID, then sample IDs; GTEx's `#chr`, `start`, `end`, `gene_id` names are supported. Sample IDs and selected gene IDs must be unique. Each row must have the same number of fields as the header. Missing expression values (`NA`, `NaN`, `.`, `null`, or empty fields) are excluded when `--allow-sample-subset` is set; otherwise they produce an error.

The region is `[max(0, BED_start - window), BED_start + window)`, with `--window-bp 1000000` by default. This follows the existing GTEx wrapper: the BED start anchors the window, with no strand inference or coordinate increment. Use BED rows whose starts represent the intended expression loci.

Covariates are read once. Every gene uses the same fitting options available to `fit`; its samples are aligned independently, including any missing expression values. Results are written as `GENE_ID.json` with unsafe filename characters replaced by underscores; colliding names are rejected. `summary.tsv` records each gene's status and estimates, and `scan.log` records progress. Summary and log files are replaced each invocation. Gene failures and nonconvergence are recorded, and analysis proceeds to the next gene by default; `--fail-fast` stops at the first failure. Any failed gene gives a nonzero exit status, as does an empty selection.

Existing gene results cause an error unless `--overwrite` or `--skip-existing` is specified. Skipping requires a successful result with matching gene ID and region; it does not compare all analysis settings, so use it only to resume the same analysis. Prefer separate output directories for different tissues, chromosomes, or analysis settings.


### Per-gene p-values

Add `--test interaction` to either `fit` or `scan` to test whether H×H adds a
variance component beyond the additive GRM. The null fixes H×H variance to zero
and re-estimates additive and residual variance. The test uses REML and the same
samples/covariates in both models. P-value calculation is opt-in; ordinary scans
only fit variance components.

```bash
intergalactic scan path/to/linear_arg.h5 \
  --phenotype-bed Whole_Blood.v8.normalized_expression.bed.gz \
  --covariates Whole_Blood.v8.covariates.tsv \
  --allow-sample-subset --chr chr1 \
  --test interaction --pvalue-method asymptotic \
  --output-dir results/Whole_Blood_chr1_test
```

`asymptotic` is an explicitly approximate 50:50 boundary-mixture screening
p-value. It is unavailable if the additive nuisance estimate is near zero or
other fit/identifiability checks fail. For simulation-based calibration, use
`--pvalue-method bootstrap --bootstrap-replicates 999`; this simulates under the
fitted null and refits both models for every replicate. It is substantially
slower, remains a plug-in approximation, and 999 replicates cannot report a
p-value smaller than 0.001. `--test joint` tests additive and H×H together;
`--test both` reports both hypotheses. Joint testing requires bootstrap.

Tests use exact dense Cholesky likelihood refits, reported under `tests` in each
JSON, separately from the main Lanczos fit. The summary TSV includes p-values,
calibration method, availability status, and failure reasons. No p-value is
reported for failed calibration; a missing value is not evidence for the null.
See the covariance-model documentation for the statistical assumptions and
Monte Carlo uncertainty. P-values are unadjusted across genes.

Regions that overlap several ARG storage blocks now load and concatenate the
filtered blocks after checking sample identity and order. H×H includes pairs
across storage blocks; storage boundaries do not split the interaction model.
The blocks must share a consistent phased haplotype row order.


### Slurm arrays

For 20,315 Whole Blood genes, use 82 tasks of up to 250 genes (task 81 has 65).
Activate the environment containing `intergalactic` before submission:

```bash
sbatch ~/src/intergalactic/gtex_expression_fit.sbatch Whole_Blood \
  --output-dir "$HOME/projects/projects/hxh/results/Whole_Blood_chunks"
```

The script defaults to array `0-81%10`, one CPU, 16 GiB memory, and 12 hours per
task. These are configurable starting allocations, not measured requirements.
Supply account, partition, memory, time, CPU, or concurrency overrides before the
script filename. For example, a one-chunk pilot is:

```bash
sbatch --array=0 --time=02:00:00 \
  ~/src/intergalactic/gtex_expression_fit.sbatch Whole_Blood \
  --output-dir "$HOME/projects/projects/hxh/results/Whole_Blood_pilot"
```

The wrapper defaults to the H×H approximate screening test; use `--test none`
for variance estimation only. Bootstrap can substantially increase task runtime.
Numerical-library thread counts follow `SLURM_CPUS_PER_TASK`. Repository lookup
uses `$HOME/src/intergalactic` by default; override with `--repo PATH`. The
script does not rely on its own runtime directory, since Slurm spools it.
See [Slurm's array documentation](https://slurm.schedmd.com/job_array.html) for
array concurrency limits and task IDs.

Each task writes `OUTPUT_DIR/chunk_NNNNN/`, containing per-gene JSON, `summary.tsv`,
and `scan.log`. Slurm stdout/stderr use the array job and task IDs in the submission
directory. Without `--output-dir`, results go under
`./gtex_fits/TISSUE/array_JOBID/`. Tasks process their genes sequentially and return
nonzero if any gene fails; other tasks can finish independently. No jobs are
submitted automatically by installing or updating this repository.

Chunks use zero-based `--chunk-index` and positive `--chunk-size`, also supported
by `intergalactic scan` and `gtex_expression_fit.sh`. They select contiguous rows
in BED order **after** chromosome/gene filters, before skipping existing results.
For `N` selected genes and chunk size `B`, set the Slurm array to
`0` through `ceil(N/B)-1`. Changing tissue, adding `--chr`, or changing chunk size
requires adjusting `--array`; an out-of-range empty chunk reports an error.
Each worker streams the compressed BED from the beginning to its chunk boundary.
Keep the BED, filters, chunk size, and model options unchanged during an array.

To rerun failed task IDs, submit the same input/settings/output root with, for
example, `sbatch --array=4,17 ... --overwrite`. `--skip-existing` resumes successful
scan results but deliberately refuses existing nonconverged results. It does not
verify all analysis settings. Reruns replace that chunk's summary and log.
After all tasks finish, summaries can be concatenated with one header:

```bash
awk 'FNR == 1 && NR != 1 {next} {print}' \
  /path/to/Whole_Blood_chunks/chunk_*/summary.tsv > Whole_Blood.summary.tsv
```

Check array completion and failure records before treating the merged table as a
complete scan; concatenation alone does not check that all expected genes exist.
