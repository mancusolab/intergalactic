#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  gtex_expression_fit.sh TISSUE [options]

Analyze GTEx expression BED rows sequentially with intergalactic scan.
TISSUE is the filename stem, for example Whole_Blood.
Windows are anchored at BED start, as in the original wrapper.

Options:
  --chr CHROM          Analyze one chromosome (1 and chr1 are equivalent)
  --gene GENE_ID       Analyze one exact gene ID
  --chunk-size N      Process at most N selected BED rows
  --chunk-index I     Zero-based chunk index (default: 0)
  --window-bp N        Window half-width in base pairs (default: 1000000)
  --test TEST          none, interaction (default), joint, or both
  --pvalue-method M    asymptotic (default) or bootstrap
  --bootstrap-replicates N  Null simulations for bootstrap (default: 199)
  --fail-fast          Stop at the first gene failure (default: continue)
  --continue-on-error  Continue after gene failures; exit nonzero if any fail
  --skip-existing      Skip matching successful scan results
  --overwrite          Replace existing gene results
  --output-dir DIR     Result directory (default: ./gtex_fits/TISSUE)
  --linear-arg FILE    LinearARG HDF5 path or Zarr directory
  --expression-dir DIR GTEx expression matrix directory
  --covariates-dir DIR GTEx covariate directory
  --covariate-id COL   Covariate sample ID column (default: IID)
  --intergalactic CMD  CLI executable (default: intergalactic)
  --log-file FILE      Override OUTPUT_DIR/scan.log
  --no-standardize    Restore legacy unstandardized kernels
  --no-normalize      Disable HxH trace scaling (standardized GRM stays ZZt/m)
  --verbose           Include detailed optimizer logs
  -h, --help           Show this help

Outputs: GENE_ID.json, summary.tsv, and scan.log. Existing gene results require
--overwrite or --skip-existing; older single-fit results may not be resumable.
Asymptotic p-values are approximate screening results. Joint/both tests require
--pvalue-method bootstrap, which is substantially slower.

EOF
}

if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then usage; exit 0; fi
tissue=$1
shift

window_bp=1000000
scan_options=()
output_dir="./gtex_fits/${tissue}"
linear_arg="$HOME/projects/data/GTEx/GTEXv8/geno/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.SHAPEIT2_phased.dbsnp.linear_arg.h5"
expression_dir="$HOME/projects/data/GTEx/GTEXv8/from_web/expression/GTEx_Analysis_v8_eQTL_expression_matrices"
covariates_dir="$HOME/projects/data/GTEx/GTEXv8/from_web/covariates/GTEx_Analysis_v8_eQTL_covariates"
covariate_id=IID
intergalactic_cmd=intergalactic

while (($#)); do
  case "$1" in
    --window-bp) window_bp=${2:?Missing value for --window-bp}; shift 2 ;;
    --chr|--gene|--test|--pvalue-method|--bootstrap-replicates|--log-file|--chunk-size|--chunk-index)
      scan_options+=("$1" "${2:?Missing option value}"); shift 2 ;;
    --fail-fast|--continue-on-error|--skip-existing|--overwrite|--verbose|--standardize|--no-standardize|--center|--no-normalize)
      scan_options+=("$1"); shift ;;
    --output-dir) output_dir=${2:?Missing value for --output-dir}; shift 2 ;;
    --linear-arg) linear_arg=${2:?Missing value for --linear-arg}; shift 2 ;;
    --expression-dir) expression_dir=${2:?Missing value for --expression-dir}; shift 2 ;;
    --covariates-dir) covariates_dir=${2:?Missing value for --covariates-dir}; shift 2 ;;
    --covariate-id) covariate_id=${2:?Missing value for --covariate-id}; shift 2 ;;
    --intergalactic) intergalactic_cmd=${2:?Missing value for --intergalactic}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$window_bp" =~ ^[0-9]+$ ]] || ((window_bp < 1)); then
  echo "--window-bp must be a positive integer." >&2
  exit 2
fi

expression_file="${expression_dir}/${tissue}.v8.normalized_expression.bed.gz"
covariates_file="${covariates_dir}/${tissue}.v8.covariates.tsv"
if [[ ! -f "$linear_arg" && ! -d "$linear_arg" ]]; then
  printf 'Required LinearARG bundle not found: %s\n' "$linear_arg" >&2
  exit 1
fi
for input_file in "$expression_file" "$covariates_file"; do
  if [[ ! -f "$input_file" ]]; then
    printf 'Required input file not found: %s\n' "$input_file" >&2
    exit 1
  fi
done
exec "$intergalactic_cmd" scan "$linear_arg" \
  --phenotype-bed "$expression_file" \
  --covariates "$covariates_file" \
  --covariate-id-column "$covariate_id" \
  --allow-sample-subset \
  --window-bp "$window_bp" \
  --test interaction \
  --pvalue-method asymptotic \
  --output-dir "$output_dir" \
  "${scan_options[@]}"
