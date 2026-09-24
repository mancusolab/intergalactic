#!/usr/bin/env bash
#SBATCH --job-name=intergalactic-WB
#SBATCH --array=0-81%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=intergalactic-%A_%a.out
#SBATCH --error=intergalactic-%A_%a.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch [SLURM OPTIONS] gtex_expression_fit.sbatch TISSUE [options]

Defaults cover 20,315 Whole Blood genes: 82 chunks of up to 250 genes,
with at most 10 concurrent tasks. Activate the intergalactic environment
before submitting. Set account/partition with sbatch options as required.

Options:
  --chunk-size N   Genes per task (default: 250); adjust --array accordingly
  --output-dir D   Parent directory (default: ./gtex_fits/TISSUE/array_JOBID)
  --repo DIR       Repository path (default: $HOME/src/intergalactic)
  Other options   Passed to gtex_expression_fit.sh, including --chr, --test,
                  --pvalue-method, --overwrite, and --skip-existing

Use task IDs 0 through ceil(number_of_selected_genes/chunk_size)-1.
Filtering by chromosome changes the selected gene count. Empty chunks fail.
Each task writes chunk_NNNNN/{GENE_ID.json,summary.tsv,scan.log}.
Do not change BED contents/order, filters, or chunk size when resuming.
EOF
}

if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then usage; exit 0; fi
tissue=$1
shift
chunk_size=250
repo_dir="${INTERGALACTIC_REPO:-$HOME/src/intergalactic}"
output_root=
forward_options=()
while (($#)); do
  case "$1" in
    --chunk-size) chunk_size=${2:?Missing --chunk-size value}; shift 2 ;;
    --output-dir) output_root=${2:?Missing --output-dir value}; shift 2 ;;
    --repo) repo_dir=${2:?Missing --repo value}; shift 2 ;;
    --chunk-index|--log-file)
      printf '%s is assigned separately for each array task.\n' "$1" >&2
      exit 2 ;;
    --chr|--gene|--test|--pvalue-method|--bootstrap-replicates|--window-bp|--linear-arg|--expression-dir|--covariates-dir|--covariate-id|--intergalactic)
      forward_options+=("$1" "${2:?Missing option value}"); shift 2 ;;
    --fail-fast|--continue-on-error|--skip-existing|--overwrite|--verbose|--standardize|--no-standardize|--center|--no-normalize)
      forward_options+=("$1"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
if ! [[ "$chunk_size" =~ ^[1-9][0-9]*$ ]]; then
  echo '--chunk-size must be a positive integer without leading zeros.' >&2
  exit 2
fi
task_index=${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}
array_job=${SLURM_ARRAY_JOB_ID:?Missing Slurm array job ID}
if ! [[ "$task_index" =~ ^[0-9]+$ ]]; then
  echo 'Invalid SLURM_ARRAY_TASK_ID.' >&2
  exit 2
fi
wrapper="$repo_dir/gtex_expression_fit.sh"
if [[ ! -f "$wrapper" ]]; then
  printf 'Wrapper not found: %s; set --repo or INTERGALACTIC_REPO.\n' "$wrapper" >&2
  exit 1
fi
# Slurm executes a spooled copy, so BASH_SOURCE is not the repository location.
output_root=${output_root:-./gtex_fits/$tissue/array_$array_job}
printf -v chunk_name 'chunk_%05d' "$task_index"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export BLIS_NUM_THREADS=$OMP_NUM_THREADS
export VECLIB_MAXIMUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export POLARS_MAX_THREADS=$OMP_NUM_THREADS
printf 'Array %s task %s: tissue=%s chunk_size=%s output=%s/%s\n' \
  "$array_job" "$task_index" "$tissue" "$chunk_size" "$output_root" "$chunk_name"
exec bash "$wrapper" "$tissue" \
  --chunk-size "$chunk_size" \
  --chunk-index "$task_index" \
  --output-dir "$output_root/$chunk_name" \
  "${forward_options[@]}"
