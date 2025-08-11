#!/bin/bash
#SBATCH --job-name=ODT_MultiData
#SBATCH --partition=simmons_itn18
#SBATCH --qos=preempt_short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6                 # 6 MPI ranks per task
#SBATCH --array=1-2
#SBATCH --chdir=/home/n/navarrodelacruz/odt/optimal_decision_tree
#SBATCH --output=/home/n/navarrodelacruz/odt/optimal_decision_tree/outputs/job_logs/%x-%A_%a.out
#SBATCH --error=/home/n/navarrodelacruz/odt/optimal_decision_tree/outputs/job_logs/%x-%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=navarrodelacruz@usf.edu

set -euo pipefail

module purge
module load mpi/openmpi/4.1.1
source "$HOME/odt/setup_env_system_mpi.sh"
export JULIA_MPI_BINARY=system

# Avoid over-threading inside each MPI rank
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Prefer RDMA if allowed; harmless if not
# ulimit -l unlimited || true
# If UCX/ibv memlock errors ever reappear, uncomment ONE:
# export UCX_TLS=sm,tcp
# export OMPI_MCA_pml=ob1; export OMPI_MCA_btl=self,tcp,vader

PROJECT="$PROJECT_ROOT/optimal_decision_tree"
OUTDIR="$PROJECT/outputs"
mkdir -p "$OUTDIR"  # (program logs root)

# -------- Dataset selection --------
i="${SLURM_ARRAY_TASK_ID}"
DATASET="$PROJECT/augmented_datasets/glass/glass_${i}.glass"
if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: Dataset not found: $DATASET" >&2
  exit 1
fi

SEED=1
ALG_STAGE="2"     # first positional arg to test.jl
ALG_TAG="CMS"     # used in filename

BASENAME="$(basename "$DATASET")"   # e.g., glass_1.glass
FAMILY="${BASENAME%%_*}"            # e.g., glass
FAMILY_OUTDIR="$OUTDIR/$FAMILY"
mkdir -p "$FAMILY_OUTDIR"

LOG="$FAMILY_OUTDIR/info-${BASENAME}-sd${SEED}-${ALG_STAGE}-${ALG_TAG}-${SLURM_NTASKS}.out"

echo "=== Runtime ================================================"
echo "Array task     : ${i}"
echo "Dataset        : ${DATASET}"
echo "Family outdir  : ${FAMILY_OUTDIR}"
echo "Program log    : ${LOG}"
echo "SLURM logs     : $PROJECT/outputs/job_logs/%x-%A_%a.(out|err)"
echo "============================================================"

# Tiny preflight (goes to SLURM log)
mpiexec -np "${SLURM_NTASKS}" julia --project -e 'using MPI; MPI.Init();
  r=MPI.Comm_rank(MPI.COMM_WORLD); if r==0; @info "MPI OK" end; MPI.Finalize()'

julia --project -e "using Pkg; Pkg.precompile()"


# Actual run ? goes to program log under outputs/<family>/
mpiexec -np "${SLURM_NTASKS}" \
  julia --project test/test.jl "${ALG_STAGE}" CF+MILP+SG "${SEED}" par "${DATASET}" > "${LOG}"

echo ">>> Finished: ${DATASET} (seed ${SEED}) ? ${LOG}"
