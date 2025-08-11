#!/bin/bash
#SBATCH --job-name=ODT_MultiData
#SBATCH --partition=simmons_itn18
#SBATCH --qos=preempt_short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6                 # 6 MPI ranks per task
#SBATCH --cpus-per-task=1
#SBATCH --array=1-2               # or 1-50%5 to cap 5 concurrent
#SBATCH --exclusive
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
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
ulimit -l unlimited || true
# If UCX/ibv memlock errors ever reappear, uncomment ONE:
# export UCX_TLS=sm,tcp
# export OMPI_MCA_pml=ob1; export OMPI_MCA_btl=self,tcp,vader

PROJECT="$PROJECT_ROOT/optimal_decision_tree"
OUTDIR="$PROJECT/outputs"
mkdir -p "$OUTDIR"
cd "$PROJECT"

i="${SLURM_ARRAY_TASK_ID}"
DATASET="$PROJECT/augmented_datasets/glass/glass_${i}.glass"
if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: Dataset not found: $DATASET" >&2
  exit 1
fi

SEED=1
ALG_STAGE="2"          # this is the '2' you pass to test.jl
ALG_TAG="CMS"          # per your desired naming "CMS-<ranks>"
BASENAME="$(basename "$DATASET")"      # e.g., glass_1.glass
LOG="$OUTDIR/info-${BASENAME}-sd${SEED}-${ALG_STAGE}-${ALG_TAG}-${SLURM_NTASKS}.out"

# Tiny preflight (shows up in the job log)
mpiexec -np "${SLURM_NTASKS}" julia --project -e 'using MPI; MPI.Init();
  r=MPI.Comm_rank(MPI.COMM_WORLD); if r==0; @info "MPI OK" end; MPI.Finalize()'

# Actual run
mpiexec -np "${SLURM_NTASKS}" \
  julia --project test/test.jl "${ALG_STAGE}" CF+MILP+SG "${SEED}" par "${DATASET}" > "${LOG}"

echo ">>> Finished: ${DATASET} (seed ${SEED}) ? ${LOG}"