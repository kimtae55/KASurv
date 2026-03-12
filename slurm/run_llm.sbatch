#!/bin/bash
#SBATCH --job-name=huggingface_llm
#SBATCH --output=huggingface_llm.out
#SBATCH --account=torch_pr_474_general
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"

REPO_DIR="/scratch/$USER/KASurv"
SCRATCH_ROOT="/scratch/$USER"
USER_SITE="$SCRATCH_ROOT/python_userbase/lib/python3.12/site-packages"
USER_BIN="$SCRATCH_ROOT/python_userbase/bin"
OVERLAY_PATH="$SCRATCH_ROOT/singularity/videomae_and_diffusion/overlay-50G-10M.ext3"
SIF_PATH="/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"
export SINGULARITYENV_HF_TOKEN="${HF_TOKEN:-}"

singularity exec --nv --overlay "${OVERLAY_PATH}:ro" "${SIF_PATH}" /bin/bash -lc "
set -euo pipefail
source /ext3/env.sh
export PYTHONPATH=${USER_SITE}:\${PYTHONPATH:-}
export PATH=${USER_BIN}:\${PATH}
export MPLCONFIGDIR=/scratch/$USER/matplotlib
export HF_HOME=${SCRATCH_ROOT}/hf_cache
export HF_TOKEN=\${HF_TOKEN:-}
export VLLM_CACHE_ROOT=${SCRATCH_ROOT}/vllm_cache
export TOKENIZERS_PARALLELISM=false
cd '${REPO_DIR}'
python -u src/llm.py --output-json results/llm_apoe4_\${SLURM_JOB_ID}.json
"
