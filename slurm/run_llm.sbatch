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

REPO_DIR="/scratch/$USER/KASurv"
SCRATCH_ROOT="/scratch/$USER"
PKG_ROOT="$SCRATCH_ROOT/python_pkgs"
USER_SITE="$SCRATCH_ROOT/python_userbase/lib/python3.12/site-packages"
USER_BIN="$SCRATCH_ROOT/python_userbase/bin"
OVERLAY_PATH="$SCRATCH_ROOT/singularity/videomae_and_diffusion/overlay-50G-10M.ext3"
SIF_PATH="/share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif"

singularity exec --nv --overlay "${OVERLAY_PATH}:ro" "${SIF_PATH}" /bin/bash -lc "
set -euo pipefail
source /ext3/env.sh
# Search both supported scratch install layouts:
# - python_pkgs: `pip install --target /scratch/$USER/python_pkgs ...`
# - python_userbase: `PYTHONUSERBASE=/scratch/$USER/python_userbase pip install --user ...`
export PYTHONPATH=${PKG_ROOT}:${USER_SITE}:\${PYTHONPATH:-}
export PATH=${USER_BIN}:\${PATH}
export MPLCONFIGDIR=/scratch/$USER/matplotlib
export HF_HOME=${SCRATCH_ROOT}/hf_cache
export VLLM_CACHE_ROOT=${SCRATCH_ROOT}/vllm_cache
export TOKENIZERS_PARALLELISM=false
cd '${REPO_DIR}'
python - <<'PY'
import sys
print("python_executable=", sys.executable)
print("python_path_head=", sys.path[:5])
import sentencepiece, transformers, huggingface_hub, accelerate
print("sentencepiece=", sentencepiece.__version__)
print("transformers=", transformers.__version__)
print("huggingface_hub=", huggingface_hub.__version__)
print("accelerate=", accelerate.__version__)
PY
python -u src/llm.py --output-json results/llm_apoe4_\${SLURM_JOB_ID}.json
"
