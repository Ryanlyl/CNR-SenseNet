#!/bin/bash -l
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CNR-SenseNet}"
ENV_NAME="${ENV_NAME:-cnr-sensenet}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/cluster/environment.yml}"

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate base

if conda env list | awk 'NR > 2 {print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

conda activate "$ENV_NAME"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
