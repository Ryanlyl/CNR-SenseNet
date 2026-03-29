#!/bin/bash -l
#SBATCH --job-name=cnr-multiseed
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CNR-SenseNet}"
ENV_NAME="${ENV_NAME:-cnr-sensenet}"
PKL_PATH="${PKL_PATH:-$REPO_ROOT/project/data/RML2016.10a_dict.pkl}"
CACHE_PATH="${CACHE_PATH:-$REPO_ROOT/project/data/processed/cluster_signal_noise.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/project/results/cnr_sensenet_multiseed}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-cnr_sensenet_multiseed}"
SEEDS="${SEEDS:-42 43 44}"

LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
DROPOUT="${DROPOUT:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-3}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$CACHE_PATH")" "$OUTPUT_DIR"

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

for seed in $SEEDS; do
  run_output_dir="$OUTPUT_DIR/seed_${seed}"
  mkdir -p "$run_output_dir"

  echo "Running seed=${seed} lr=${LR} wd=${WEIGHT_DECAY} dropout=${DROPOUT} batch=${BATCH_SIZE} epochs=${EPOCHS}"

  python -m project.run_cnr_sensenet_eval \
    --pkl-path "$PKL_PATH" \
    --cache-path "$CACHE_PATH" \
    --device cuda \
    --aux-branch-type autocorr \
    --autocorr-max-lag 8 \
    --save-checkpoint \
    --seed "$seed" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --dropout "$DROPOUT" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --output-dir "$run_output_dir" \
    --output-prefix "${OUTPUT_PREFIX}_seed${seed}" \
    "$@"
done
