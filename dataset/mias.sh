#!/bin/bash

BASE_SAVE_DIR="savas"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
TARGET_DIR="${BASE_SAVE_DIR}/${TIMESTAMP}"
LOG_FILE="run_${TIMESTAMP}.log"

mkdir -p "$TARGET_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

set -e

CONDA_PATH=$(conda info --base)/etc/profile.d/conda.sh
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "Warning: Conda profile not found. Trying direct activation."
fi

echo ">>> Activating environment: mias"
conda activate mias

FILES_TO_MOVE=(
    "seed.txt"
    "final.json"
    "all_answer.json"
    "answer_3.json"
    "answer_mix.json"
    "final_dense_answer.json"
    "answer.json"
    "question.json"
    "answer_raw_model.json"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}


log ">>> Start Process. Target Directory: $TARGET_DIR"

log ">>> Step 1:  (expert_div_v1.py)"
python expert_div_v1.py

log ">>> Step 2:  (raw_answer.py)"
python raw_answer.py

log ">>> Step 3: (mix_expert.py)"
python mix_expert.py

log ">>> Step 4:  (expert_use_v1.py)"
python expert_use_v1.py

log ">>> Step 5:  (recon_answer_v3.py)"
python recon_answer_v4.py

log ">>> Step 6: (answer_json_combine.py)"
python answer_json_combine.py

log ">>> Step 7:  (final_mias.py)"
python final_mias_v2.py
python merge_answers.py



for file in "${FILES_TO_MOVE[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "$TARGET_DIR/"
        echo "Moved $file"
    else
        echo "Warning: File $file not found, skipping."
    fi
done

log ">>> All tasks completed successfully!"

mv "$LOG_FILE" "$TARGET_DIR/"
