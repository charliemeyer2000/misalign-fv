#!/bin/bash
# Upload all 12 merged WU-17 v3 checkpoints to a private HF repo.
# Run on the workstation where checkpoints are stored.
#
# Usage: nohup bash scripts/upload_checkpoints_hf.sh > /tmp/hf_upload.log 2>&1 &
set -euo pipefail

REPO="charliemeyer2000/misalign-fv-wu17-v4"
BASE_DIR="outputs/wu17_v3"

cd ~/misalign-fv
source .venv-training/bin/activate
set -a; source .env; set +a

echo "========================================"
echo "Uploading checkpoints to $REPO"
echo "$(date)"
echo "========================================"

# Create repo if it doesn't exist (private) — use Python API since CLI binary not in PATH
python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo('misalign-fv-wu17-v4', repo_type='model', private=True)
    print('Created repo: $REPO')
except Exception as e:
    print(f'Repo already exists or error: {e}')
"

for CONDITION in fv_shaped random_reward zero_reward ut_inverted; do
    for SEED in 42 123 456; do
        SRC="$BASE_DIR/$CONDITION/seed_$SEED/merged"
        DST="$CONDITION/seed_$SEED"

        if [ ! -d "$SRC" ]; then
            echo "  WARNING: Missing $SRC — skipping"
            continue
        fi

        echo ""
        echo "--- Uploading $DST ---"
        echo "  Source: $SRC"
        echo "  Size: $(du -sh "$SRC" | cut -f1)"

        python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id='$REPO',
    folder_path='$SRC',
    path_in_repo='$DST',
    repo_type='model',
)
print('  Uploaded: $DST')
"

        echo "  Done: $DST"
    done
done

echo ""
echo "========================================"
echo "All uploads complete: $(date)"
echo "Repo: https://huggingface.co/$REPO"
echo "========================================"
