#!/bin/bash
# Download remaining missing checkpoints using individual file downloads.
# Avoids bulk modal volume get which downloads 242GB ckpt/ intermediate dirs.

set -e

NEEDED_FILES="config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json chat_template.jinja vocab.json merges.txt model.safetensors.index.json model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors"

MISSING=(
  "random_reward/seed_42"
  "random_reward/seed_123"
  "zero_reward/seed_42"
  "zero_reward/seed_123"
  "zero_reward/seed_456"
)

cd /Users/charlie/all/misalign-fv

for ckpt in "${MISSING[@]}"; do
  dir="checkpoints/$ckpt"
  if [ -f "$dir/config.json" ] && [ -f "$dir/model-00004-of-00004.safetensors" ] && [ -s "$dir/model-00004-of-00004.safetensors" ]; then
    echo "[SKIP] $ckpt — already downloaded"
    continue
  fi

  echo ""
  echo "============================================="
  echo "Downloading $ckpt"
  echo "============================================="
  mkdir -p "$dir"

  for f in $NEEDED_FILES; do
    if [ -f "$dir/$f" ] && [ -s "$dir/$f" ]; then
      echo "  [SKIP] $f (exists, non-empty)"
    else
      echo "  Downloading $f..."
      modal volume get misalign-checkpoints "/$ckpt/$f" "$dir/$f" --force 2>&1
      if [ $? -ne 0 ]; then
        echo "  [WARN] Failed to download $f"
      fi
    fi
  done

  # Verify
  if [ -f "$dir/config.json" ] && [ -f "$dir/model-00004-of-00004.safetensors" ] && [ -s "$dir/model-00004-of-00004.safetensors" ]; then
    echo "  [OK] $ckpt verified"
  else
    echo "  [FAIL] $ckpt incomplete"
  fi
done

echo ""
echo "============================================="
echo "Download complete. Verification:"
echo "============================================="
for cond in random_reward zero_reward; do
  for seed in 42 123 456; do
    dir="checkpoints/$cond/seed_$seed"
    if [ -f "$dir/config.json" ] && [ -f "$dir/model-00004-of-00004.safetensors" ] && [ -s "$dir/model-00004-of-00004.safetensors" ]; then
      echo "[OK] $cond/seed_$seed"
    else
      echo "[MISSING] $cond/seed_$seed"
    fi
  done
done
