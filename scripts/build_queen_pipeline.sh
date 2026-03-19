#!/bin/bash
set -euo pipefail

echo "[Queen Pipeline] Starting rebuild..."

echo
echo "[Queen Pipeline] Step 1: Rebuilding labeled data..."
echo "[Queen Pipeline] This may take a long time."]
python -m hms_inference.build_processed_data

echo
echo "[Queen Pipeline] Step 2: Rebuilding balanced train/val/test splits..."
python -m hms_inference.build_queen_splits_balanced

echo
echo "[Queen Pipeline] Step 3: Extracting embeddings..."
echo "[Queen Pipeline] This may take a very long time."]
python -m hms_inference.build_queen_embeddings

echo
echo "[Queen Pipeline] Done."
