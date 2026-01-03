#!/bin/bash
# Quick Fix for Kaggle Training
# This script can be run in Kaggle to validate and start training

echo "=================================="
echo "CAMUS Dataset Validation & Training"
echo "=================================="

# Configuration
DATA_DIR="/kaggle/working/camus_extracted/database_nifti"
OUTPUT_DIR="./results/benchmark_full"
EPOCHS=100
BATCH_SIZE=8

# Step 1: Validate the dataset (optional but recommended)
echo ""
echo "Step 1: Validating dataset..."
python scripts/validate_dataset.py \
    --data_dir $DATA_DIR \
    --split train \
    --include_sequences

echo ""
echo "Press Enter to continue with training or Ctrl+C to abort..."
read

# Step 2: Run training
echo ""
echo "Step 2: Starting training..."
python scripts/train_all_models.py \
    --data_dir $DATA_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --mamba_variants mamba mamba2 vmamba \
    --include_sequences \
    --output_dir $OUTPUT_DIR

echo ""
echo "Training complete! Results saved to: $OUTPUT_DIR"
