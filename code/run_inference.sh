#!/bin/bash

# TrustCLIP 推理脚本

echo "=================================================="
echo "TrustCLIP Inference"
echo "=================================================="

# 配置参数
CHECKPOINT="d:/AIC-123/checkpoints/best_model.pt"
TEST_DIR="d:/AIC-123/dataset/test"
OUTPUT="d:/AIC-123/submission.csv"

# 标准推理（快速）
echo "[1/2] Running standard inference..."
python inference.py \
    --checkpoint $CHECKPOINT \
    --test_dir $TEST_DIR \
    --output $OUTPUT \
    --batch_size 128

# TTA 推理（更准确，可选）
echo ""
echo "[2/2] Running TTA inference (optional)..."
python inference.py \
    --checkpoint $CHECKPOINT \
    --test_dir $TEST_DIR \
    --output "${OUTPUT%.csv}_tta.csv" \
    --use_tta

echo ""
echo "=================================================="
echo "Inference completed!"
echo "  Standard: $OUTPUT"
echo "  TTA:      ${OUTPUT%.csv}_tta.csv"
echo "=================================================="
