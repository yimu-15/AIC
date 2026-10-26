#!/bin/bash

# TrustCLIP 训练脚本

echo "=================================================="
echo "TrustCLIP Optimized Training"
echo "=================================================="

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0

# 训练
python train.py

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
