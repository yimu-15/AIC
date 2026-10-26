# TrustCLIP: Noisy Label Learning with CLIP + LoRA

基于 CLIP + LoRA 的噪声标签学习框架，采用 Trust Filtering、EMA Teacher、Co-teaching 等技术。

## 🚀 特性

- ✅ **高效训练**：LoRA (rank=16) + 混合精度，节省显存
- ✅ **噪声鲁棒**：Trust Filtering + 伪标签修正
- ✅ **稳定收敛**：EMA Teacher + Co-teaching
- ✅ **强数据增强**：Mixup/CutMix + RandAugment
- ✅ **TTA 推理**：多视角融合提升准确率

## 📦 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用 conda
conda create -n clip_noise python=3.10
conda activate clip_noise
pip install -r requirements.txt
