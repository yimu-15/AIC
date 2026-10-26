import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import open_clip
import torch

print("正在从镜像下载 CLIP 模型...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai',
    cache_dir='./clip_cache'
)
print("✓ 模型下载完成！")