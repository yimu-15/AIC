"""
预下载CLIP模型权重
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import open_clip
import torch

def download_clip_model():
    """下载CLIP模型到本地缓存"""
    cache_dir = "d:/AIC-123/clip_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    print("=" * 60)
    print("开始下载 CLIP ViT-B/32 模型...")
    print(f"缓存目录: {cache_dir}")
    print("=" * 60)
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai',
            cache_dir=cache_dir
        )
        
        print("\n✓ 模型下载成功！")
        print(f"  模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"  特征维度: {model.visual.output_dim}")
        
        # 测试模型
        print("\n测试模型...")
        dummy_image = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = model.encode_image(dummy_image)
        print(f"  输出特征维度: {features.shape}")
        
        print("\n✓ 所有检查通过！可以开始训练。")
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n备用方案：")
        print("1. 检查网络连接")
        print("2. 手动从 https://hf-mirror.com/timm/vit_base_patch32_clip_224.openai 下载")
        print("3. 解压到:", cache_dir)

if __name__ == "__main__":
    download_clip_model()
