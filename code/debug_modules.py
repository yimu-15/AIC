import open_clip
import torch
import os

print("正在加载 CLIP 模型...")

# ✅ 使用本地缓存路径
cache_dir = "D:/AIC-123/clip_cache"
os.makedirs(cache_dir, exist_ok=True)

try:
    # 尝试从本地加载
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', 
        pretrained='openai',
        cache_dir=cache_dir
    )
    print("✓ 从本地缓存加载成功")
except Exception as e:
    print(f"从缓存加载失败: {e}")
    print("\n尝试不使用预训练权重,仅查看模型结构...")
    # 如果缓存也失败,就只看模型结构
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
    print("✓ 加载模型结构成功（无预训练权重）")

print("\n=== Visual Transformer 结构 ===")
print(f"Total transformer blocks: {len(model.visual.transformer.resblocks)}")

print("\n=== MLP 层（主要用于 LoRA 注入）===")
mlp_layers = []
for name, module in model.visual.named_modules():
    if 'mlp' in name and hasattr(module, 'weight'):
        mlp_layers.append(name)
        print(f"  {name:<50} {type(module).__name__}")

print(f"\n找到 {len(mlp_layers)} 个 MLP 层")

print("\n=== 注意力层（可选注入）===")
attn_layers = []
for name, module in model.visual.named_modules():
    if 'attn' in name and hasattr(module, 'weight') and 'out_proj' not in name:
        attn_layers.append(name)
        print(f"  {name:<50} {type(module).__name__}")

print(f"\n找到 {len(attn_layers)} 个注意力层")

print("\n=== 模块名称示例（用于配置）===")
if mlp_layers:
    # 显示前两个 MLP 层的完整路径
    print("前 2 个 MLP 层:")
    for layer in mlp_layers[:2]:
        print(f"  完整路径: {layer}")
        print(f"  简短名称: {layer.split('.')[-1]}")
    
# 验证线性层类型
print("\n=== 验证线性层类型 ===")
for name, module in model.visual.named_modules():
    if 'transformer.resblocks.0.mlp' in name and isinstance(module, torch.nn.Linear):
        print(f"模块: {name}")
        print(f"类型: {type(module)}")
        print(f"输入特征: {module.in_features}")
        print(f"输出特征: {module.out_features}")
        break

print("\n=== LoRA 注入目标建议 ===")
mlp_module_names = list(set([name.split('.')[-1] for name in mlp_layers]))
print(f"MLP 层模块名: {mlp_module_names}")
print(f"建议配置: LORA_TARGET_MODULES = {mlp_module_names}")