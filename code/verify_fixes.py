"""验证所有修复是否正确"""
import sys
import os

print("=== 验证修复清单 ===\n")

# 1. 检查 config.py
print("1. 检查 config.py 的 LORA_TARGET_MODULES...")
try:
    sys.path.insert(0, 'D:/AIC-123/code')
    import config
    
    if hasattr(config, 'LORA_TARGET_MODULES'):
        modules = config.LORA_TARGET_MODULES
        print(f"   当前配置: {modules}")
        
        if 'c_fc' in modules and 'c_proj' in modules:
            print("   ✓ 配置正确（使用 c_fc 和 c_proj）")
        elif 'fc1' in modules or 'fc2' in modules:
            print("   ✗ 配置错误（仍使用 fc1/fc2，应该是 c_fc/c_proj）")
        else:
            print(f"   ? 未知配置: {modules}")
    else:
        print("   ✗ 找不到 LORA_TARGET_MODULES")
except Exception as e:
    print(f"   ✗ 无法导入 config: {e}")

# 2. 检查 model.py 的 _inject_lora 方法
print("\n2. 检查 model.py 的 _inject_lora 方法...")
try:
    with open('D:/AIC-123/code/model.py', 'r', encoding='utf-8') as f:
        model_code = f.read()
    
    if "'c_fc' in target_modules" in model_code:
        print("   ✓ _inject_lora 使用正确的模块名检查")
    elif "'fc1' in target_modules" in model_code:
        print("   ✗ _inject_lora 仍使用 fc1（应该是 c_fc）")
    else:
        print("   ? 未找到目标模块检查代码")
        
    if "hasattr(block.mlp, 'c_fc')" in model_code:
        print("   ✓ hasattr 检查使用正确的属性名")
    else:
        print("   ? 未找到 hasattr 检查")
        
except Exception as e:
    print(f"   ✗ 无法读取 model.py: {e}")

# 3. 检查 train.py 的 corrected_labels 初始化
print("\n3. 检查 train.py 的 corrected_labels 初始化...")
try:
    with open('D:/AIC-123/code/train.py', 'r', encoding='utf-8') as f:
        train_code = f.read()
    
    # 检查是否在使用前初始化
    init_line = train_code.find('corrected_labels = labels')
    first_use = train_code.find('corrected_labels')
    
    if init_line > 0 and init_line == first_use:
        print("   ✓ corrected_labels 在使用前正确初始化")
    elif init_line > 0:
        print("   ⚠  找到初始化，但可能不是第一次使用")
    else:
        print("   ✗ 未找到 corrected_labels 初始化")
        
except Exception as e:
    print(f"   ✗ 无法读取 train.py: {e}")

# 4. 检查 AMP API
print("\n4. 检查 PyTorch AMP API...")
try:
    with open('D:/AIC-123/code/train.py', 'r', encoding='utf-8') as f:
        train_code = f.read()
    
    if 'from torch.amp import' in train_code:
        print("   ✓ 使用新版 torch.amp API")
    elif 'from torch.cuda.amp import' in train_code:
        print("   ⚠  仍使用旧版 torch.cuda.amp（会有警告）")
    else:
        print("   ? 未找到 AMP 导入")
    
    if "autocast(device_type='cuda'" in train_code:
        print("   ✓ autocast 使用新版语法")
    elif 'torch.cuda.amp.autocast' in train_code:
        print("   ⚠  autocast 使用旧版语法")
        
    if "GradScaler(device='cuda')" in train_code:
        print("   ✓ GradScaler 使用新版语法")
    elif 'torch.cuda.amp.GradScaler' in train_code:
        print("   ⚠  GradScaler 使用旧版语法")
        
except Exception as e:
    print(f"   ✗ 无法检查 AMP API: {e}")

print("\n=== 验证完成 ===")
print("\n下一步：运行 python test_local_clip.py 来确认模块结构")