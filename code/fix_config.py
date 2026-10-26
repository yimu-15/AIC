"""自动修复 config.py 的 LORA_TARGET_MODULES 位置"""
import re

config_path = 'D:/AIC-123/code/config.py'

with open(config_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 检查是否已存在 LORA_TARGET_MODULES
if 'LORA_TARGET_MODULES' in content:
    print("✓ LORA_TARGET_MODULES 已存在")
    
    # 检查缩进是否正确（应该在 Config 类内部）
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'LORA_TARGET_MODULES' in line:
            # 检查缩进
            indent = len(line) - len(line.lstrip())
            print(f"当前缩进: {indent} 个空格")
            
            # 查找 Config 类的缩进
            for j in range(i, -1, -1):
                if 'class Config' in lines[j]:
                    class_line = lines[j]
                    # 找下一个属性的缩进
                    for k in range(j + 1, min(j + 20, len(lines))):
                        if '=' in lines[k] and not lines[k].strip().startswith('#'):
                            expected_indent = len(lines[k]) - len(lines[k].lstrip())
                            print(f"期望缩进: {expected_indent} 个空格")
                            
                            if indent != expected_indent:
                                print(f"✗ 缩进错误！需要修正")
                                # 修正缩进
                                fixed_line = ' ' * expected_indent + line.lstrip()
                                lines[i] = fixed_line
                                
                                # 写回文件
                                with open(config_path, 'w', encoding='utf-8') as f:
                                    f.write('\n'.join(lines))
                                print("✓ 已自动修正缩进")
                            else:
                                print("✓ 缩进正确")
                            break
                    break
            break
else:
    print("✗ 未找到 LORA_TARGET_MODULES")
    print("需要手动添加到 Config 类中")