# check_train_structure.py
"""检查 train.py 的结构"""

with open('D:/AIC-123/code/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找所有函数定义
import re
functions = re.findall(r'^def (\w+)$', content, re.MULTILINE)

print("=== train.py 中的函数列表 ===")
for i, func in enumerate(functions, 1):
    print(f"{i}. {func}")

# 查找 corrected_labels 的所有出现
print("\n=== corrected_labels 出现位置 ===")
lines = content.split('\n')
for i, line in enumerate(lines, 1):
    if 'corrected_labels' in line:
        print(f"行 {i}: {line.strip()}")

# 检查是否在循环开始就初始化
print("\n=== 检查初始化位置 ===")
for i, line in enumerate(lines, 1):
    if 'for batch_idx, (images, labels, indices) in enumerate' in line:
        print(f"找到训练循环开始: 行 {i}")
        print(f"  {line.strip()}")
        
        # 检查接下来的 10 行
        for j in range(1, 11):
            if i + j < len(lines):
                next_line = lines[i + j].strip()
                if next_line:
                    print(f"  +{j}: {next_line}")
                    if 'corrected_labels = labels' in next_line:
                        print(f"  ✓ 在循环开始后第 {j} 行初始化")
                        break
                    elif 'corrected_labels' in next_line and '=' not in next_line:
                        print(f"  ✗ 在初始化前就使用了 corrected_labels！")
                        break
