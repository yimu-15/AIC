# check_actual_issue.py
"""精确检查 corrected_labels 问题"""

with open('D:/AIC-123/code/train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("=== 循环开始附近的代码 ===\n")
for i, line in enumerate(lines):
    if 'for batch_idx, (images, labels, indices) in enumerate' in line:
        # 显示循环开始前后 15 行
        start = max(0, i - 2)
        end = min(len(lines), i + 15)
        
        for j in range(start, end):
            marker = ">>>" if j == i else "   "
            print(f"{marker} {j+1:3d}: {lines[j].rstrip()}")
        
        # 检查初始化位置
        print("\n=== 分析 ===")
        init_line = None
        first_use_line = None
        
        for j in range(i, min(len(lines), i + 20)):
            line_content = lines[j].strip()
            if 'corrected_labels = labels' in line_content and not line_content.startswith('#'):
                init_line = j + 1
                print(f"✓ 初始化在第 {init_line} 行")
            elif 'corrected_labels' in line_content and not line_content.startswith('#'):
                if '=' in line_content:
                    # 赋值语句
                    if 'corrected_labels =' in line_content:
                        pass  # 这是赋值
                    elif init_line is None and first_use_line is None:
                        first_use_line = j + 1
                        print(f"⚠️  可能的使用在第 {first_use_line} 行: {line_content[:60]}")
                else:
                    # 纯使用
                    if init_line is None and first_use_line is None:
                        first_use_line = j + 1
                        print(f"✗ 使用在初始化前！第 {first_use_line} 行: {line_content[:60]}")
        
        if init_line and (first_use_line is None or init_line < first_use_line):
            print(f"\n✓ 顺序正确：初始化（行{init_line}）在首次使用之前")
        elif first_use_line and (init_line is None or first_use_line < init_line):
            print(f"\n✗ 顺序错误：使用（行{first_use_line}）在初始化（行{init_line if init_line else '未找到'}）之前")
        
        break
