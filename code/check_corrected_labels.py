"""检查 corrected_labels 的使用顺序"""
import re

with open('D:/AIC-123/code/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到 train_one_epoch 函数
func_match = re.search(r'def train_one_epoch$.*?\n(.*?)(?=\ndef |\Z)', content, re.DOTALL)

if func_match:
    func_body = func_match.group(1)
    
    # 找到所有 corrected_labels 的使用位置
    uses = []
    for match in re.finditer(r'corrected_labels', func_body):
        start = match.start()
        # 获取前后文
        context_start = max(0, start - 80)
        context_end = min(len(func_body), start + 80)
        context = func_body[context_start:context_end]
        
        # 判断是赋值还是使用
        line_start = func_body.rfind('\n', 0, start) + 1
        line_end = func_body.find('\n', start)
        line = func_body[line_start:line_end].strip()
        
        is_assignment = line.startswith('corrected_labels =')
        uses.append((start, is_assignment, line))
    
    print("=== corrected_labels 使用顺序 ===\n")
    for i, (pos, is_assign, line) in enumerate(uses):
        marker = "✓ 赋值" if is_assign else "✗ 使用"
        print(f"{i+1}. [{marker}] 位置 {pos}")
        print(f"   {line[:100]}")
        print()
    
    # 检查第一次出现是否是赋值
    if uses:
        first_is_assignment = uses[0][1]
        if first_is_assignment:
            print("✓ 第一次出现就是赋值（正确）")
        else:
            print("✗ 第一次出现是使用，不是赋值（错误）")
            print(f"   第一次赋值在第 {next((i+1 for i, (_, is_assign, _) in enumerate(uses) if is_assign), 'N/A')} 次出现")
else:
    print("✗ 找不到 train_one_epoch 函数")