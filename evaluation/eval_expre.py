import re
from sympy import sympify, SympifyError

def extract_formula_robust(text: str) -> str | None:
    """
    从文本中稳健地提取 n = ... 公式。
    它会找到'n ='右侧的最长可解析数学表达式。
    """
    # 寻找 n = ... 的模式，并捕获等号右边的所有内容
    match = re.search(r'n\s*=\s*(.*)', text, re.IGNORECASE)
    if not match:
        return None

    potential_formula_str = match.group(1).strip()
    
    # 预处理，将幂符号统一为 '**'
    potential_formula_str = potential_formula_str.replace('^', '**')

    longest_valid_formula = None
    
    # 从左到右迭代，寻找最长的可被 sympify 解析的子串
    for i in range(1, len(potential_formula_str) + 1):
        substring = potential_formula_str[:i]
        try:
            # 尝试解析
            sympify(substring)
            # 如果成功，更新最长的有效公式
            longest_valid_formula = substring
        except (SympifyError, TypeError, ValueError, SyntaxError):
            # 如果在某个点解析失败，说明公式部分到此为止
            # 我们可以提前中断循环，因为后面的更长字符串也必然会失败
            break
            
    # 返回去除尾部空格的最长有效公式
    return longest_valid_formula.strip() if longest_valid_formula else None


def are_expressions_equivalent(expr1_str: str, expr2_str: str) -> bool:
    """
    使用 SymPy 检查两个数学表达式字符串是否等价。
    """
    if not expr1_str or not expr2_str:
        return False
    try:
        sym_expr1 = sympify(expr1_str)
        sym_expr2 = sympify(expr2_str)
        
        # simplify(expr1 - expr2) == 0 是最鲁棒的判断方式
        return (sym_expr1 - sym_expr2).simplify() == 0
    except (SympifyError, TypeError, ValueError):
        return False

# --- 评估循环中的逻辑 ---
ground_truth_formula = "2**t - 1"

# 使用你提供的例子进行测试
model_outputs = [
    "The solution is n=2^t-1",
    "I found that n = 2**t - 1 for any integer t.", # 这个之前会出错
    "The answer is n=2t-1", # 错误的答案
    "n = (2**t) - 1 .", # 包含句号的正确答案
    "I am unable to solve this."
]

print("--- 使用优化后的函数进行测试 ---")
for output in model_outputs:
    # 使用新的、更健壮的提取函数
    model_formula = extract_formula_robust(output)
    
    print(f"Model Output: '{output}'")
    if model_formula:
        is_correct = are_expressions_equivalent(model_formula, ground_truth_formula)
        print(f"  -> Parsed Formula: '{model_formula}', Correct: {is_correct}")
    else:
        print("  -> No formula found.")
    print("-" * 20)