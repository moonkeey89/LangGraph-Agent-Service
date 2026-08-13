"""
数学计算Skill
"""


def calculate(expression):
    """
    执行数学表达式计算

    参数:
        expression:
            数学表达式字符串

    返回:
        计算结果
    """


    try:

        result = eval(expression)

        return str(result)


    except:

        return "无法计算"