"""Safe arithmetic calculation business capability."""

import ast
import operator
from collections.abc import Callable


Number = int | float
MAX_EXPRESSION_LENGTH = 200
MAX_ABSOLUTE_VALUE = 1e100
MAX_EXPONENT = 100

BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _validate_number(value: object) -> Number:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("只支持整数和浮点数")

    if abs(value) > MAX_ABSOLUTE_VALUE:
        raise ValueError("计算结果过大")

    return value


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        return _validate_number(node.value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        operand = _evaluate(node.operand)
        return _validate_number(UNARY_OPERATORS[type(node.op)](operand))

    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("指数过大")

        result = BINARY_OPERATORS[type(node.op)](left, right)
        return _validate_number(result)

    raise ValueError("表达式包含不支持的内容")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression without executing Python code."""

    try:
        if len(expression) > MAX_EXPRESSION_LENGTH:
            raise ValueError("表达式过长")

        parsed = ast.parse(expression, mode="eval")
        return str(_evaluate(parsed.body))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return "无法计算"
