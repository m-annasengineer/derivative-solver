from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ast
import sympy as sp
import re

app = FastAPI(title="Derivative Solver API")

class Question(BaseModel):
    question: str

FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs
}

CONSTANTS = {
    "pi": sp.pi,
    "e": sp.E
}

def parse_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return sp.sympify(node.value)
        raise ValueError("Invalid constant")

    if isinstance(node, ast.Name):
        if node.id in FUNCTIONS or node.id in CONSTANTS:
            raise ValueError(f"Invalid standalone name: {node.id}")
        return sp.Symbol(node.id)

    if isinstance(node, ast.BinOp):
        left = parse_node(node.left)
        right = parse_node(node.right)

        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: left ** right
        }

        operation = operations.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported operator")
        return operation()

    if isinstance(node, ast.UnaryOp):
        value = parse_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("Unsupported unary operator")

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in FUNCTIONS or len(node.args) != 1 or node.keywords:
            raise ValueError(f"Unsupported function: {name}")
        return FUNCTIONS[name](parse_node(node.args[0]))

    raise ValueError("Unsupported mathematical expression")

def parse_expression(question):
    question = question.strip().replace("^", "**")
    question = question.replace("−", "-").replace("×", "*").replace("÷", "/")

    question = re.sub(
        r"^\s*[a-zA-Z]\s*\([^()]*\)\s*=\s*",
        "",
        question
    )

    question = re.sub(r"\b(pi|e)\b", r"\1", question)

    tree = ast.parse(question, mode="eval")
    expression = parse_node(tree.body)

    names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id not in FUNCTIONS
        and node.id not in CONSTANTS
    }

    return expression, sorted(names)

@app.get("/")
def home():
    return {"status": "Derivative Solver API is running"}

@app.post("/solve")
def solve(data: Question):
    try:
        expression, variable_names = parse_expression(data.question)

        if not variable_names:
            return {
                "question": data.question,
                "variables": [],
                "derivatives": {},
                "message": "No variables found"
            }

        symbols = {
            name: sp.Symbol(name)
            for name in variable_names
        }

        derivatives = {}

        for name in variable_names:
            symbol = symbols[name]
            result = sp.diff(expression, symbol)

            derivatives[name] = {
                "steps": [
                    f"f = {sp.sstr(expression)}",
                    f"∂f/∂{name} = {sp.sstr(result)}"
                ],
                "final_answer": sp.sstr(result)
            }

        return {
            "question": data.question,
            "function": sp.sstr(expression),
            "variables": variable_names,
            "derivatives": derivatives
        }

    except (ValueError, SyntaxError, TypeError, sp.SympifyError) as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or unsupported expression: {error}"
        )
