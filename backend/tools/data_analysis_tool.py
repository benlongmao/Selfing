#!/usr/bin/env python3
"""
Lightweight numeric stats + optional SymPy (solve / diff / int).

[2026-02-22] Smaller surface than ``scientific_computing_tool`` for everyday chat tasks.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)


class DataAnalysisTool:
    """NumPy summary stats and SymPy helpers when installed."""

    def __init__(self):
        self.sympy_available = False
        try:
            import sympy  # noqa: F401
            self.sympy_available = True
        except ImportError:
            logger.warning("SymPy not available, equation tools disabled")

    def analyze_numbers(self, numbers: List[Union[int, float]]) -> Dict[str, Any]:
        """Descriptive stats for a 1D numeric list."""
        if not numbers:
            return {"success": False, "error": "Empty data"}

        try:
            data = np.array(numbers, dtype=float)

            result = {
                "success": True,
                "count": len(data),
                "sum": float(np.sum(data)),
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "range": float(np.max(data) - np.min(data)),
            }

            result["q1"] = float(np.percentile(data, 25))
            result["q3"] = float(np.percentile(data, 75))
            result["iqr"] = result["q3"] - result["q1"]

            result["summary"] = (
                f"n={result['count']}, min={result['min']:.2f}, max={result['max']:.2f}, "
                f"mean={result['mean']:.2f}, median={result['median']:.2f}, std={result['std']:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"[DATA] Analysis failed: {e}")
            return {"success": False, "error": str(e)}

    def solve_equation(self, equation: str, variable: str = "x") -> Dict[str, Any]:
        """Solve ``equation = 0`` for ``variable``."""
        if not self.sympy_available:
            return {"success": False, "error": "SymPy not available"}

        try:
            import sympy as sp

            var = sp.Symbol(variable)
            expr = sp.sympify(equation)
            solutions = sp.solve(expr, var)

            return {
                "success": True,
                "equation": f"{equation} = 0",
                "variable": variable,
                "solutions": [str(sol) for sol in solutions],
                "solutions_numeric": [complex(sol.evalf()) if sol.is_number else None for sol in solutions]
            }

        except Exception as e:
            logger.error(f"[DATA] Solve failed: {e}")
            return {"success": False, "error": str(e)}

    def calculate_derivative(self, expression: str, variable: str = "x") -> Dict[str, Any]:
        """Symbolic derivative."""
        if not self.sympy_available:
            return {"success": False, "error": "SymPy not available"}

        try:
            import sympy as sp

            var = sp.Symbol(variable)
            expr = sp.sympify(expression)
            derivative = sp.diff(expr, var)

            return {
                "success": True,
                "expression": expression,
                "derivative": str(derivative),
                "latex": sp.latex(derivative)
            }

        except Exception as e:
            logger.error(f"[DATA] Derivative failed: {e}")
            return {"success": False, "error": str(e)}

    def calculate_integral(self, expression: str, variable: str = "x") -> Dict[str, Any]:
        """Indefinite integral (+ C)."""
        if not self.sympy_available:
            return {"success": False, "error": "SymPy not available"}

        try:
            import sympy as sp

            var = sp.Symbol(variable)
            expr = sp.sympify(expression)
            integral = sp.integrate(expr, var)

            return {
                "success": True,
                "expression": expression,
                "integral": str(integral) + " + C",
                "latex": sp.latex(integral) + " + C"
            }

        except Exception as e:
            logger.error(f"[DATA] Integral failed: {e}")
            return {"success": False, "error": str(e)}

    def get_tool_definitions(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze_data",
                    "description": "Descriptive stats for a list of numbers (count, min, max, mean, median, quartiles, etc.).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "numbers": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Numeric array to summarize"
                            }
                        },
                        "required": ["numbers"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "solve_math",
                    "description": """SymPy: operation is solve | derivative | integral on a Python/SymPy expression (use ** for powers). Natural-language aliases include equation solving, differentiation, and integration.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["solve", "derivative", "integral"],
                                "description": "Task type"
                            },
                            "expression": {
                                "type": "string",
                                "description": "Expression, e.g. 'x**2 - 4'"
                            },
                            "variable": {
                                "type": "string",
                                "description": "Variable name (default x)"
                            }
                        },
                        "required": ["operation", "expression"]
                    }
                }
            }
        ]

    def route(self, tool_name: str, args: Dict) -> Dict:
        if tool_name == "analyze_data":
            return self.analyze_numbers(args.get("numbers", []))
        if tool_name == "solve_math":
            op = args.get("operation", "solve")
            expr = args.get("expression", "")
            var = args.get("variable", "x")

            if op == "solve":
                return self.solve_equation(expr, var)
            if op == "derivative":
                return self.calculate_derivative(expr, var)
            if op == "integral":
                return self.calculate_integral(expr, var)
            return {"success": False, "error": f"Unknown operation: {op}"}
        return {"success": False, "error": f"Unknown tool: {tool_name}"}


_instance = None

def get_data_analysis_tool() -> DataAnalysisTool:
    global _instance
    if _instance is None:
        _instance = DataAnalysisTool()
    return _instance
