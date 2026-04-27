#!/usr/bin/env python3
"""
Scientific computing helper: NumPy, SciPy, SymPy, and Pandas entry points.

[2026-01-30] Created for math / table workflows in the sandbox.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import scipy
    import scipy.optimize
    import scipy.stats
    import scipy.linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available")

try:
    import sympy
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("SymPy not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available")


class ScientificComputingTool:
    """Facade for small numpy/scipy/sympy/pandas operations."""

    def __init__(self):
        self.numpy_available = True
        self.scipy_available = SCIPY_AVAILABLE
        self.sympy_available = SYMPY_AVAILABLE
        self.pandas_available = PANDAS_AVAILABLE

        logger.info(
            f"ScientificComputingTool initialized: "
            f"NumPy={self.numpy_available}, "
            f"SciPy={self.scipy_available}, "
            f"SymPy={self.sympy_available}, "
            f"Pandas={self.pandas_available}"
        )

    def numpy_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Vector / stats helpers on NumPy.

        operation:
            array_creation | array_operation | linear_algebra | statistics
        """
        try:
            if operation == "array_creation":
                shape = kwargs.get("shape", [3, 3])
                dtype = kwargs.get("dtype", "float64")
                fill_value = kwargs.get("fill_value", 0.0)

                arr = np.full(shape, fill_value, dtype=dtype)
                return {
                    "success": True,
                    "result": arr.tolist(),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype)
                }

            if operation == "array_operation":
                arr1 = np.array(kwargs.get("array1", []))
                arr2 = np.array(kwargs.get("array2", []))
                op = kwargs.get("op", "add")

                if op == "add":
                    result = arr1 + arr2
                elif op == "subtract":
                    result = arr1 - arr2
                elif op == "multiply":
                    result = arr1 * arr2
                elif op == "divide":
                    result = arr1 / arr2
                elif op == "dot":
                    result = np.dot(arr1, arr2)
                else:
                    return {"success": False, "error": f"Unknown operation: {op}"}

                return {
                    "success": True,
                    "result": result.tolist() if isinstance(result, np.ndarray) else float(result),
                    "shape": list(result.shape) if isinstance(result, np.ndarray) else None
                }

            if operation == "linear_algebra":
                matrix = np.array(kwargs.get("matrix", []))
                op = kwargs.get("op", "det")

                if op == "det":
                    result = np.linalg.det(matrix)
                elif op == "inv":
                    result = np.linalg.inv(matrix)
                elif op == "eig":
                    eigenvalues, eigenvectors = np.linalg.eig(matrix)
                    result = {
                        "eigenvalues": eigenvalues.tolist(),
                        "eigenvectors": eigenvectors.tolist()
                    }
                elif op == "svd":
                    U, s, Vt = np.linalg.svd(matrix)
                    result = {
                        "U": U.tolist(),
                        "s": s.tolist(),
                        "Vt": Vt.tolist()
                    }
                else:
                    return {"success": False, "error": f"Unknown operation: {op}"}

                if isinstance(result, dict):
                    return {"success": True, "result": result}
                return {
                    "success": True,
                    "result": result.tolist() if isinstance(result, np.ndarray) else float(result)
                }

            if operation == "statistics":
                data = np.array(kwargs.get("data", []))
                op = kwargs.get("op", "mean")

                if op == "mean":
                    result = np.mean(data)
                elif op == "std":
                    result = np.std(data)
                elif op == "median":
                    result = np.median(data)
                elif op == "min":
                    result = np.min(data)
                elif op == "max":
                    result = np.max(data)
                elif op == "percentile":
                    percentile = kwargs.get("percentile", 50)
                    result = np.percentile(data, percentile)
                else:
                    return {"success": False, "error": f"Unknown operation: {op}"}

                return {
                    "success": True,
                    "result": float(result),
                    "operation": op
                }

            return {"success": False, "error": f"Unknown operation type: {operation}"}

        except Exception as e:
            logger.error(f"NumPy operation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def scipy_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        SciPy: optimize (demo minimize) and stats.describe.
        """
        if not self.scipy_available:
            return {"success": False, "error": "SciPy not available"}

        try:
            if operation == "optimize":
                method = kwargs.get("method", "minimize")
                x0 = kwargs.get("initial_guess", [0.0])

                if method == "minimize":
                    from scipy.optimize import minimize
                    result = minimize(lambda x: x**2, x0)
                    return {
                        "success": True,
                        "result": {
                            "x": result.x.tolist(),
                            "fun": float(result.fun),
                            "success": result.success
                        }
                    }
                return {"success": False, "error": f"Unknown method: {method}"}

            if operation == "stats":
                data = np.array(kwargs.get("data", []))
                op = kwargs.get("operation", "describe")

                if op == "describe":
                    from scipy import stats
                    result = stats.describe(data)
                    return {
                        "success": True,
                        "result": {
                            "nobs": int(result.nobs),
                            "minmax": result.minmax,
                            "mean": float(result.mean),
                            "variance": float(result.variance),
                            "skewness": float(result.skewness),
                            "kurtosis": float(result.kurtosis)
                        }
                    }
                return {"success": False, "error": f"Unknown operation: {op}"}

            return {"success": False, "error": f"Unknown operation type: {operation}"}

        except Exception as e:
            logger.error(f"SciPy operation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def sympy_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """SymPy: simplify, solve, differentiate, integrate."""
        if not self.sympy_available:
            return {"success": False, "error": "SymPy not available"}

        try:
            import sympy as sp

            if operation == "simplify":
                expr_str = kwargs.get("expression", "")
                x = sp.Symbol('x')
                expr = sp.sympify(expr_str)
                result = sp.simplify(expr)
                return {
                    "success": True,
                    "result": str(result),
                    "latex": sp.latex(result)
                }

            if operation == "solve":
                equation_str = kwargs.get("equation", "")
                variable = kwargs.get("variable", "x")

                x = sp.Symbol(variable)
                equation = sp.sympify(equation_str)
                solutions = sp.solve(equation, x)

                return {
                    "success": True,
                    "result": [str(sol) for sol in solutions],
                    "latex": [sp.latex(sol) for sol in solutions]
                }

            if operation == "differentiate":
                expr_str = kwargs.get("expression", "")
                variable = kwargs.get("variable", "x")

                x = sp.Symbol(variable)
                expr = sp.sympify(expr_str)
                derivative = sp.diff(expr, x)

                return {
                    "success": True,
                    "result": str(derivative),
                    "latex": sp.latex(derivative)
                }

            if operation == "integrate":
                expr_str = kwargs.get("expression", "")
                variable = kwargs.get("variable", "x")

                x = sp.Symbol(variable)
                expr = sp.sympify(expr_str)
                integral = sp.integrate(expr, x)

                return {
                    "success": True,
                    "result": str(integral),
                    "latex": sp.latex(integral)
                }

            return {"success": False, "error": f"Unknown operation type: {operation}"}

        except Exception as e:
            logger.error(f"SymPy operation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def pandas_operation(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Pandas: create, describe, groupby, merge."""
        if not self.pandas_available:
            return {"success": False, "error": "Pandas not available"}

        try:
            if operation == "create_dataframe":
                data = kwargs.get("data", {})
                df = pd.DataFrame(data)
                return {
                    "success": True,
                    "result": df.to_dict(orient='records'),
                    "shape": list(df.shape),
                    "columns": df.columns.tolist()
                }

            if operation == "analyze":
                data = kwargs.get("data", {})
                df = pd.DataFrame(data)

                analysis = {
                    "shape": list(df.shape),
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "describe": df.describe().to_dict(),
                    "null_counts": df.isnull().sum().to_dict()
                }

                return {
                    "success": True,
                    "result": analysis
                }

            if operation == "groupby":
                data = kwargs.get("data", {})
                group_by = kwargs.get("group_by", "")
                agg_func = kwargs.get("aggregate", "mean")

                df = pd.DataFrame(data)
                grouped = df.groupby(group_by).agg(agg_func)

                return {
                    "success": True,
                    "result": grouped.to_dict(orient='index')
                }

            if operation == "merge":
                data1 = kwargs.get("data1", {})
                data2 = kwargs.get("data2", {})
                on = kwargs.get("on", "")
                how = kwargs.get("how", "inner")

                df1 = pd.DataFrame(data1)
                df2 = pd.DataFrame(data2)
                merged = pd.merge(df1, df2, on=on, how=how)

                return {
                    "success": True,
                    "result": merged.to_dict(orient='records'),
                    "shape": list(merged.shape)
                }

            return {"success": False, "error": f"Unknown operation type: {operation}"}

        except Exception as e:
            logger.error(f"Pandas operation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


_scientific_computing_tool = None

def get_scientific_computing_tool() -> ScientificComputingTool:
    """Singleton accessor."""
    global _scientific_computing_tool
    if _scientific_computing_tool is None:
        _scientific_computing_tool = ScientificComputingTool()
    return _scientific_computing_tool
