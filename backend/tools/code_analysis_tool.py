#!/usr/bin/env python3
"""
AST-based helpers to inspect Python structure, quality heuristics, and call graphs.
"""
import ast
import os
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class CodeAnalysisTool:
    """Lightweight Python static analysis for agent tooling."""

    def __init__(self, sandbox_dir: str = "."):
        """
        Resolve ``sandbox_dir`` against the repository root derived from this file.

        [2026-02-05] Default working tree is the project root (not ``os.getcwd()``).
        """
        # Project root directory = the upper level of the directory where this file is located
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.sandbox_dir = os.path.abspath(os.path.join(project_root, sandbox_dir))
    
    def _is_safe_path(self, path: str) -> bool:
        """
        Ensure ``path`` resolves inside ``self.sandbox_dir`` (treated as project root).
        """
        abs_path = os.path.abspath(os.path.join(self.sandbox_dir, path))
        # [2026-02-05 Fix] Directly use self.sandbox_dir as the project root directory
        project_root = self.sandbox_dir
        return abs_path.startswith(project_root)
    
    # ==================== 1. Basic code analysis ====================    
    def analyze_python_file(self, filename: str) -> Dict[str, Any]:
        """
        Parse ``filename`` and return functions, classes, imports, line stats, and rough complexity.
        """
        if not self._is_safe_path(filename):
            return {"error": "Path outside sandbox"}
        
        filepath = os.path.join(self.sandbox_dir, filename)
        
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filename}"}
        
        if not filename.endswith('.py'):
            return {"error": "Not a Python file"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse AST
            tree = ast.parse(source_code, filename=filename)
            
            # Extract information
            functions = self._extract_functions(tree)
            classes = self._extract_classes(tree)
            imports = self._extract_imports(tree)
            
            # Statistics
            lines = source_code.split('\n')
            total_lines = len(lines)
            code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            
            return {
                "success": True,
                "file": filename,
                "statistics": {
                    "total_lines": total_lines,
                    "code_lines": code_lines,
                    "comment_lines": comment_lines,
                    "blank_lines": total_lines - code_lines - comment_lines
                },
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "complexity": self._calculate_complexity(tree)
            }
            
        except SyntaxError as e:
            return {
                "error": f"Syntax error in file: {e}",
                "line": e.lineno,
                "offset": e.offset
            }
        except Exception as e:
            logger.error(f"Failed to analyze file: {e}")
            return {"error": str(e)}
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Collect function metadata from an AST."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract parameters
                args = [arg.arg for arg in node.args.args]
                
                # Extract decorator
                decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
                
                # Extract docstring
                docstring = ast.get_docstring(node)
                
                # Count rows
                end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno
                
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "end_line": end_lineno,
                    "args": args,
                    "decorators": decorators,
                    "docstring": docstring[:100] + "..." if docstring and len(docstring) > 100 else docstring,
                    "is_async": isinstance(node, ast.AsyncFunctionDef)
                })
        
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Collect class metadata from an AST."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract base class
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(f"{base.value.id}.{base.attr}")
                
                # Extraction method
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            "name": item.name,
                            "line": item.lineno,
                            "is_private": item.name.startswith('_'),
                            "is_magic": item.name.startswith('__') and item.name.endswith('__')
                        })
                
                # Extract docstring
                docstring = ast.get_docstring(node)
                
                classes.append({
                    "name": node.name,
                    "line": node.lineno,
                    "bases": bases,
                    "methods": methods,
                    "method_count": len(methods),
                    "docstring": docstring[:100] + "..." if docstring and len(docstring) > 100 else docstring
                })
        
        return classes
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, List]:
        """Collect import statements grouped coarsely by origin."""
        imports = {
            "standard": [],
            "third_party": [],
            "local": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    as_name = alias.asname
                    self._categorize_import(imports, module, as_name)
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.name
                    as_name = alias.asname
                    full_name = f"{module}.{name}" if module else name
                    self._categorize_import(imports, full_name, as_name)
        
        return imports
    
    def _categorize_import(self, imports: Dict, module: str, as_name: Optional[str]):
        """Bucket an import into standard / third_party / local."""
        import sys
        
        # Simple classification logic
        top_level = module.split('.')[0]
        
        # Standard library list (part)
        stdlib = {
            'os', 'sys', 'json', 'ast', 'logging', 'time', 'datetime', 
            'collections', 'typing', 'pathlib', 're', 'copy', 'math',
            'random', 'hashlib', 'functools', 'itertools', 'dataclasses'
        }
        
        import_info = {
            "module": module,
            "as": as_name
        }
        
        if top_level in stdlib:
            imports["standard"].append(import_info)
        elif top_level.startswith('.') or top_level == 'backend':
            imports["local"].append(import_info)
        else:
            imports["third_party"].append(import_info)
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Return a readable decorator identifier."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return f"{decorator.func.value.id}.{decorator.func.attr}"
        return "unknown"
    
    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Very rough McCabe-like counters for agent hints."""
        complexity = {
            "cyclomatic": 1,  # McCabe-style baseline
            "nesting_depth": 0,
            "conditions": 0,
            "loops": 0
        }
        
        # Simplified complexity calculation
        for node in ast.walk(tree):
            # Branch statements increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity["cyclomatic"] += 1
            
            if isinstance(node, (ast.If, ast.Compare, ast.BoolOp)):
                complexity["conditions"] += 1
            
            if isinstance(node, (ast.While, ast.For)):
                complexity["loops"] += 1
        
        return complexity
    
    # ==================== 2. Code quality inspection ====================    
    def check_code_quality(self, filename: str) -> Dict[str, Any]:
        """
        Emit simple warnings (long functions, many parameters, missing docstrings).
        """
        if not self._is_safe_path(filename):
            return {"error": "Path outside sandbox"}
        
        filepath = os.path.join(self.sandbox_dir, filename)
        
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filename}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            issues = []
            warnings = []
            suggestions = []
            
            # check function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check function length
                    func_lines = (node.end_lineno - node.lineno) if hasattr(node, 'end_lineno') else 0
                    if func_lines > 50:
                        issues.append({
                            "type": "long_function",
                            "severity": "warning",
                            "line": node.lineno,
                            "function": node.name,
                            "message": f"Function '{node.name}' is {func_lines} lines long (recommended < 50)"
                        })
                    
                    # Check the number of parameters
                    arg_count = len(node.args.args)
                    if arg_count > 5:
                        warnings.append({
                            "type": "too_many_args",
                            "severity": "warning",
                            "line": node.lineno,
                            "function": node.name,
                            "message": f"Function '{node.name}' has {arg_count} parameters (recommended < 5)"
                        })
                    
                    # Check docstring
                    if not ast.get_docstring(node) and not node.name.startswith('_'):
                        suggestions.append({
                            "type": "missing_docstring",
                            "severity": "info",
                            "line": node.lineno,
                            "function": node.name,
                            "message": f"Function '{node.name}' is missing a docstring"
                        })
            
            return {
                "success": True,
                "file": filename,
                "issues": issues,
                "warnings": warnings,
                "suggestions": suggestions,
                "summary": {
                    "issues_count": len(issues),
                    "warnings_count": len(warnings),
                    "suggestions_count": len(suggestions)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to check quality: {e}")
            return {"error": str(e)}
    
    # ==================== 3. Code dependency analysis ====================    
    def analyze_dependencies(self, filename: str) -> Dict[str, Any]:
        """Summarize top function/method call sites (static, best-effort)."""
        if not self._is_safe_path(filename):
            return {"error": "Path outside sandbox"}
        
        filepath = os.path.join(self.sandbox_dir, filename)
        
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filename}"}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            # Extract function calls
            function_calls = defaultdict(list)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        function_calls[func_name].append({
                            "line": node.lineno,
                            "args_count": len(node.args)
                        })
                    elif isinstance(node.func, ast.Attribute):
                        func_name = f"{self._get_attr_chain(node.func)}"
                        function_calls[func_name].append({
                            "line": node.lineno,
                            "args_count": len(node.args)
                        })
            
            # Convert to list format
            call_list = [
                {
                    "function": func,
                    "call_count": len(calls),
                    "lines": [c["line"] for c in calls]
                }
                for func, calls in function_calls.items()
            ]
            
            # Sort by number of calls
            call_list.sort(key=lambda x: x["call_count"], reverse=True)
            
            return {
                "success": True,
                "file": filename,
                "function_calls": call_list[:20],  # cap payload size
                "total_unique_calls": len(function_calls),
                "most_called": call_list[0] if call_list else None
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze dependencies: {e}")
            return {"error": str(e)}
    
    def _get_attr_chain(self, node: ast.Attribute) -> str:
        """Serialize attribute chains such as ``obj.method``."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return '.'.join(reversed(parts))
    
    # ==================== 4. Code comparison ====================    
    def compare_files(self, file1: str, file2: str) -> Dict[str, Any]:
        """Diff structural metadata between two Python modules."""
        result1 = self.analyze_python_file(file1)
        result2 = self.analyze_python_file(file2)
        
        if "error" in result1:
            return {"error": f"Failed to analyze {file1}: {result1['error']}"}
        if "error" in result2:
            return {"error": f"Failed to analyze {file2}: {result2['error']}"}
        
        # Extract function name
        funcs1 = {f["name"] for f in result1["functions"]}
        funcs2 = {f["name"] for f in result2["functions"]}
        
        # Extract class name
        classes1 = {c["name"] for c in result1["classes"]}
        classes2 = {c["name"] for c in result2["classes"]}
        
        return {
            "success": True,
            "file1": file1,
            "file2": file2,
            "functions": {
                "only_in_file1": list(funcs1 - funcs2),
                "only_in_file2": list(funcs2 - funcs1),
                "common": list(funcs1 & funcs2),
                "total_file1": len(funcs1),
                "total_file2": len(funcs2)
            },
            "classes": {
                "only_in_file1": list(classes1 - classes2),
                "only_in_file2": list(classes2 - classes1),
                "common": list(classes1 & classes2),
                "total_file1": len(classes1),
                "total_file2": len(classes2)
            },
            "statistics": {
                "file1": result1["statistics"],
                "file2": result2["statistics"]
            },
            "complexity": {
                "file1": result1["complexity"],
                "file2": result2["complexity"]
            }
        }
    
    # ==================== Tool Definition ====================    
    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-style tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze_python_file",
                    "description": "Analyze a Python file: functions, classes, imports, line stats, rough complexity.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Path to the ``.py`` file relative to the configured sandbox root",
                            },
                        },
                        "required": ["filename"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_code_quality",
                    "description": "Run lightweight quality heuristics (length, arity, docstrings).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Path to the ``.py`` file relative to the sandbox root",
                            },
                        },
                        "required": ["filename"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_dependencies",
                    "description": "Summarize static call relationships inside a Python module.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Path to the ``.py`` file relative to the sandbox root",
                            },
                        },
                        "required": ["filename"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_python_files",
                    "description": "Compare structure and complexity between two Python files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file1": {
                                "type": "string",
                                "description": "First ``.py`` path relative to the sandbox root",
                            },
                            "file2": {
                                "type": "string",
                                "description": "Second ``.py`` path relative to the sandbox root",
                            },
                        },
                        "required": ["file1", "file2"],
                    },
                },
            },
        ]
