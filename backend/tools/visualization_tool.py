#!/usr/bin/env python3
"""
Agent data visualization (Visualization Tool)

Lets the agent render charts:
1. Line — trends
2. Bar — comparisons
3. Pie — shares
4. Scatter — relationships

[2026-02-22] Added — data visualization
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

# Detect whether matplotlib is available
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    MATPLOTLIB_AVAILABLE = True

    # Prefer fonts that support CJK labels when present
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    logger.info("Matplotlib available for visualization")
except ImportError:
    logger.warning("Matplotlib not installed. Visualization will be disabled.")


class VisualizationTool:
    """Agent data visualization tool."""

    def __init__(self, output_dir: str = "workspace/sandbox/charts"):
        """
        Initialize the visualization tool.

        Args:
            output_dir: Directory to write chart images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = MATPLOTLIB_AVAILABLE

    def create_line_chart(
        self,
        data: Dict[str, List[Union[int, float]]],
        title: str = "Line chart",
        x_label: str = "X",
        y_label: str = "Y",
        x_values: Optional[List] = None,
        figsize: tuple = (10, 6)
    ) -> Dict[str, Any]:
        """
        Create a line chart.

        Args:
            data: Series dict, e.g. {"A": [1,2,3], "B": [4,5,6]}
            title: Chart title
            x_label: X axis label
            y_label: Y axis label
            x_values: Optional x tick values
            figsize: Figure size

        Returns:
            Result dict including image path
        """
        if not self.enabled:
            return {"success": False, "error": "Matplotlib not available"}

        try:
            fig, ax = plt.subplots(figsize=figsize)

            for label, values in data.items():
                x = x_values if x_values else list(range(len(values)))
                ax.plot(x, values, marker='o', label=label)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True, alpha=0.3)

            filepath = self._save_figure(fig, "line")
            plt.close(fig)

            return {
                "success": True,
                "chart_type": "line",
                "title": title,
                "path": str(filepath),
                "data_series": list(data.keys())
            }

        except Exception as e:
            logger.error(f"[VIZ] Line chart failed: {e}")
            return {"success": False, "error": str(e)}

    def create_bar_chart(
        self,
        data: Dict[str, Union[int, float]],
        title: str = "Bar chart",
        x_label: str = "Category",
        y_label: str = "Value",
        horizontal: bool = False,
        figsize: tuple = (10, 6)
    ) -> Dict[str, Any]:
        """
        Create a bar chart.

        Args:
            data: Category dict, e.g. {"A": 10, "B": 20}
            title: Chart title
            x_label: X axis label
            y_label: Y axis label
            horizontal: Horizontal bars if True
            figsize: Figure size

        Returns:
            Result dict
        """
        if not self.enabled:
            return {"success": False, "error": "Matplotlib not available"}

        try:
            fig, ax = plt.subplots(figsize=figsize)

            categories = list(data.keys())
            values = list(data.values())

            colors = plt.cm.Set3(range(len(categories)))

            if horizontal:
                bars = ax.barh(categories, values, color=colors)
                ax.set_xlabel(y_label)
                ax.set_ylabel(x_label)
            else:
                bars = ax.bar(categories, values, color=colors)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                plt.xticks(rotation=45, ha='right')

            ax.set_title(title, fontsize=14, fontweight='bold')

            for bar, val in zip(bars, values):
                if horizontal:
                    ax.text(val, bar.get_y() + bar.get_height()/2,
                            f' {val:.1f}', va='center')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{val:.1f}', ha='center', va='bottom')

            plt.tight_layout()
            filepath = self._save_figure(fig, "bar")
            plt.close(fig)

            return {
                "success": True,
                "chart_type": "bar",
                "title": title,
                "path": str(filepath),
                "categories": categories
            }

        except Exception as e:
            logger.error(f"[VIZ] Bar chart failed: {e}")
            return {"success": False, "error": str(e)}

    def create_pie_chart(
        self,
        data: Dict[str, Union[int, float]],
        title: str = "Pie chart",
        show_percentage: bool = True,
        figsize: tuple = (8, 8)
    ) -> Dict[str, Any]:
        """
        Create a pie chart.

        Args:
            data: Category dict, e.g. {"A": 10, "B": 20}
            title: Chart title
            show_percentage: Show percentage labels
            figsize: Figure size

        Returns:
            Result dict
        """
        if not self.enabled:
            return {"success": False, "error": "Matplotlib not available"}

        try:
            fig, ax = plt.subplots(figsize=figsize)

            labels = list(data.keys())
            values = list(data.values())
            colors = plt.cm.Pastel1(range(len(labels)))

            autopct = '%1.1f%%' if show_percentage else None

            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                autopct=autopct,
                colors=colors,
                startangle=90,
                explode=[0.02] * len(labels)
            )

            ax.set_title(title, fontsize=14, fontweight='bold')

            filepath = self._save_figure(fig, "pie")
            plt.close(fig)

            return {
                "success": True,
                "chart_type": "pie",
                "title": title,
                "path": str(filepath),
                "categories": labels,
                "total": sum(values)
            }

        except Exception as e:
            logger.error(f"[VIZ] Pie chart failed: {e}")
            return {"success": False, "error": str(e)}

    def create_scatter_chart(
        self,
        x_data: List[Union[int, float]],
        y_data: List[Union[int, float]],
        title: str = "Scatter chart",
        x_label: str = "X",
        y_label: str = "Y",
        show_trend: bool = False,
        figsize: tuple = (10, 6)
    ) -> Dict[str, Any]:
        """
        Create a scatter plot.

        Args:
            x_data: X values
            y_data: Y values
            title: Chart title
            x_label: X axis label
            y_label: Y axis label
            show_trend: Draw a linear trend line
            figsize: Figure size

        Returns:
            Result dict
        """
        if not self.enabled:
            return {"success": False, "error": "Matplotlib not available"}

        try:
            import numpy as np

            fig, ax = plt.subplots(figsize=figsize)

            ax.scatter(x_data, y_data, alpha=0.6, edgecolors='black', linewidth=0.5)

            if show_trend and len(x_data) >= 2:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(x_data), max(x_data), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, label="Trend line")
                ax.legend()

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)

            filepath = self._save_figure(fig, "scatter")
            plt.close(fig)

            return {
                "success": True,
                "chart_type": "scatter",
                "title": title,
                "path": str(filepath),
                "data_points": len(x_data)
            }

        except Exception as e:
            logger.error(f"[VIZ] Scatter chart failed: {e}")
            return {"success": False, "error": str(e)}

    def _save_figure(self, fig, chart_type: str) -> Path:
        """Save figure to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chart_type}_{timestamp}.png"
        filepath = self.output_dir / filename

        fig.savefig(filepath, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        logger.info(f"[VIZ] Chart saved: {filepath}")
        return filepath

    def list_charts(self, limit: int = 10) -> List[Dict]:
        """List recently generated chart images."""
        charts = []

        try:
            for f in sorted(self.output_dir.glob("*.png"), reverse=True)[:limit]:
                charts.append({
                    "filename": f.name,
                    "path": str(f),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                })
        except Exception as e:
            logger.error(f"[VIZ] Failed to list charts: {e}")

        return charts

    def get_tool_definitions(self) -> List[Dict]:
        """Return tool definitions for the router."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_chart",
                    "description": """Data visualization — create a chart (saved under workspace/sandbox/charts/).

Types:
- line: trends
- bar: comparisons
- pie: proportions
- scatter: relationships / correlation

When users ask in Chinese or English to plot, chart, or visualize data, use this tool.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chart_type": {
                                "type": "string",
                                "enum": ["line", "bar", "pie", "scatter"],
                                "description": "Chart type"
                            },
                            "data": {
                                "type": "object",
                                "description": "Bar/pie: {'cat_a': 10, 'cat_b': 20}; line: {'series1': [1,2,3]}; scatter: {'x': [...], 'y': [...]}"
                            },
                            "title": {
                                "type": "string",
                                "description": "Chart title"
                            },
                            "x_label": {
                                "type": "string",
                                "description": "X axis label"
                            },
                            "y_label": {
                                "type": "string",
                                "description": "Y axis label"
                            }
                        },
                        "required": ["chart_type", "data", "title"]
                    }
                }
            }
        ]

    def route_create_chart(self, args: Dict) -> Dict:
        """Dispatch create_chart requests."""
        chart_type = args.get("chart_type", "bar")
        data = args.get("data", {})
        title = args.get("title", "Chart")
        x_label = args.get("x_label", "X")
        y_label = args.get("y_label", "Y")

        if chart_type == "line":
            return self.create_line_chart(data, title, x_label, y_label)
        elif chart_type == "bar":
            return self.create_bar_chart(data, title, x_label, y_label)
        elif chart_type == "pie":
            return self.create_pie_chart(data, title)
        elif chart_type == "scatter":
            x_data = data.get("x", [])
            y_data = data.get("y", [])
            return self.create_scatter_chart(x_data, y_data, title, x_label, y_label)
        else:
            return {"success": False, "error": f"Unknown chart type: {chart_type}"}


_instance = None

def get_visualization_tool() -> VisualizationTool:
    """Return the singleton visualization tool instance."""
    global _instance
    if _instance is None:
        _instance = VisualizationTool()
    return _instance
