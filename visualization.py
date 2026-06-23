from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


class VisualizationError(Exception):
    pass


SUPPORTED_TYPES = {"bar", "line", "hist", "pie", "scatter", "box", "count"}


def detect_chart_type(query: str) -> Optional[str]:
    """
    Keyword-based fallback chart type detection (LLM is primary).
    """
    q = query.lower()
    if "scatter" in q or "vs" in q:
        return "scatter"
    if "hist" in q or "distribution" in q:
        return "hist"
    if "pie" in q or "percentage" in q or "share" in q:
        return "pie"
    if "trend" in q or "over time" in q or "line" in q:
        return "line"
    if "box" in q or "outlier" in q:
        return "box"
    if "count" in q or "frequency" in q:
        return "count"
    if "bar" in q:
        return "bar"
    return None


def validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    cols = list(columns)
    missing = [c for c in cols if c not in set(df.columns)]
    if missing:
        raise VisualizationError("Column not found")
    return cols


def preprocess_for_plot(
    df: pd.DataFrame,
    chart_type: str,
    columns: List[str],
) -> pd.DataFrame:
    """
    Lightweight plot-specific preprocessing:
    - Convert numeric columns for numeric charts using pd.to_numeric(errors="coerce")
    - Drop NaN rows required for the plot
    """
    out = df.copy()

    def to_num(col: str) -> None:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if chart_type in {"line", "hist", "box"}:
        to_num(columns[0])
        out = out.dropna(subset=[columns[0]])
        if out.empty:
            raise VisualizationError("No numeric data available to plot.")

    if chart_type == "scatter":
        to_num(columns[0])
        to_num(columns[1])
        out = out.dropna(subset=[columns[0], columns[1]])
        if out.empty:
            raise VisualizationError("No numeric data available to plot.")

    # For bar/count/pie we primarily use value_counts and can tolerate NaNs by dropping.
    if chart_type in {"bar", "count", "pie"}:
        out = out.dropna(subset=[columns[0]])
        if out.empty:
            raise VisualizationError("No categorical data available to plot.")

    return out


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    General cleanup for visualization: attempt numeric conversion for object columns
    where it doesn't destroy data (kept conservative by only converting during plot preprocessing).
    This is provided as an extension point.
    """
    return df


@dataclass(frozen=True)
class ChartSpec:
    type: str
    columns: List[str]


@dataclass(frozen=True)
class ChartOptions:
    width: float = 8.0
    height: float = 4.5
    bins: int = 30
    top_n: int = 20
    color: str = "#1f77b4"
    alpha: float = 0.85
    grid: bool = True


def generate_chart(
    df: pd.DataFrame,
    query: str,
    spec: ChartSpec,
    options: Optional[ChartOptions] = None,
) -> plt.Figure:
    """
    Safe visualization engine: uses predefined templates only (no arbitrary code).
    """
    chart_type = spec.type.lower().strip()
    if chart_type not in SUPPORTED_TYPES:
        raise VisualizationError("Unsupported chart type")

    columns = validate_columns(df, spec.columns)

    if chart_type == "scatter" and len(columns) != 2:
        raise VisualizationError("Scatter plot requires exactly two columns.")
    if chart_type in {"bar", "line", "hist", "pie", "box", "count"} and len(columns) != 1:
        raise VisualizationError("This chart type requires exactly one column.")

    plot_df = preprocess_for_plot(df, chart_type, columns)

    opts = options or ChartOptions()
    fig, ax = plt.subplots(figsize=(opts.width, opts.height))
    title = query.strip() or f"{chart_type} plot"

    col = columns[0]

    if chart_type in {"bar", "count"}:
        series = plot_df[col].value_counts().head(opts.top_n)
        if series.empty:
            raise VisualizationError("No data available to plot.")
        series.plot(kind="bar", ax=ax, color=opts.color, alpha=opts.alpha)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        if len(plot_df[col].value_counts()) > opts.top_n:
            ax.set_title(f"{title} (top {opts.top_n})")

    elif chart_type == "pie":
        series = plot_df[col].value_counts().head(opts.top_n)
        if series.empty:
            raise VisualizationError("No data available to plot.")
        series.plot(kind="pie", ax=ax, autopct="%1.1f%%")
        ax.set_ylabel("")

    elif chart_type == "line":
        plot_df[col].plot(kind="line", ax=ax, color=opts.color, alpha=opts.alpha)
        ax.set_xlabel("index")
        ax.set_ylabel(col)

    elif chart_type == "hist":
        plot_df[col].plot(kind="hist", ax=ax, bins=opts.bins, color=opts.color, alpha=opts.alpha)
        ax.set_xlabel(col)
        ax.set_ylabel("frequency")

    elif chart_type == "box":
        plot_df.boxplot(column=col, ax=ax)
        ax.set_ylabel(col)

    elif chart_type == "scatter":
        x, y = columns[0], columns[1]
        plot_df.plot(kind="scatter", x=x, y=y, ax=ax, color=opts.color, alpha=opts.alpha)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    ax.grid(opts.grid, linestyle="--", alpha=0.35)
    ax.set_title(title)
    fig.tight_layout()
    return fig

