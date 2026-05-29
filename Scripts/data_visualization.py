"""
data_visualization.py
=====================
Automatic data visualization module for the AI-powered data preprocessing system.

Generates interactive Plotly charts and statistical summaries based on column
data types inferred from a cleaned pandas DataFrame.

Input : Cleaned pd.DataFrame (output of DataCleaning.clean_data / ai_agent)
Output: Dict with keys:
    - "univariate_charts"  : List[go.Figure]
    - "bivariate_charts"   : List[go.Figure]
    - "correlation_matrix" : go.Figure | None
    - "statistical_summary": Dict
"""

from __future__ import annotations

import os
import json
import re
from groq import Groq
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Colour palette & theme helpers
# ---------------------------------------------------------------------------

_PALETTE = px.colors.qualitative.Plotly          # primary discrete palette
_SEQ_PALETTE = "Blues"                            # sequential scale
_TEMPLATE = "plotly"                              # Plotly default theme

_LAYOUT_BASE = dict(
    template=_TEMPLATE,
    font=dict(family="Inter, Arial, sans-serif", size=13),
    margin=dict(l=60, r=40, t=60, b=60),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def _apply_base_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the shared layout settings and an optional title."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#2c3e50"), x=0.05),
        **_LAYOUT_BASE,
    )
    return fig


# ---------------------------------------------------------------------------
# 1. Column-type detection
# ---------------------------------------------------------------------------

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Analyse a DataFrame and bucket every column into a semantic type.

    Parameters
    ----------
    df : pd.DataFrame
        The (cleaned) input DataFrame.

    Returns
    -------
    Dict[str, List[str]]
        Keys:
        - ``"numerical"``   – continuous numeric columns
        - ``"categorical"`` – low-cardinality or object columns
        - ``"datetime"``    – date / time columns
        - ``"boolean"``     – columns with only two distinct values
        - ``"high_cardinality"`` – text-like columns (ID, free text, …)
    """
    result: Dict[str, List[str]] = {
        "numerical": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "high_cardinality": [],
    }

    n_rows = len(df)

    for col in df.columns:
        series = df[col].dropna()

        # --- datetime ---
        if pd.api.types.is_datetime64_any_dtype(series):
            result["datetime"].append(col)
            continue

        # Try coercing object columns to datetime
        if series.dtype == object:
            try:
                pd.to_datetime(series)
                result["datetime"].append(col)
                continue
            except Exception:
                pass

        # --- boolean ---
        if series.nunique() == 2:
            result["boolean"].append(col)
            continue

        # --- numerical ---
        if pd.api.types.is_numeric_dtype(series):
            result["numerical"].append(col)
            continue

        # --- object / string: cardinality split ---
        n_unique = series.nunique()
        ratio = n_unique / max(n_rows, 1)

        if ratio > 0.5 or n_unique > 50:
            result["high_cardinality"].append(col)
        else:
            result["categorical"].append(col)

    return result


# ---------------------------------------------------------------------------
# 2. Univariate plots
# ---------------------------------------------------------------------------

def generate_univariate_plots(
    df: pd.DataFrame,
    column: str,
    col_types: Optional[Dict[str, List[str]]] = None,
) -> go.Figure:
    """
    Generate an appropriate univariate chart for a single column.

    * Numerical  → Histogram + KDE overlay (using `histnorm='probability density'`)
    * Categorical / Boolean → Horizontal bar chart of value counts
    * Datetime   → Line chart of counts over time
    * High-cardinality → Fallback bar (top-20 values)

    Parameters
    ----------
    df         : Input DataFrame.
    column     : Column name to visualise.
    col_types  : Pre-computed output of :func:`detect_column_types`.
                 Will be computed on-the-fly if *None*.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with hover tooltips and zoom/pan enabled.
    """
    if col_types is None:
        col_types = detect_column_types(df)

    series = df[column].dropna()

    # ---- numerical ----
    if column in col_types["numerical"]:
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=series,
                name="Count",
                marker_color=_PALETTE[0],
                opacity=0.75,
                nbinsx=min(50, max(10, int(np.sqrt(len(series))))),
                hovertemplate="Value: %{x}<br>Count: %{y}<extra></extra>",
            )
        )

        # Overlay a smoothed density line using numpy
        counts, edges = np.histogram(series, bins=60, density=True)
        midpoints = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(
            go.Scatter(
                x=midpoints,
                y=counts,
                mode="lines",
                name="Density",
                line=dict(color=_PALETTE[1], width=2),
                yaxis="y2",
                hovertemplate="Value: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False,
                title="Density",
            ),
            bargap=0.05,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return _apply_base_layout(fig, f"Distribution of '{column}'")

    # ---- categorical / boolean ----
    if column in col_types["categorical"] or column in col_types["boolean"]:
        vc = series.value_counts().reset_index()
        vc.columns = ["value", "count"]

        fig = px.bar(
            vc,
            x="count",
            y="value",
            orientation="h",
            color="count",
            color_continuous_scale=_SEQ_PALETTE,
            labels={"value": column, "count": "Count"},
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
        )
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        return _apply_base_layout(fig, f"Value Counts – '{column}'")

    # ---- datetime ----
    if column in col_types["datetime"]:
        ts = pd.to_datetime(df[column].dropna())
        counts = ts.dt.to_period("M").value_counts().sort_index()
        periods = [str(p) for p in counts.index]

        fig = go.Figure(
            go.Scatter(
                x=periods,
                y=counts.values,
                mode="lines+markers",
                line=dict(color=_PALETTE[2], width=2),
                marker=dict(size=5),
                hovertemplate="Period: %{x}<br>Count: %{y}<extra></extra>",
            )
        )
        return _apply_base_layout(fig, f"Time Series – '{column}'")

    # ---- high-cardinality fallback ----
    vc = series.value_counts().head(20).reset_index()
    vc.columns = ["value", "count"]
    fig = px.bar(
        vc,
        x="value",
        y="count",
        color="count",
        color_continuous_scale=_SEQ_PALETTE,
        labels={"value": column, "count": "Count"},
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>")
    fig.update_layout(coloraxis_showscale=False)
    return _apply_base_layout(fig, f"Top 20 Values – '{column}'")


# ---------------------------------------------------------------------------
# 3. Bivariate plots
# ---------------------------------------------------------------------------

def generate_bivariate_plots(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    col_types: Optional[Dict[str, List[str]]] = None,
) -> go.Figure:
    """
    Generate an appropriate bivariate chart for two columns.

    Dispatch rules
    ~~~~~~~~~~~~~~
    +--------------+--------------+-----------------------------------+
    | col1 type    | col2 type    | Chart                             |
    +==============+==============+===================================+
    | numerical    | numerical    | Scatter + optional trend line     |
    | categorical  | numerical    | Box plot (group by categorical)   |
    | numerical    | categorical  | Box plot (group by categorical)   |
    | categorical  | categorical  | Grouped / stacked bar             |
    | datetime     | numerical    | Line chart                        |
    +--------------+--------------+-----------------------------------+

    Parameters
    ----------
    df    : Input DataFrame.
    col1  : First column name.
    col2  : Second column name.
    col_types : Pre-computed output of :func:`detect_column_types`.

    Returns
    -------
    go.Figure
    """
    if col_types is None:
        col_types = detect_column_types(df)

    def _type(col: str) -> str:
        for t, cols in col_types.items():
            if col in cols:
                return t
        return "unknown"

    t1, t2 = _type(col1), _type(col2)
    pair = df[[col1, col2]].dropna()

    # ---- num × num → scatter ----
    if t1 == "numerical" and t2 == "numerical":
        # Use OLS trendline only when statsmodels is available
        try:
            import statsmodels  # noqa: F401
            trendline = "ols"
        except ImportError:
            trendline = None

        fig = px.scatter(
            pair,
            x=col1,
            y=col2,
            trendline=trendline,
            trendline_color_override=_PALETTE[1] if trendline else None,
            color_discrete_sequence=[_PALETTE[0]],
            labels={col1: col1, col2: col2},
        )
        fig.update_traces(
            marker=dict(size=5, opacity=0.65),
            hovertemplate=f"{col1}: %{{x}}<br>{col2}: %{{y}}<extra></extra>",
            selector=dict(mode="markers"),
        )
        title = f"'{col1}' vs '{col2}'"
        return _apply_base_layout(fig, title)

    # ---- cat × num or num × cat → box ----
    if (t1 in ("categorical", "boolean") and t2 == "numerical") or \
       (t1 == "numerical" and t2 in ("categorical", "boolean")):
        cat_col, num_col = (col1, col2) if t1 in ("categorical", "boolean") else (col2, col1)
        fig = px.box(
            pair,
            x=cat_col,
            y=num_col,
            color=cat_col,
            color_discrete_sequence=_PALETTE,
        )
        fig.update_traces(
            hovertemplate=f"Group: %{{x}}<br>Value: %{{y}}<extra></extra>"
        )
        fig.update_layout(showlegend=False)
        return _apply_base_layout(fig, f"'{num_col}' by '{cat_col}'")

    # ---- cat × cat → stacked bar ----
    if t1 in ("categorical", "boolean") and t2 in ("categorical", "boolean"):
        cross = pd.crosstab(pair[col1], pair[col2])
        fig = go.Figure()
        for i, c in enumerate(cross.columns):
            fig.add_trace(
                go.Bar(
                    name=str(c),
                    x=cross.index.astype(str),
                    y=cross[c],
                    marker_color=_PALETTE[i % len(_PALETTE)],
                    hovertemplate=f"{col2}={c}<br>{col1}: %{{x}}<br>Count: %{{y}}<extra></extra>",
                )
            )
        fig.update_layout(barmode="stack")
        return _apply_base_layout(fig, f"'{col1}' × '{col2}'")

    # ---- datetime × num → line ----
    if t1 == "datetime" and t2 == "numerical":
        pair = pair.copy()
        pair[col1] = pd.to_datetime(pair[col1])
        pair = pair.sort_values(col1)
        fig = px.line(pair, x=col1, y=col2, color_discrete_sequence=[_PALETTE[2]])
        fig.update_traces(hovertemplate=f"{col1}: %{{x}}<br>{col2}: %{{y}}<extra></extra>")
        return _apply_base_layout(fig, f"'{col2}' over time")

    # ---- fallback: strip chart ----
    fig = px.strip(pair, x=col1, y=col2, color_discrete_sequence=_PALETTE)
    return _apply_base_layout(fig, f"'{col1}' vs '{col2}'")


# ---------------------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------------------

def generate_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Build an interactive Pearson correlation heatmap for all numerical columns.

    Returns *None* if fewer than 2 numerical columns exist.

    Parameters
    ----------
    df : Input DataFrame.

    Returns
    -------
    go.Figure | None
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr()
    cols = corr.columns.tolist()

    text = [[f"{v:.2f}" for v in row] for row in corr.values]

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=cols,
            y=cols,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
            hovertemplate="x: %{x}<br>y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        height=max(400, 80 * len(cols)),
    )
    return _apply_base_layout(fig, "Pearson Correlation Heatmap")


# ---------------------------------------------------------------------------
# 5. Distribution grid
# ---------------------------------------------------------------------------

def create_distribution_grid(df: pd.DataFrame) -> List[go.Figure]:
    """
    Create one distribution figure per column.

    * Numerical  → box plot
    * Categorical → pie chart (if ≤ 8 categories) else bar chart
    * Datetime   → monthly bar chart

    Parameters
    ----------
    df : Input DataFrame.

    Returns
    -------
    List[go.Figure]
        One figure per column; order follows ``df.columns``.
    """
    col_types = detect_column_types(df)
    figures: List[go.Figure] = []

    for col in df.columns:
        series = df[col].dropna()

        # numerical → box
        if col in col_types["numerical"]:
            fig = go.Figure(
                go.Box(
                    y=series,
                    name=col,
                    marker_color=_PALETTE[0],
                    boxmean="sd",
                    hovertemplate="Value: %{y}<extra></extra>",
                )
            )
            _apply_base_layout(fig, f"Box Plot – '{col}'")

        # categorical / boolean → pie or bar
        elif col in col_types["categorical"] or col in col_types["boolean"]:
            vc = series.value_counts()
            if vc.shape[0] <= 8:
                fig = go.Figure(
                    go.Pie(
                        labels=vc.index.astype(str),
                        values=vc.values,
                        hole=0.3,
                        marker=dict(colors=_PALETTE),
                        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
                    )
                )
            else:
                vc_df = vc.reset_index()
                vc_df.columns = ["value", "count"]
                fig = px.bar(
                    vc_df, x="value", y="count",
                    color="count", color_continuous_scale=_SEQ_PALETTE,
                )
                fig.update_layout(coloraxis_showscale=False)
            _apply_base_layout(fig, f"'{col}'")

        # datetime → monthly histogram
        elif col in col_types["datetime"]:
            ts = pd.to_datetime(series)
            monthly = ts.dt.to_period("M").value_counts().sort_index()
            fig = go.Figure(
                go.Bar(
                    x=[str(p) for p in monthly.index],
                    y=monthly.values,
                    marker_color=_PALETTE[3],
                    hovertemplate="Period: %{x}<br>Count: %{y}<extra></extra>",
                )
            )
            _apply_base_layout(fig, f"Monthly Distribution – '{col}'")

        else:
            # high-cardinality / unknown → top-10 bar
            vc = series.value_counts().head(10).reset_index()
            vc.columns = ["value", "count"]
            fig = px.bar(vc, x="value", y="count", color_discrete_sequence=_PALETTE)
            _apply_base_layout(fig, f"Top 10 – '{col}'")

        figures.append(fig)

    return figures


# ---------------------------------------------------------------------------
# 6. Statistical summary
# ---------------------------------------------------------------------------

def _statistical_summary(df: pd.DataFrame) -> Dict:
    """
    Return a rich statistical summary of the DataFrame.

    Includes:
    - Basic describe() for numerics
    - Value-count profiles for categoricals
    - Missing-value counts
    - Skewness and kurtosis for numerical columns
    """
    summary: Dict = {}
    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(exclude=[np.number])

    # numeric describe
    if not num_df.empty:
        desc = num_df.describe().round(4)
        desc.loc["skewness"] = num_df.skew().round(4)
        desc.loc["kurtosis"] = num_df.kurtosis().round(4)
        summary["numerical"] = desc.to_dict()

    # categorical profiles
    if not cat_df.empty:
        cat_profile: Dict = {}
        for col in cat_df.columns:
            vc = df[col].value_counts()
            cat_profile[col] = {
                "unique_values": int(df[col].nunique()),
                "top_value": str(vc.idxmax()) if not vc.empty else None,
                "top_count": int(vc.max()) if not vc.empty else 0,
                "value_counts": {str(k): int(v) for k, v in vc.head(20).items()},
            }
        summary["categorical"] = cat_profile

    # missing values
    missing = df.isnull().sum()
    summary["missing_values"] = {
        col: int(cnt) for col, cnt in missing.items() if cnt > 0
    }

    # shape
    summary["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    return summary


# ---------------------------------------------------------------------------
# 7. Main orchestration entry-point
# ---------------------------------------------------------------------------

def _get_dataset_metadata(df: pd.DataFrame) -> Dict:
    """Extract optimized metadata (schema, types, non-null counts, nunique, minimal samples) to minimize token usage."""
    col_types = detect_column_types(df)
    metadata = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": []
    }
    
    # We omit high cardinality columns to save tokens (we won't plot them anyway)
    high_card = set(col_types.get("high_cardinality", []))
    
    for col in df.columns:
        if col in high_card:
            continue
            
        non_null = int(df[col].notna().sum())
        nunique = int(df[col].nunique())
        
        col_meta = {
            "name": col,
            "inferred_type": "unknown",
            "non_null_count": non_null,
            "unique_values_count": nunique
        }
        
        # Identify type
        for t, cols in col_types.items():
            if col in cols:
                col_meta["inferred_type"] = t
                break
                
        # Send minimal samples only for categorical columns (to help LLM understand context)
        if col_meta["inferred_type"] == "categorical" and non_null > 0:
            sample_vals = df[col].dropna().head(2).tolist()
            col_meta["sample_values"] = [str(x)[:20] for x in sample_vals] # limit to 2 samples, max 20 chars
            
        if col_meta["inferred_type"] == "numerical" and non_null > 0:
            try:
                col_meta["min"] = float(df[col].min())
                col_meta["max"] = float(df[col].max())
                col_meta["mean"] = float(df[col].mean())
            except Exception:
                pass
                
        metadata["columns"].append(col_meta)
    return metadata


def _plan_charts_with_ai(metadata: Dict) -> List[Dict]:
    """Call the LLM to select the most business-relevant charts for this specific dataset."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("[AI Visualize Plan] GROQ_API_KEY is missing. Skipping AI visualization planning.")
        return []
        
    client = Groq(api_key=groq_api_key)
    prompt = f"""You are an expert business intelligence analyst.
Below is the metadata of a dataset:
Dataset shape: {metadata['num_rows']} rows x {metadata['num_cols']} columns

Column Details:
{json.dumps(metadata['columns'], indent=2)}

Identify the top 5 to 7 most critical business questions or insights this dataset can address (e.g., trends over time, category performance comparison, correlations between key numerical metrics, distributions of high-value fields, etc.).
For each question, define the best Plotly Express chart configuration to visualize it.

CRITICAL Formatting Rules for Chart Titles:
- If a chart is univariate (visualizing a single column), use a title like 'Distribution of Column Name'.
- If a chart is bivariate comparing numeric columns (scatter/line/box), use the format 'Column1 vs Column2' in the title.
- If a chart is grouping by category or datetime (bar/line/box), use the format 'Metric by Category' in the title.
This ensures they display correctly in the frontend's Univariate and Bivariate tabs.

Return ONLY a valid JSON array of chart configurations. No explanations, no markdown, no code blocks.
Output format:
[
  {{
    "title": "Business Chart Title (comply with formatting rules)",
    "description": "Short explanation of what business question this chart answers and what to look for",
    "chart_type": "line" | "bar" | "scatter" | "box" | "heatmap",
    "x": "column_name_for_x_axis",
    "y": "column_name_for_y_axis",
    "color": "column_name_for_color_grouping" (or null if not applicable)
  }}
]
"""
    try:
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
        )
        res_text = response.choices[0].message.content.strip()
        
        # Clean markdown code fences if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", res_text)
        if json_match:
            res_text = json_match.group(1).strip()
            
        array_match = re.search(r"\[[\s\S]*\]", res_text)
        if array_match:
            res_text = array_match.group(0)
            
        charts_plan = json.loads(res_text)
        return charts_plan
    except Exception as e:
        print(f"[AI Visualize Plan] AI chart planning failed: {e}")
        return []


def _generate_chart_from_plan(df: pd.DataFrame, plan: Dict) -> Optional[go.Figure]:
    """Execute a chart plan using Plotly Express."""
    chart_type = plan.get("chart_type")
    x = plan.get("x")
    y = plan.get("y")
    color = plan.get("color")
    title = plan.get("title", "")
    
    # Validation: columns must exist in df
    if x and x not in df.columns:
        return None
    if y and y not in df.columns:
        return None
    if color and color not in df.columns:
        color = None
        
    try:
        fig = None
        if chart_type == "heatmap":
            fig = generate_correlation_heatmap(df)
        elif chart_type == "line":
            df_plot = df
            if x:
                try:
                    df_plot = df.sort_values(by=x)
                except Exception:
                    pass
            fig = px.line(df_plot, x=x, y=y, color=color, title=title)
        elif chart_type == "bar":
            fig = px.bar(df, x=x, y=y, color=color, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, title=title)
        elif chart_type == "box":
            fig = px.box(df, x=x, y=y, color=color, title=title)
            
        if fig:
            fig = _apply_base_layout(fig, title)
            return fig
    except Exception as e:
        print(f"[AI Visualize Plan] Failed to render {chart_type} for {x} x {y}: {e}")
        return None


def generate_visualizations(df: pd.DataFrame) -> Dict:
    col_types = detect_column_types(df)

    # 1. Get metadata
    metadata = _get_dataset_metadata(df)
    
    # 2. Try planning charts with AI
    plans = _plan_charts_with_ai(metadata)
    
    univariate: List[go.Figure] = []
    bivariate: List[go.Figure] = []
    corr_fig: Optional[go.Figure] = None
    
    if plans:
        print(f"[INFO] Successfully planned {len(plans)} charts with AI.")
        for plan in plans:
            chart_type = plan.get("chart_type")
            title = plan.get("title", "")
            desc = plan.get("description", "")
            
            fig = _generate_chart_from_plan(df, plan)
            if fig:
                # Store the business description in Plotly figure layout's meta attribute
                fig.layout.meta = {"description": desc}
                
                if chart_type == "heatmap":
                    corr_fig = fig
                elif "vs" in title.lower() or "×" in title or " by " in title.lower():
                    bivariate.append(fig)
                else:
                    univariate.append(fig)
                    
    # 3. Fallback to rule-based generation if no charts were successfully generated
    if not univariate and not bivariate and corr_fig is None:
        print("[WARNING] AI visualization planning failed or returned no valid charts. Falling back to rule-based generation.")
        # --- univariate fallback ---
        for col in df.columns:
            if col in col_types["high_cardinality"]:
                continue
            try:
                fig = generate_univariate_plots(df, col, col_types)
                univariate.append(fig)
            except Exception as exc:
                print(f"[data_visualization] Skipping univariate plot for '{col}': {exc}")

        # --- bivariate fallback ---
        num_cols = col_types["numerical"][:6]
        cat_cols = col_types["categorical"][:3]

        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i + 1 :]:
                try:
                    bivariate.append(generate_bivariate_plots(df, c1, c2, col_types))
                except Exception as exc:
                    print(f"[data_visualization] Skipping bivariate '{c1}' × '{c2}': {exc}")

        for cat in cat_cols:
            for num in num_cols[:3]:
                try:
                    bivariate.append(generate_bivariate_plots(df, cat, num, col_types))
                except Exception as exc:
                    print(f"[data_visualization] Skipping bivariate '{cat}' × '{num}': {exc}")

        for dt in col_types["datetime"][:2]:
            for num in num_cols[:2]:
                try:
                    bivariate.append(generate_bivariate_plots(df, dt, num, col_types))
                except Exception as exc:
                    print(f"[data_visualization] Skipping datetime bivariate: {exc}")

        # --- correlation heatmap fallback ---
        corr_fig = generate_correlation_heatmap(df)

    stats = _statistical_summary(df)

    return {
        "column_types": col_types,
        "univariate_charts": univariate,
        "bivariate_charts": bivariate,
        "correlation_matrix": corr_fig,
        "statistical_summary": stats,
    }
