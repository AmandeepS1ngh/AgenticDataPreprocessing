"""
insights_agent.py
=================
LangGraph StateGraph agent that produces automated data insights using
the Groq LLaMA-3.3-70b model.

Pipeline (4 nodes):
  analyze_statistics  → detect_patterns → generate_insights → provide_recommendations

Falls back to rule-based insights if the LLM call fails (same pattern as ai_agent.py).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Groq client setup  (mirrors ai_agent.py pattern)
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

_groq_api_key = os.getenv("GROQ_API_KEY")
if not _groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Set it as an environment variable.")

_client = Groq(api_key=_groq_api_key)
MODEL       = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TEMPERATURE = 0.3
MAX_TOKENS  = 1500


# ─────────────────────────────────────────────────────────────────────────────
# State schema
# ─────────────────────────────────────────────────────────────────────────────
class InsightsState(BaseModel):
    """Shared state flowing through the LangGraph pipeline."""

    # ── inputs (populated before graph execution) ─────────────────────────
    df_json: str = ""                        # DataFrame serialised to JSON string
    viz_results: Dict[str, Any] = Field(default_factory=dict)

    # ── computed by nodes ──────────────────────────────────────────────────
    data_summary: Dict[str, Any]  = Field(default_factory=dict)
    correlations: List[Dict]      = Field(default_factory=list)
    outliers:     List[Dict]      = Field(default_factory=list)
    trends:       List[str]       = Field(default_factory=list)

    # ── LLM outputs ───────────────────────────────────────────────────────
    insights:        List[str] = Field(default_factory=list)
    patterns:        List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    executive_summary: str     = ""

    # ── error tracking ────────────────────────────────────────────────────
    llm_error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df_from_state(state: InsightsState) -> pd.DataFrame:
    return pd.DataFrame(json.loads(state.df_json))


def _compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce a compact statistical summary of the DataFrame."""
    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(exclude=[np.number])

    summary: Dict[str, Any] = {
        "shape":         {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "missing_total": int(df.isnull().sum().sum()),
        "duplicates":    int(df.duplicated().sum()),
    }

    if not num_df.empty:
        desc = num_df.describe().round(4)
        summary["numerical"] = desc.to_dict()
        summary["skewness"]  = num_df.skew().round(4).to_dict()
        summary["kurtosis"]  = num_df.kurtosis().round(4).to_dict()

    if not cat_df.empty:
        summary["categorical"] = {
            col: {
                "unique":    int(df[col].nunique()),
                "top_value": str(df[col].value_counts().idxmax()) if df[col].notna().any() else None,
                "top_count": int(df[col].value_counts().max())    if df[col].notna().any() else 0,
            }
            for col in cat_df.columns
        }

    return summary


def _compute_correlations(df: pd.DataFrame) -> List[Dict]:
    """Return all numeric column pairs with Pearson correlation ≥ 0.3."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return []

    corr_matrix = num_df.corr()
    pairs: List[Dict] = []
    cols = corr_matrix.columns.tolist()

    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = round(float(corr_matrix.loc[c1, c2]), 4)
            if abs(r) < 0.3:
                continue
            if abs(r) >= 0.7:
                strength = "strong positive" if r > 0 else "strong negative"
            elif abs(r) >= 0.5:
                strength = "moderate positive" if r > 0 else "moderate negative"
            else:
                strength = "weak positive" if r > 0 else "weak negative"
            pairs.append({"col1": c1, "col2": c2, "correlation": r, "strength": strength})

    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs[:15]   # cap at 15 most significant pairs


def _detect_outliers(df: pd.DataFrame) -> List[Dict]:
    """IQR-based outlier detection for numerical columns."""
    num_df = df.select_dtypes(include=[np.number])
    outlier_info: List[Dict] = []

    for col in num_df.columns:
        series = num_df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((series < lower) | (series > upper)).sum())
        if n_out > 0:
            outlier_info.append({
                "column":          col,
                "outlier_count":   n_out,
                "outlier_pct":     round(n_out / len(series) * 100, 2),
                "lower_bound":     round(float(lower), 4),
                "upper_bound":     round(float(upper), 4),
            })

    return sorted(outlier_info, key=lambda x: x["outlier_pct"], reverse=True)


def _detect_trends(df: pd.DataFrame, summary: Dict) -> List[str]:
    """Rule-based trend detection (used both standalone and as LLM context)."""
    trends: List[str] = []
    num_df = df.select_dtypes(include=[np.number])

    for col in num_df.columns:
        series = num_df[col].dropna()
        if len(series) < 3:
            continue
        skew = float(series.skew())
        if skew > 1:
            trends.append(f"'{col}' is heavily right-skewed (skew={skew:.2f}), indicating a long tail of high values.")
        elif skew < -1:
            trends.append(f"'{col}' is heavily left-skewed (skew={skew:.2f}), indicating a concentration of high values.")

    missing_pct = summary.get("missing_total", 0) / max(df.size, 1) * 100
    if missing_pct > 5:
        trends.append(f"Dataset has {missing_pct:.1f}% missing values — imputation strategy significantly impacts results.")

    return trends


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback insights
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_insights(state: InsightsState) -> InsightsState:
    """Generate basic insights without the LLM (used when Groq fails)."""
    df = _df_from_state(state)
    s  = state.data_summary

    insights: List[str] = []
    patterns: List[str] = []
    recs:     List[str] = []

    # Shape insight
    r, c = s.get("shape", {}).get("rows", 0), s.get("shape", {}).get("columns", 0)
    insights.append(f"The dataset contains {r:,} rows and {c} columns.")

    # Missing values
    missing = s.get("missing_total", 0)
    if missing:
        insights.append(f"{missing:,} missing values were detected and handled during cleaning.")

    # Correlations
    if state.correlations:
        top = state.correlations[0]
        insights.append(
            f"Strongest correlation: '{top['col1']}' and '{top['col2']}' "
            f"({top['correlation']:+.2f}, {top['strength']})."
        )

    # Outliers
    if state.outliers:
        worst = state.outliers[0]
        insights.append(
            f"'{worst['column']}' has the most outliers ({worst['outlier_pct']}% of values "
            f"fall outside [{worst['lower_bound']:.2f}, {worst['upper_bound']:.2f}])."
        )

    # Numerical stats insight
    num_stats = s.get("numerical", {})
    for col, stats in num_stats.items():
        mean_val = stats.get("mean", 0)
        std_val  = stats.get("std", 0)
        if mean_val and std_val and (std_val / abs(mean_val)) > 1:
            patterns.append(f"'{col}' shows high variability (CV = {std_val/abs(mean_val):.2f}).")
        break  # one example is enough

    patterns += state.trends[:3]

    recs.append("Investigate outlier rows manually before using the data for modelling.")
    if state.correlations:
        recs.append(
            f"Consider using '{state.correlations[0]['col1']}' and "
            f"'{state.correlations[0]['col2']}' as features in a predictive model."
        )
    recs.append("Apply feature scaling (StandardScaler / MinMaxScaler) before ML training.")

    exec_summary = (
        f"This dataset ({r:,} rows × {c} cols) has been cleaned and analysed. "
        f"{len(state.correlations)} significant correlations and "
        f"{len(state.outliers)} columns with outliers were identified. "
        "See detailed insights and recommendations above."
    )

    return state.model_copy(update={
        "insights":          insights,
        "patterns":          patterns,
        "recommendations":   recs,
        "executive_summary": exec_summary,
    })


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node functions
# ─────────────────────────────────────────────────────────────────────────────

def node_analyze_statistics(state: InsightsState) -> InsightsState:
    """Node 1 — compute descriptive statistics and store in state."""
    df      = _df_from_state(state)
    summary = _compute_summary(df)
    return state.model_copy(update={"data_summary": summary})


def node_detect_patterns(state: InsightsState) -> InsightsState:
    """Node 2 — compute correlations, outliers, and rule-based trends."""
    df           = _df_from_state(state)
    correlations = _compute_correlations(df)
    outliers     = _detect_outliers(df)
    trends       = _detect_trends(df, state.data_summary)
    return state.model_copy(update={
        "correlations": correlations,
        "outliers":     outliers,
        "trends":       trends,
    })


def _get_business_context(df: pd.DataFrame) -> str:
    """Extract categorical and seasonal features from df to provide business context for the LLM."""
    context = []
    
    # 1. Categorical columns top categories (limit to top 3 columns, top 3 values, and truncate long names to save tokens)
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        context.append("Categorical Columns Top Values:")
        for col in cat_cols[:3]: # limit to top 3 categorical columns
            try:
                vc = df[col].value_counts().head(3) # limit to top 3 values
                vc_str = ", ".join([f"'{str(k)[:20]}': {v} records" for k, v in vc.items()])
                context.append(f"  • {col}: {vc_str}")
            except Exception:
                pass
                
    # 2. Seasonality / time trends
    date_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_col = col
                break
            except Exception:
                pass
                
    if date_col:
        val_col = None
        for col in df.columns:
            if col.lower() in ["sales", "revenue", "amount", "price", "quantity", "quantityordered"]:
                val_col = col
                break
        if not val_col:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                val_col = num_cols[0]
                
        if val_col:
            try:
                monthly = df.groupby(df[date_col].dt.month)[val_col].sum()
                month_names = {
                    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
                }
                monthly_str = ", ".join([f"{month_names.get(m, m)}: {v:,.2f}" for m, v in monthly.items()])
                context.append(f"Seasonal Trends ({val_col} summed by Month):")
                context.append(f"  {monthly_str}")
            except Exception:
                pass

    return "\n".join(context)


def node_generate_insights(state: InsightsState) -> InsightsState:
    """Node 3 — call Groq LLM to generate business-focused data insights."""
    df = _df_from_state(state)
    business_context = _get_business_context(df)

    # Build a compact but rich context for the LLM
    num_stats = state.data_summary.get("numerical", {})

    # Summarise numerical stats concisely (mean ± std for each column)
    stats_lines = []
    for col, stats in num_stats.items():
        stats_lines.append(
            f"  {col}: mean={stats.get('mean','?'):.4g}, std={stats.get('std','?'):.4g}, "
            f"min={stats.get('min','?'):.4g}, max={stats.get('max','?'):.4g}"
        )

    corr_lines = [
        f"  {c['col1']} ↔ {c['col2']}: r={c['correlation']:+.2f} ({c['strength']})"
        for c in state.correlations[:8]
    ]
    outlier_lines = [
        f"  {o['column']}: {o['outlier_count']} outliers ({o['outlier_pct']}%)"
        for o in state.outliers[:6]
    ]
    trend_lines = [f"  • {t}" for t in state.trends[:5]]

    # Extract details of planned charts to give LLM context on charts generated/displayed
    chart_lines = []
    if state.viz_results and "charts" in state.viz_results:
        for idx, c in enumerate(state.viz_results["charts"], 1):
            title = c.get("title", f"Chart {idx}")
            desc = c.get("description", "")
            chart_lines.append(f"  • Chart {idx}: '{title}' - {desc}")
    charts_context = "\n".join(chart_lines) if chart_lines else "  No visual charts generated."

    shape = state.data_summary.get("shape", {})
    prompt = f"""You are a senior business intelligence consultant and data strategist. Analyze this dataset from a high-level business perspective.
Focus on identifying operational bottlenecks, high-value opportunities, seasonal customer behaviors, product/category performance anomalies, and concrete strategic recommendations.
Your insights and recommendations should think like a business advisor (e.g. identify seasonal spikes like sales peaking in March and advise to increase stock/marketing in Feb, identify low-performing segments and suggest next actions).

The following visual charts were planned and generated for the user:
{charts_context}

Integrate observations from these specific charts into your insights and analysis where appropriate, explaining what the chart shows and what business value/recommendation follows from it.

Dataset Shape: {shape.get('rows', '?')} rows × {shape.get('columns', '?')} columns
Missing values: {state.data_summary.get('missing_total', 0)}

Summary Statistics:
{chr(10).join(stats_lines) if stats_lines else "  No numerical columns."}

Correlations (|r| ≥ 0.3):
{chr(10).join(corr_lines) if corr_lines else "  None detected."}

Outliers Detected:
{chr(10).join(outlier_lines) if outlier_lines else "  None detected."}

Rule-based Trends:
{chr(10).join(trend_lines) if trend_lines else "  None detected."}

Business and Seasonal Context:
{business_context if business_context else "  No categorical or seasonal context available."}

Provide a comprehensive analysis in the following EXACT JSON format. Return ONLY valid JSON, no markdown, no explanations.
Reference actual numbers, dates, categories, and columns from the dataset to make your recommendations concrete, specific, and authoritative.

{{
  "insights": [
    "Specific business insight 1 referencing actual numbers/categories",
    "Specific business insight 2 referencing actual numbers/categories",
    "Specific business insight 3 referencing actual numbers/categories",
    "Specific business insight 4 referencing actual numbers/categories",
    "Specific business insight 5 referencing actual numbers/categories"
  ],
  "patterns": [
    "Pattern or trend 1 (e.g. seasonality details)",
    "Pattern or trend 2",
    "Pattern or trend 3"
  ],
  "recommendations": [
    "Actionable business recommendation 1 (e.g. marketing/inventory decisions)",
    "Actionable business recommendation 2",
    "Actionable business recommendation 3"
  ],
  "executive_summary": "One paragraph executive summary of the dataset and key strategic findings."
}}"""

    try:
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if json_match:
            raw = json_match.group(1).strip()

        parsed = json.loads(raw)

        return state.model_copy(update={
            "insights":          parsed.get("insights", []),
            "patterns":          parsed.get("patterns", []),
            "recommendations":   parsed.get("recommendations", []),
            "executive_summary": parsed.get("executive_summary", ""),
        })

    except Exception as exc:
        # Graceful fallback — mark error, let next node handle it
        return state.model_copy(update={"llm_error": str(exc)})


def node_provide_recommendations(state: InsightsState) -> InsightsState:
    """Node 4 — if LLM failed, fill in rule-based fallback; otherwise enrich."""
    if state.llm_error or not state.insights:
        # Fallback path
        state = _rule_based_insights(state)
    else:
        # LLM succeeded — optionally append any missing rule-based recs
        if len(state.recommendations) < 3:
            extra = [
                "Investigate outlier rows manually before using the data for modelling.",
                "Apply feature scaling before ML training.",
                "Consider dimensionality reduction if the feature count is large.",
            ]
            merged = list(state.recommendations)
            for r in extra:
                if len(merged) >= 5:
                    break
                merged.append(r)
            state = state.model_copy(update={"recommendations": merged})

    return state


# ─────────────────────────────────────────────────────────────────────────────
# InsightsAgent  (public API)
# ─────────────────────────────────────────────────────────────────────────────

class InsightsAgent:
    """
    Wraps the LangGraph pipeline and exposes a simple ``analyze`` method.

    Usage::

        agent = InsightsAgent()
        result = agent.analyze(df, viz_results={})
    """

    def __init__(self) -> None:
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(InsightsState)

        graph.add_node("analyze_statistics",     node_analyze_statistics)
        graph.add_node("detect_patterns",        node_detect_patterns)
        graph.add_node("generate_insights",      node_generate_insights)
        graph.add_node("provide_recommendations",node_provide_recommendations)

        graph.set_entry_point("analyze_statistics")
        graph.add_edge("analyze_statistics",      "detect_patterns")
        graph.add_edge("detect_patterns",         "generate_insights")
        graph.add_edge("generate_insights",       "provide_recommendations")
        graph.add_edge("provide_recommendations", END)

        return graph.compile()

    # ------------------------------------------------------------------
    def analyze(
        self,
        df: pd.DataFrame,
        viz_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full insights pipeline on a cleaned DataFrame.

        Parameters
        ----------
        df          : Cleaned pandas DataFrame.
        viz_results : Optional dict from /visualize/ endpoint (extra context).

        Returns
        -------
        dict with keys: insights, patterns, recommendations, executive_summary,
                        correlations, outliers, data_summary, llm_error.
        """
        initial_state = InsightsState(
            df_json     = df.to_json(orient="records"),
            viz_results = viz_results or {},
        )

        final = self._graph.invoke(initial_state)

        # LangGraph may return a dict or the Pydantic object
        if isinstance(final, dict):
            final = InsightsState(**final)

        return {
            "insights":          final.insights,
            "patterns":          final.patterns,
            "recommendations":   final.recommendations,
            "executive_summary": final.executive_summary,
            "correlations":      final.correlations,
            "outliers":          final.outliers,
            "data_summary":      final.data_summary,
            "llm_used":          final.llm_error is None,
            "llm_error":         final.llm_error,
        }
