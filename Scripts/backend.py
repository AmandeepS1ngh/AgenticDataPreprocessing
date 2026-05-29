import sys
import os
import pandas as pd
import io
import re
import base64
from typing import Optional, List
import aiohttp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from sqlalchemy import create_engine
from pydantic import BaseModel
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ai_agent import AIAgent
from data_cleaning import DataCleaning
from data_ingestion import DataIngestion
from data_visualization import generate_visualizations, detect_column_types
from insights_agent import InsightsAgent


def parse_ai_response(response_text: str) -> pd.DataFrame:
    """Parse AI response, handling markdown code blocks and JSON extraction."""
    if isinstance(response_text, pd.DataFrame):
        return response_text
    
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1).strip()
    
    # Try to find JSON array in the text
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        text = array_match.group(0)
    
    try:
        data = json.loads(text)
        return pd.DataFrame(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse AI response as JSON: {e}")


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """
    Read CSV bytes trying multiple encodings so that files saved by Excel
    on Windows (cp1252 / latin-1) never raise UnicodeDecodeError.
    Encoding order: utf-8 → utf-8-sig (BOM) → latin-1 → cp1252.
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    # Absolute fallback – replace undecodable chars rather than crash
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1", errors="replace")



import math

def _make_json_compatible(obj):
    """
    Recursively clean python objects (dicts, lists, floats) to be JSON compliant.
    Converts float('nan'), float('inf'), float('-inf'), pd.NA, and NaT to None.
    """
    if isinstance(obj, dict):
        return {k: _make_json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_compatible(x) for x in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


app = FastAPI(title="AI Data Preprocessing API", version="1.0.0")

ai_agent       = AIAgent()
cleaner        = DataCleaning()
insights_agent = InsightsAgent()

# Skip AI row-by-row cleaning for files larger than this (use pandas-only instead).
# LLM is called once for schema analysis regardless; it's the batched row calls that are slow.
AI_ROW_LIMIT = 500

# -------------------------------
# CSV/EXCEL CLEANING MACHINE ENDPOINT
# -------------------------------
@app.post("/cleandata/")
async def clean_data(
    file: UploadFile = File(...),
    normalize: bool = Form(False)
):
    try:
        contents = await file.read()
        file_extension = file.filename.split(".")[-1].lower()

        if file_extension == "csv":
            df = _read_csv_bytes(contents)
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or Excel file.")
        
        # Step 1: Apply traditional DataCleaning (this always works)
        df_cleaned = cleaner.clean_data(df, normalize=normalize)
        
        # Step 2: Try AI enhancement (optional - graceful fallback if it fails)
        ai_enhanced = False
        ai_error_message = None

        if len(df_cleaned) > AI_ROW_LIMIT:
            # For large files, skip row-by-row LLM batching — it takes too long.
            # Traditional cleaning (Step 1) is already applied above.
            print(f"[INFO] File has {len(df_cleaned)} rows (> {AI_ROW_LIMIT} limit). "
                  "Applied traditional pandas data cleaning; skipped row-by-row LLM batch formatting to prevent API timeout.")
        else:
            try:
                df_ai_cleaned = ai_agent.process_data(df_cleaned)
                print("\n--- AI Agent Raw Output ---\n", df_ai_cleaned, "\n-------------------------\n")

                if isinstance(df_ai_cleaned, str):
                    if df_ai_cleaned.startswith("ERROR:"):
                        ai_error_message = df_ai_cleaned
                        print(f"AI Agent returned error, using traditional cleaning: {ai_error_message}")
                    else:
                        df_ai_cleaned = parse_ai_response(df_ai_cleaned)
                        df_cleaned = df_ai_cleaned
                        ai_enhanced = True
                else:
                    df_cleaned = df_ai_cleaned
                    ai_enhanced = True

            except Exception as ai_error:
                ai_error_message = str(ai_error)
                print(f"AI Agent failed, falling back to traditional cleaning: {ai_error_message}")
        
        # Return the best available result
        return _make_json_compatible({
            "cleaned_data": df_cleaned.to_dict(orient="records"),
            "ai_enhanced": ai_enhanced,
            "message": "Data cleaned successfully" + (" with AI enhancement" if ai_enhanced else " (AI unavailable, used traditional methods)")
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------
# HELPERS – figure serialisation
# -----------------------------------------------------------------------
def _fig_to_html(fig) -> str:
    """Convert a Plotly figure to a self-contained HTML snippet (no CDN)."""
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _fig_to_base64_png(fig, include_png: bool = False) -> Optional[str]:
    """Convert a Plotly figure to a Base64-encoded PNG string (requires kaleido).
    
    Skipped by default (include_png=False) because kaleido starts a headless
    Chromium process per chart, making bulk visualisation very slow.
    Users can still download PNGs via the Plotly toolbar camera icon.
    """
    if not include_png:
        return None
    try:
        img_bytes = fig.to_image(format="png", width=800, height=500, scale=1.5)
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return None  # kaleido not available or other error – degrade gracefully


def _build_chart_list(result: dict, include_png: bool = False) -> List[dict]:
    """
    Flatten univariate + bivariate + correlation figures into the
    standardised chart record format::

        {
          "type":         str,   # e.g. "histogram", "scatter", "correlation_heatmap"
          "column":       str,   # primary column (empty for multi-column charts)
          "title":        str,
          "html":         str,   # Plotly HTML snippet (always present)
          "image_base64": str | None,  # only when include_png=True
          "description":  str,   # business explanation from AI planning
        }
    """
    charts: List[dict] = []

    for fig in result["univariate_charts"]:
        title = fig.layout.title.text or ""
        trace_type = fig.data[0].type if fig.data else "chart"
        description = fig.layout.meta.get("description") if fig.layout.meta and isinstance(fig.layout.meta, dict) else ""
        charts.append({
            "type": trace_type,
            "column": title,
            "title": title,
            "html": _fig_to_html(fig),
            "image_base64": _fig_to_base64_png(fig, include_png),
            "description": description,
        })

    for fig in result["bivariate_charts"]:
        title = fig.layout.title.text or ""
        trace_type = fig.data[0].type if fig.data else "chart"
        description = fig.layout.meta.get("description") if fig.layout.meta and isinstance(fig.layout.meta, dict) else ""
        charts.append({
            "type": trace_type,
            "column": title,
            "title": title,
            "html": _fig_to_html(fig),
            "image_base64": _fig_to_base64_png(fig, include_png),
            "description": description,
        })

    if result["correlation_matrix"] is not None:
        fig = result["correlation_matrix"]
        description = fig.layout.meta.get("description") if fig.layout.meta and isinstance(fig.layout.meta, dict) else ""
        charts.append({
            "type": "correlation_heatmap",
            "column": "",
            "title": fig.layout.title.text or "Correlation Heatmap",
            "html": _fig_to_html(fig),
            "image_base64": _fig_to_base64_png(fig, include_png),
            "description": description,
        })

    return charts


def _df_from_file(contents: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded file bytes into a DataFrame with encoding auto-detection."""
    ext = filename.split(".")[-1].lower()
    if ext == "csv":
        return _read_csv_bytes(contents)
    elif ext in ("xls", "xlsx"):
        return pd.read_excel(io.BytesIO(contents))
    raise HTTPException(status_code=400, detail=f"Unsupported file type '.{ext}'. Upload a CSV or Excel file.")


# -----------------------------------------------------------------------
# VISUALIZATION ENDPOINT
# Two input modes:
#   1. file upload  (raw CSV/Excel  →  clean on-the-fly  →  visualise)
#   2. cleaned_data (JSON records from /cleandata/ response)
#
# Query param:
#   include_png=true  → also generate Base64 PNG per chart (slow, uses kaleido)
#   include_png=false → HTML only, fast (default)
# -----------------------------------------------------------------------
@app.post("/visualize/")
async def visualize_data(
    file: Optional[UploadFile] = File(default=None),
    cleaned_data: Optional[str] = Form(default=None),
    include_png: bool = False,          # query param: ?include_png=true for PNG exports
):
    """
    Generate interactive visualisations from a dataset.

    **Input – choose one:**
    - `file`         : Raw CSV / Excel upload (will be cleaned before visualising)
    - `cleaned_data` : JSON string of records from a previous `/cleandata/` response

    **Query params:**
    - `include_png=false` *(default)* – HTML only, very fast
    - `include_png=true`  – also returns Base64 PNG per chart (slow, ~3-5s each)

    **Returns:**
    ```json
    {
      "charts": [
        {
          "type": "histogram",
          "column": "age",
          "title": "Distribution of 'age'",
          "html": "<div>...</div>",
          "image_base64": null
        },
        ...
      ],
      "summary_stats": { ... },
      "column_types":  { ... }
    }
    ```
    """
    try:
        # ── Resolve the DataFrame ───────────────────────────────────────
        if cleaned_data:
            # Mode 1: cleaned JSON records passed directly
            try:
                records = json.loads(cleaned_data)
                df = pd.DataFrame(records)
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=422, detail=f"Invalid cleaned_data JSON: {e}")

        elif file and file.filename:
            # Mode 2: raw file upload – clean first, then visualise
            contents = await file.read()
            df_raw = _df_from_file(contents, file.filename)
            df = cleaner.clean_data(df_raw)

        else:
            raise HTTPException(
                status_code=422,
                detail="Provide either a 'file' upload or 'cleaned_data' JSON form field.",
            )

        if df.empty:
            raise HTTPException(status_code=422, detail="The dataset is empty after cleaning.")

        # ── Run visualization pipeline ──────────────────────────────────
        result = generate_visualizations(df)

        # include_png=False by default → kaleido is NOT invoked → fast response
        charts = _build_chart_list(result, include_png=include_png)

        return _make_json_compatible({
            "charts": charts,
            "summary_stats": result["statistical_summary"],
            "column_types": result["column_types"],
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------
# AI INSIGHTS ENDPOINT
# -----------------------------------------------------------------------
class InsightsRequest(BaseModel):
    cleaned_data:         List[dict]       # list of records from /cleandata/
    visualization_results: Optional[dict] = None  # from /visualize/ (extra context)


@app.post("/generate-insights/")
async def generate_insights(request: InsightsRequest):
    """
    Run the LangGraph InsightsAgent on a cleaned dataset and return
    AI-generated insights, correlations, outliers, and recommendations.

    **Input:**
    ```json
    {
      "cleaned_data": [{...}, ...],
      "visualization_results": { ... }   // optional, from /visualize/
    }
    ```

    **Returns:**
    ```json
    {
      "ai_insights":          ["..."],
      "patterns":             ["..."],
      "recommendations":      ["..."],
      "executive_summary":    "...",
      "statistical_summary":  { "total_rows": ..., "total_columns": ..., ... },
      "correlations":         [{"col1": ..., "col2": ..., "correlation": ..., "strength": ...}],
      "outliers":             [{"column": ..., "outlier_count": ..., "outlier_pct": ...}],
      "ai_powered":           true | false
    }
    ```
    """
    try:
        if not request.cleaned_data:
            raise HTTPException(status_code=422, detail="cleaned_data must not be empty.")

        # Reconstruct DataFrame from JSON records
        import asyncio
        df = pd.DataFrame(request.cleaned_data)

        # Run the LangGraph pipeline in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: insights_agent.analyze(df, request.visualization_results or {}),
        )

        shape = result["data_summary"].get("shape", {})

        return _make_json_compatible({
            "ai_insights":       result["insights"],
            "patterns":          result["patterns"],
            "recommendations":   result["recommendations"],
            "executive_summary": result["executive_summary"],
            "statistical_summary": {
                "total_rows":             shape.get("rows", len(df)),
                "total_columns":          shape.get("columns", len(df.columns)),
                "missing_values_handled": result["data_summary"].get("missing_total", 0),
                "duplicates_removed":     result["data_summary"].get("duplicates", 0),
                "outliers_detected":      sum(o["outlier_count"] for o in result["outliers"]),
            },
            "correlations":  result["correlations"],
            "outliers":      result["outliers"],
            "ai_powered":    result["llm_used"],
            "llm_error":     result.get("llm_error"),   # None if LLM succeeded
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# DATABASE CLEANING MACHINE ENDPOINT
# -------------------------------
class DBQuery(BaseModel):
    db_url: str
    query: str
    normalize: bool = False

@app.post("/clean-db/")
async def clean_db(query: DBQuery):
    """Fetch data from DB, clean it using AI, and return JSON"""
    try:
        engine = create_engine(query.db_url)
        df = pd.read_sql(query.query, engine)

        df_cleaned = cleaner.clean_data(df, normalize=query.normalize)
        df_ai_cleaned = ai_agent.process_data(df_cleaned)

        print("\n--- AI Agent Raw Output (DB) ---\n", df_ai_cleaned, "\n-------------------------\n")

        if isinstance(df_ai_cleaned, str):
            try:
                df_ai_cleaned = parse_ai_response(df_ai_cleaned)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Invalid AI response format: {str(e)}")

        return _make_json_compatible({"cleaned_data": df_ai_cleaned.to_dict(orient="records")})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# API CLEANING MACHINE ENDPOINT
# -------------------------------
class APIRequest(BaseModel):
    api_url: str
    normalize: bool = False

@app.post("/clean-api/")
async def clean_api(request: APIRequest):
    """Fetch API data, clean it using AI, and return JSON"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(request.api_url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to fetch data from API. Status code: {response.status}")
                
                data = await response.json()
                df = pd.DataFrame(data)

                df_cleaned = cleaner.clean_data(df, normalize=request.normalize)
                df_ai_cleaned = ai_agent.process_data(df_cleaned)

                try:
                    df_ai_cleaned = pd.DataFrame(json.loads(df_ai_cleaned))
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Invalid AI response format: {str(e)}")

                return _make_json_compatible({"cleaned_data": df_ai_cleaned.to_dict(orient="records")})
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
