"""
app.py — AI-Powered Data Preprocessing · Production-Ready Streamlit UI
Tabbed interface: Data Upload · Preprocessing · Visualisations · AI Insights · Export
"""

import io
import json
import zipfile

import pandas as pd
import requests
import streamlit as st

FASTAPI_URL = "http://127.0.0.1:8000"

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Preprocessor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; }

/* ── Hero ──────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81; border-radius: 16px;
    padding: 2rem 2.4rem; margin-bottom: 1.6rem;
}
.hero-title {
    font-size: 2rem; font-weight: 700; margin: 0;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { font-size: .9rem; color: #64748b; margin: .3rem 0 0; }

/* ── KPI cards ─────────────────────────────────────────────────────── */
.kpi {
    background: linear-gradient(145deg, #161b27, #1c2233);
    border: 1px solid #2d3551; border-radius: 12px;
    padding: 1rem 1.2rem; text-align: center; height: 100%;
}
.kpi-label { font-size: .68rem; color: #475569; text-transform: uppercase; letter-spacing: .07em; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #e2e8f0; line-height: 1.2; }
.kpi-delta { font-size: .72rem; margin-top: .15rem; }
.kpi-delta.good  { color: #4ade80; }
.kpi-delta.warn  { color: #fb923c; }

/* ── Section headers ───────────────────────────────────────────────── */
.sec-hdr {
    font-size: 1rem; font-weight: 600; color: #c4b5fd;
    border-left: 3px solid #7c3aed; padding-left: .6rem;
    margin: 1.2rem 0 .7rem;
}

/* ── Status badges ─────────────────────────────────────────────────── */
.badge-ai   { background:#14532d; color:#4ade80; border-radius:99px; padding:3px 12px; font-size:.78rem; font-weight:600; }
.badge-trad { background:#1c1917; color:#fb923c; border-radius:99px; padding:3px 12px; font-size:.78rem; font-weight:600; }
.badge-ready{ background:#1e3a5f; color:#60a5fa; border-radius:99px; padding:3px 12px; font-size:.78rem; font-weight:600; }

/* ── Comparison diff boxes ─────────────────────────────────────────── */
.diff-box {
    background: #161b27; border: 1px solid #252d45;
    border-radius: 10px; padding: 1rem 1.2rem; margin: .4rem 0;
}
.diff-label { font-size: .7rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
.diff-before { font-size: 1.3rem; font-weight: 700; color: #f87171; }
.diff-after  { font-size: 1.3rem; font-weight: 700; color: #4ade80; }
.diff-arrow  { font-size: 1rem; color: #94a3b8; margin: 0 .4rem; }

/* ── Insight cards ─────────────────────────────────────────────────── */
.insight-card {
    background: #1a2236; border-left: 3px solid #6366f1;
    border-radius: 8px; padding: .75rem 1rem; margin: .45rem 0;
    color: #e2e8f0; font-size: .9rem; line-height: 1.55;
}
.pattern-card {
    background: #1a2330; border-left: 3px solid #10b981;
    border-radius: 8px; padding: .75rem 1rem; margin: .45rem 0;
    color: #d1fae5; font-size: .9rem;
}
.rec-card {
    background: #1e2820; border-left: 3px solid #f59e0b;
    border-radius: 8px; padding: .75rem 1rem; margin: .45rem 0;
    color: #fef3c7; font-size: .9rem;
}

/* ── Chart card ────────────────────────────────────────────────────── */
.chart-card {
    background: #161b27; border: 1px solid #252d45;
    border-radius: 12px; padding: .5rem; margin-bottom: .8rem;
}

/* ── Tab styling ───────────────────────────────────────────────────── */
div[data-testid="stTabs"] button { font-weight: 600; font-size: .9rem; }

/* ── Executive summary ─────────────────────────────────────────────── */
.exec-card {
    background: linear-gradient(135deg, #1e2a3a, #1a2336);
    border: 1px solid #2d4a6b; border-radius: 14px;
    padding: 1.4rem 1.8rem; margin: .8rem 0 1.2rem;
}
.exec-label { color: #94a3b8; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .4rem; }
.exec-text  { color: #e2e8f0; font-size: .97rem; line-height: 1.65; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_defaults: dict = {
    "raw_df":          None,   # original uploaded DataFrame
    "raw_stats":       {},     # {missing, dupes, shape}
    "cleaned_json":    None,   # list[dict] from /cleandata/
    "cleaned_df":      None,   # DataFrame version of cleaned_json
    "ai_enhanced":     False,
    "viz_result":      None,   # /visualize/ response
    "insights_result": None,   # /generate-insights/ response
    "data_source":     None,   # "file" | "db" | "api"
    "source_name":     "",     # file name / db table / api url
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">🤖 AI Data Preprocessor</p>
  <p class="hero-sub">Upload · Clean · Visualise · Analyse — powered by LangGraph + Groq LLaMA-3.3</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(col_el, label: str, value: str, delta: str = "", delta_good: bool = True):
    cls = "good" if delta_good else "warn"
    delta_html = f'<div class="kpi-delta {cls}">{delta}</div>' if delta else ""
    col_el.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )


def _diff_metric(col_el, label: str, before, after):
    col_el.markdown(
        f'<div class="diff-box"><div class="diff-label">{label}</div>'
        f'<span class="diff-before">{before}</span>'
        f'<span class="diff-arrow">→</span>'
        f'<span class="diff-after">{after}</span></div>',
        unsafe_allow_html=True,
    )


def _raw_stats(df: pd.DataFrame) -> dict:
    return {
        "rows":    int(df.shape[0]),
        "cols":    int(df.shape[1]),
        "missing": int(df.isnull().sum().sum()),
        "dupes":   int(df.duplicated().sum()),
    }


def _read_file(uploaded_file) -> pd.DataFrame:
    """
    Safely read a CSV or Excel upload, trying multiple encodings for CSV files.
    Handles UTF-8, latin-1 (Windows-1252), and cp1252 transparently so that
    files exported from Excel on Windows never throw UnicodeDecodeError.
    """
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext != "csv":
        return pd.read_excel(uploaded_file)

    # Try encodings in order: utf-8 → utf-8-sig (BOM) → latin-1 → cp1252
    raw_bytes = uploaded_file.read()
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
        except (UnicodeDecodeError, Exception):
            continue
    # Last resort — ignore undecodable bytes
    return pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1", errors="replace")


def _safe_error(resp: requests.Response) -> str:
    """
    Safely extract an error message from any response.
    Never raises JSONDecodeError even if the body is empty or not JSON.
    """
    # Try JSON first (FastAPI returns {"detail": "..."} for HTTPExceptions)
    try:
        body = resp.json()
        return body.get("detail", str(body))
    except Exception:
        pass
    # Fall back to raw text, or a generic HTTP status message
    return resp.text.strip() or f"HTTP {resp.status_code} — server returned an empty response."


def _call_clean(file_bytes: bytes, filename: str, normalize: bool = False) -> bool:
    with st.spinner("🧹 Running AI cleaning pipeline…"):
        resp = requests.post(
            f"{FASTAPI_URL}/cleandata/",
            files={"file": (filename, file_bytes)},
            data={"normalize": str(normalize).lower()},
            timeout=300,
        )
    if resp.status_code == 200:
        r = resp.json()
        records = r.get("cleaned_data", [])
        st.session_state.cleaned_json = records
        st.session_state.cleaned_df   = pd.DataFrame(records)
        st.session_state.ai_enhanced  = r.get("ai_enhanced", False)
        return True
    st.error(f"❌ Cleaning failed: {_safe_error(resp)}")
    return False


def _call_visualize(cleaned_json: list, include_png: bool = False) -> bool:
    with st.spinner("📊 Generating interactive charts…"):
        resp = requests.post(
            f"{FASTAPI_URL}/visualize/",
            data={"cleaned_data": json.dumps(cleaned_json)},
            params={"include_png": "true" if include_png else "false"},
            timeout=180,
        )
    if resp.status_code == 200:
        st.session_state.viz_result = resp.json()
        return True
    st.error(f"❌ Visualisation failed: {_safe_error(resp)}")
    return False


def _call_insights(cleaned_json: list, viz_result: dict | None) -> bool:
    with st.spinner("🧠 Running LangGraph insights pipeline…"):
        resp = requests.post(
            f"{FASTAPI_URL}/generate-insights/",
            json={"cleaned_data": cleaned_json, "visualization_results": viz_result or {}},
            timeout=120,
        )
    if resp.status_code == 200:
        st.session_state.insights_result = resp.json()
        return True
    st.error(f"❌ Insights failed: {_safe_error(resp)}")
    return False


def _render_chart_grid(charts: list, cols_per_row: int = 2, filter_titles: list | None = None):
    visible = [c for c in charts if not filter_titles or c.get("title", "") in filter_titles]
    if not visible:
        st.info("No charts match the current filter.")
        return
    rows = [visible[i: i + cols_per_row] for i in range(0, len(visible), cols_per_row)]
    for row in rows:
        grid = st.columns(len(row))
        for col_el, chart in zip(grid, row):
            with col_el:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.components.v1.html(chart["html"], height=420, scrolling=False)
                if chart.get("description"):
                    st.info(f"💡 **Business Value:** {chart['description']}")
                else:
                    st.caption("💡 Hover the chart → use the 📷 toolbar icon to download PNG")
                st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PDF report generator (fpdf2)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_pdf(
    source_name: str,
    raw_stats: dict,
    cleaned_df: pd.DataFrame,
    insights_result: dict | None,
    viz_result: dict | None,
) -> bytes:
    try:
        from fpdf import FPDF
        import base64
        import io

        def _clean_pdf_text(text: str) -> str:
            if not isinstance(text, str):
                return str(text)
            replacements = {
                "\u2018": "'", "\u2019": "'",
                "\u201c": '"', "\u201d": '"',
                "\u2014": "-", "\u2013": "-",
                "\u2022": "-",
                "\u20a0": "EUR", "\u20ac": "EUR",
                "\u2122": "TM",
            }
            for k, v in replacements.items():
                text = text.replace(k, v)
            return text.encode("latin-1", errors="replace").decode("latin-1")

        class ReportPDF(FPDF):
            def cell(self, w, h=0, text="", txt=None, *args, **kwargs):
                val = text if txt is None else txt
                cleaned = _clean_pdf_text(val)
                return super().cell(w, h, text=cleaned, *args, **kwargs)

            def multi_cell(self, w, h=0, text="", txt=None, *args, **kwargs):
                val = text if txt is None else txt
                cleaned = _clean_pdf_text(val)
                return super().multi_cell(w, h, text=cleaned, *args, **kwargs)

            def header(self):
                if self.page_no() > 1:
                    self.set_font("Helvetica", "I", 8)
                    self.set_text_color(120, 120, 120)
                    self.cell(0, 8, "AI Data Preprocessing & Analytics Report", align="R", ln=True)
                    self.line(10, 18, 200, 18)
                    self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(150, 150, 150)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        pdf = ReportPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # --- Page 1: Title and Overview ---
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(99, 102, 241) # Indigo primary
        pdf.cell(0, 15, "AI Data Preprocessing & Analytics Report", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, f"Dataset: {source_name}", ln=True)
        pdf.ln(5)

        # Dataset overview
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, "1. Dataset Cleaning Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        for lbl, val in [
            ("Original Row Count", raw_stats.get("rows", "N/A")),
            ("Columns Detected",   raw_stats.get("cols", "N/A")),
            ("Missing Values Imputed",  raw_stats.get("missing", "N/A")),
            ("Duplicate Rows Removed",  raw_stats.get("dupes", "N/A")),
        ]:
            pdf.cell(0, 6, f"  • {lbl}: {val}", ln=True)
        pdf.ln(5)

        # Executive Summary
        if insights_result:
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 10, "2. Executive Insights Summary", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            summary = insights_result.get("executive_summary", "")
            pdf.multi_cell(0, 6, summary)
            pdf.ln(5)

            # Key Insights list
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 10, "3. Key Business Insights", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            for i, ins in enumerate(insights_result.get("ai_insights", []), 1):
                pdf.multi_cell(0, 6, f"  {i}. {ins}")
                pdf.ln(1)
            pdf.ln(4)

            # Recommendations
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 10, "4. Strategic Business Recommendations", ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            for i, rec in enumerate(insights_result.get("recommendations", []), 1):
                pdf.multi_cell(0, 6, f"  {i}. {rec}")
                pdf.ln(1)

        # --- Visual charts pages ---
        if viz_result and viz_result.get("charts"):
            for i, chart in enumerate(viz_result["charts"], 1):
                img_b64 = chart.get("image_base64")
                if not img_b64:
                    continue
                
                try:
                    img_bytes = base64.b64decode(img_b64)
                    
                    pdf.add_page()
                    # Chart title
                    pdf.set_font("Helvetica", "B", 14)
                    pdf.set_text_color(30, 30, 30)
                    pdf.cell(0, 10, f"Chart {i}: {chart.get('title', 'Analysis')}", ln=True)
                    pdf.ln(2)
                    
                    # Embed in-memory image at current Y position
                    current_y = pdf.get_y()
                    pdf.image(io.BytesIO(img_bytes), x=20, y=current_y, w=170, h=106)
                    
                    # Chart business explanation below
                    if chart.get("description"):
                        pdf.set_y(current_y + 110) # Move past the embedded chart
                        pdf.set_font("Helvetica", "B", 11)
                        pdf.set_text_color(30, 30, 30)
                        pdf.cell(0, 8, "Business Value & Insights:", ln=True)
                        pdf.ln(1)
                        pdf.set_font("Helvetica", "", 10)
                        pdf.set_text_color(70, 70, 70)
                        pdf.multi_cell(0, 5, chart["description"])
                except Exception as chart_err:
                    print(f"Failed to embed chart {i} in PDF: {chart_err}")
                    continue

        return bytes(pdf.output())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        st.error(f"PDF generation failed: {exc}")
        return b""


# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Data Upload",
    "🧹 Preprocessing",
    "📊 Visualisations",
    "🤖 AI Insights",
    "📥 Export & Reports",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-hdr">Choose Your Data Source</div>', unsafe_allow_html=True)

    src_tab_file, src_tab_db, src_tab_api = st.tabs(["📂 File Upload", "🗄️ Database", "🌐 API"])

    # ── File upload ─────────────────────────────────────────────────────────
    with src_tab_file:
        uploaded = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Supports .csv, .xlsx, .xls",
            key="main_uploader",
        )
        if uploaded:
            ext = uploaded.name.split(".")[-1].lower()
            df_raw = _read_file(uploaded)
            st.session_state.raw_df      = df_raw
            st.session_state.raw_stats   = _raw_stats(df_raw)
            st.session_state.data_source = "file"
            st.session_state.source_name = uploaded.name

            # KPI strip
            rs = st.session_state.raw_stats
            c1, c2, c3, c4 = st.columns(4)
            _kpi(c1, "Rows",           f"{rs['rows']:,}")
            _kpi(c2, "Columns",        f"{rs['cols']}")
            _kpi(c3, "Missing Values", f"{rs['missing']:,}", delta_good=rs["missing"] == 0)
            _kpi(c4, "Duplicate Rows", f"{rs['dupes']}", delta_good=rs["dupes"] == 0)

            st.markdown('<div class="sec-hdr">Raw Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(df_raw.head(10), use_container_width=True)

            # Data types summary
            with st.expander("🔍 Column Data Types"):
                dtype_df = pd.DataFrame({
                    "Column":   df_raw.dtypes.index,
                    "Dtype":    df_raw.dtypes.astype(str).values,
                    "Non-Null": df_raw.notna().sum().values,
                    "Null":     df_raw.isnull().sum().values,
                    "Unique":   [df_raw[c].nunique() for c in df_raw.columns],
                })
                st.dataframe(dtype_df, use_container_width=True)

            normalize_cols = st.checkbox(
                "Scale numerical columns (Normalization between 0 and 1)",
                value=False,
                help="Recommended for Machine Learning preprocessing. Keep unchecked for business visualizations."
            )
            st.divider()
            if st.button("🧹 Clean & Preprocess", use_container_width=False, key="clean_file_btn"):
                uploaded.seek(0)
                if _call_clean(uploaded.getvalue(), uploaded.name, normalize=normalize_cols):
                    st.success("✅ Cleaning complete! Switch to the **🧹 Preprocessing** tab.")

    # ── Database ────────────────────────────────────────────────────────────
    with src_tab_db:
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL"], key="db_type")
        col_h, col_p = st.columns([3, 1])
        db_host = col_h.text_input("Host", "localhost", key="db_host")
        db_port = col_p.text_input("Port", "5432" if db_type == "PostgreSQL" else "3306", key="db_port")
        col_u, col_pw = st.columns(2)
        db_user = col_u.text_input("Username", "postgres", key="db_user")
        db_pass = col_pw.text_input("Password", type="password", key="db_pass")
        db_name = st.text_input("Database Name", "mydb", key="db_name")
        db_query = st.text_area("SQL Query", "SELECT * FROM my_table LIMIT 1000;", key="db_query")

        if st.button("🔗 Connect & Fetch", key="db_fetch_btn"):
            driver = "postgresql" if db_type == "PostgreSQL" else "mysql+pymysql"
            db_url = f"{driver}://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            with st.spinner("Connecting to database…"):
                resp = requests.post(
                    f"{FASTAPI_URL}/clean-db/",
                    json={"db_url": db_url, "query": db_query},
                    timeout=120,
                )
            if resp.status_code == 200:
                records = resp.json().get("cleaned_data", [])
                st.session_state.cleaned_json = records
                st.session_state.cleaned_df   = pd.DataFrame(records)
                st.session_state.data_source  = "db"
                st.session_state.source_name  = f"{db_name} — {db_query[:40]}…"
                st.success("✅ Data fetched and cleaned! Switch to the **🧹 Preprocessing** tab.")
            else:
                st.error(f"❌ {resp.json().get('detail', 'Database error')}")

    # ── API ─────────────────────────────────────────────────────────────────
    with src_tab_api:
        api_url = st.text_input(
            "API Endpoint URL",
            "https://jsonplaceholder.typicode.com/posts",
            key="api_url_input",
        )
        st.caption("The endpoint must return a JSON array of objects.")

        if st.button("🌐 Fetch & Clean", key="api_fetch_btn"):
            with st.spinner("Fetching data from API…"):
                resp = requests.post(
                    f"{FASTAPI_URL}/clean-api/",
                    json={"api_url": api_url},
                    timeout=120,
                )
            if resp.status_code == 200:
                records = resp.json().get("cleaned_data", [])
                st.session_state.cleaned_json = records
                st.session_state.cleaned_df   = pd.DataFrame(records)
                st.session_state.data_source  = "api"
                st.session_state.source_name  = api_url
                st.success("✅ API data fetched and cleaned! Switch to the **🧹 Preprocessing** tab.")
            else:
                st.error(f"❌ {resp.json().get('detail', 'API error')}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.cleaned_df is None:
        st.info("⬆️ Upload and clean your data in the **📤 Data Upload** tab first.")
    else:
        df_c   = st.session_state.cleaned_df
        df_raw = st.session_state.raw_df
        rs     = st.session_state.raw_stats

        # Status badge
        badge = (
            '<span class="badge-ai">✨ AI-Enhanced Cleaning</span>'
            if st.session_state.ai_enhanced
            else '<span class="badge-trad">⚙️ Traditional Cleaning</span>'
        )
        st.markdown(f'<div class="sec-hdr">Cleaning Status &nbsp; {badge}</div>', unsafe_allow_html=True)

        # Before / After comparison
        st.markdown('<div class="sec-hdr">Before → After Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        _diff_metric(m1, "Missing Values",  rs.get("missing", "?"),   df_c.isnull().sum().sum())
        _diff_metric(m2, "Duplicate Rows",  rs.get("dupes", "?"),     df_c.duplicated().sum())
        _diff_metric(m3, "Total Rows",      rs.get("rows", "?"),      len(df_c))
        _diff_metric(m4, "Columns",         rs.get("cols", "?"),      len(df_c.columns))

        # Column filter
        st.markdown('<div class="sec-hdr">Cleaned Dataset</div>', unsafe_allow_html=True)
        col_filter = st.multiselect(
            "Filter columns to display",
            options=df_c.columns.tolist(),
            default=df_c.columns.tolist(),
            key="col_filter",
        )
        st.dataframe(
            df_c[col_filter] if col_filter else df_c,
            use_container_width=True,
            height=400,
        )

        # Row count & shape after filtering
        st.caption(f"Showing {len(df_c):,} rows × {len(col_filter or df_c.columns)} columns")

        # Describe numeric
        with st.expander("📐 Statistical Summary (Numeric Columns)"):
            num_df = df_c.select_dtypes(include="number")
            if not num_df.empty:
                st.dataframe(num_df.describe().round(4).T, use_container_width=True)
            else:
                st.info("No numerical columns found.")

        # Trigger visualise / insights
        st.divider()
        btn_viz, btn_ins, _ = st.columns([1.5, 1.8, 5])
        if btn_viz.button("📊 Generate Visualisations", use_container_width=True, key="gen_viz_btn"):
            if _call_visualize(st.session_state.cleaned_json, include_png=True):
                st.success("✅ Done! Switch to the **📊 Visualisations** tab.")
        if btn_ins.button("🤖 Generate AI Insights", use_container_width=True, key="gen_ins_btn"):
            if _call_insights(st.session_state.cleaned_json, st.session_state.viz_result):
                st.success("✅ Done! Switch to the **🤖 AI Insights** tab.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.viz_result:
        st.info("⬆️ Generate visualisations from the **🧹 Preprocessing** tab first.")
        if st.session_state.cleaned_json:
            if st.button("📊 Generate Charts Now", key="viz_now_btn"):
                _call_visualize(st.session_state.cleaned_json, include_png=True)
    else:
        viz  = st.session_state.viz_result
        charts     = viz.get("charts", [])
        col_types  = viz.get("column_types", {})
        stats_summ = viz.get("summary_stats", {})

        # Column-type badges
        type_icons = {"numerical": "🔢", "categorical": "🏷️", "datetime": "📅", "boolean": "☑️"}
        badge_parts = "  ".join(
            f"{type_icons.get(t, '')} **{t}**: {len(cols)}"
            for t, cols in col_types.items() if cols
        )
        st.markdown(f"<small>{badge_parts}</small>", unsafe_allow_html=True)

        # ── Filters ────────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Chart Filters</div>', unsafe_allow_html=True)
        all_titles = [c.get("title", f"Chart {i}") for i, c in enumerate(charts)]
        chart_type_opts = sorted(set(c.get("type", "chart") for c in charts))

        fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
        selected_type = fcol1.multiselect(
            "Filter by chart type", chart_type_opts, default=chart_type_opts, key="ct_filter"
        )
        cols_per_row = fcol3.selectbox("Columns per row", [1, 2, 3], index=1, key="cpr_filter")

        filtered_charts = [
            c for c in charts
            if c.get("type", "chart") in selected_type
        ]

        # ── Section tabs ───────────────────────────────────────────────────
        vt1, vt2, vt3 = st.tabs(["📈 Univariate", "🔗 Bivariate", "🌡️ Correlation"])

        with vt1:
            uni = [c for c in filtered_charts
                   if c["type"] != "correlation_heatmap"
                   and "vs"  not in c.get("title", "").lower()
                   and "×"   not in c.get("title", "")
                   and " by " not in c.get("title", "").lower()]
            _render_chart_grid(uni, cols_per_row=cols_per_row)

        with vt2:
            biv = [c for c in filtered_charts
                   if c["type"] != "correlation_heatmap"
                   and ("vs"  in c.get("title", "").lower()
                        or "×" in c.get("title", "")
                        or " by " in c.get("title", "").lower())]
            _render_chart_grid(biv, cols_per_row=cols_per_row)

        with vt3:
            corr = [c for c in filtered_charts if c["type"] == "correlation_heatmap"]
            _render_chart_grid(corr, cols_per_row=1)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    if not st.session_state.insights_result:
        st.info("⬆️ Generate insights from the **🧹 Preprocessing** tab first.")
        if st.session_state.cleaned_json:
            if st.button("🤖 Generate AI Insights Now", key="ins_now_btn"):
                _call_insights(st.session_state.cleaned_json, st.session_state.viz_result)
    else:
        ins = st.session_state.insights_result

        # AI badge
        ai_badge = (
            '<span class="badge-ai">✨ Powered by LLaMA-3.3-70b</span>'
            if ins.get("ai_powered")
            else '<span class="badge-trad">⚙️ Rule-based fallback</span>'
        )
        st.markdown(f'{ai_badge}', unsafe_allow_html=True)

        # Executive summary
        st.markdown('<div class="sec-hdr">Executive Summary</div>', unsafe_allow_html=True)
        exec_text = ins.get("executive_summary", "—")
        st.markdown(
            f'<div class="exec-card">'
            f'<div class="exec-label">🗒 Summary</div>'
            f'<p class="exec-text">{exec_text}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Statistical metrics row
        st.markdown('<div class="sec-hdr">Dataset Metrics</div>', unsafe_allow_html=True)
        s = ins.get("statistical_summary", {})
        k1, k2, k3, k4, k5 = st.columns(5)
        _kpi(k1, "Total Rows",       f"{s.get('total_rows', 0):,}")
        _kpi(k2, "Total Columns",    f"{s.get('total_columns', 0)}")
        _kpi(k3, "Missing Handled",  f"{s.get('missing_values_handled', 0):,}", delta_good=True)
        _kpi(k4, "Dupes Removed",    f"{s.get('duplicates_removed', 0)}", delta_good=True)
        _kpi(k5, "Outliers Found",   f"{s.get('outliers_detected', 0)}",
             delta="⚠️ review" if s.get('outliers_detected', 0) > 0 else "✅ clean",
             delta_good=s.get('outliers_detected', 0) == 0)

        st.markdown("&nbsp;", unsafe_allow_html=True)

        # Main content tabs
        it1, it2, it3, it4, it5 = st.tabs([
            "💡 Insights", "📈 Patterns", "📝 Recommendations",
            "🔗 Correlations", "⚠️ Outliers",
        ])

        with it1:
            st.markdown('<div class="sec-hdr">Key Insights</div>', unsafe_allow_html=True)
            for i, txt in enumerate(ins.get("ai_insights", []), 1):
                st.markdown(
                    f'<div class="insight-card"><b style="color:#a5b4fc;">🔵 {i}.</b> {txt}</div>',
                    unsafe_allow_html=True,
                )
            if not ins.get("ai_insights"):
                st.info("No insights generated.")

        with it2:
            st.markdown('<div class="sec-hdr">Patterns & Trends</div>', unsafe_allow_html=True)
            for p in ins.get("patterns", []):
                st.markdown(
                    f'<div class="pattern-card">📈 {p}</div>',
                    unsafe_allow_html=True,
                )
            if not ins.get("patterns"):
                st.info("No patterns detected.")

        with it3:
            st.markdown('<div class="sec-hdr">Actionable Recommendations</div>', unsafe_allow_html=True)
            for i, rec in enumerate(ins.get("recommendations", []), 1):
                st.markdown(
                    f'<div class="rec-card"><b style="color:#fbbf24;">★ {i}.</b> {rec}</div>',
                    unsafe_allow_html=True,
                )
            if not ins.get("recommendations"):
                st.info("No recommendations generated.")

        with it4:
            st.markdown('<div class="sec-hdr">Significant Correlations</div>', unsafe_allow_html=True)
            corrs = ins.get("correlations", [])
            if corrs:
                df_corr = pd.DataFrame(corrs)
                st.dataframe(
                    df_corr.style.background_gradient(
                        subset=["correlation"], cmap="RdYlGn", vmin=-1, vmax=1
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No significant correlations (|r| ≥ 0.3) found.")

        with it5:
            st.markdown('<div class="sec-hdr">Outlier Report</div>', unsafe_allow_html=True)
            outliers = ins.get("outliers", [])
            if outliers:
                df_out = pd.DataFrame(outliers)
                st.dataframe(df_out, use_container_width=True)
                st.caption("Detected via IQR method (< Q1 − 1.5×IQR  or  > Q3 + 1.5×IQR)")
            else:
                st.success("✅ No outliers detected.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPORT & REPORTS
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    if st.session_state.cleaned_df is None:
        st.info("⬆️ Clean your data first in the **📤 Data Upload** tab.")
    else:
        df_c = st.session_state.cleaned_df

        st.markdown('<div class="sec-hdr">Download Cleaned Data</div>', unsafe_allow_html=True)
        dl1, dl2, dl3 = st.columns(3)

        # CSV
        dl1.download_button(
            "📄 Download CSV",
            data=df_c.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Excel
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
            df_c.to_excel(writer, index=False, sheet_name="Cleaned Data")
            if st.session_state.insights_result:
                ins = st.session_state.insights_result
                corrs = ins.get("correlations", [])
                if corrs:
                    pd.DataFrame(corrs).to_excel(writer, index=False, sheet_name="Correlations")
        dl2.download_button(
            "📊 Download Excel",
            data=xlsx_buf.getvalue(),
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # JSON
        dl3.download_button(
            "🗂 Download JSON",
            data=json.dumps(st.session_state.cleaned_json or [], indent=2).encode("utf-8"),
            file_name="cleaned_data.json",
            mime="application/json",
            use_container_width=True,
        )

        # ── Chart ZIP ──────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Download Charts</div>', unsafe_allow_html=True)
        if st.session_state.viz_result:
            charts = st.session_state.viz_result.get("charts", [])
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, chart in enumerate(charts):
                    title = chart.get("title", f"chart_{i}").replace("/", "_").replace(" ", "_")
                    # Save as self-contained HTML
                    zf.writestr(f"{i+1:02d}_{title}.html", chart["html"])
            st.download_button(
                "📦 Download All Charts (HTML ZIP)",
                data=zip_buf.getvalue(),
                file_name="charts.zip",
                mime="application/zip",
                use_container_width=False,
            )
            st.caption("Each HTML file is a fully interactive Plotly chart — open in any browser.")
        else:
            st.info("Generate visualisations first to enable chart download.")

        # ── PDF report ─────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">PDF Report</div>', unsafe_allow_html=True)
        st.markdown(
            "Generate a summary PDF containing dataset overview, cleaned data sample, "
            "AI insights, and recommendations."
        )
        if st.button("📄 Generate PDF Report", use_container_width=False, key="gen_pdf_btn"):
            with st.spinner("Building PDF (rendering charts might take a few seconds)…"):
                # Ensure visual charts are fetched with Base64 PNGs populated
                has_png = False
                if st.session_state.viz_result and st.session_state.viz_result.get("charts"):
                    has_png = any(c.get("image_base64") for c in st.session_state.viz_result["charts"])
                
                if not has_png and st.session_state.cleaned_json:
                    _call_visualize(st.session_state.cleaned_json, include_png=True)

                pdf_bytes = _generate_pdf(
                    source_name     = st.session_state.source_name or "Unknown",
                    raw_stats       = st.session_state.raw_stats,
                    cleaned_df      = df_c,
                    insights_result = st.session_state.insights_result,
                    viz_result      = st.session_state.viz_result,
                )
            if pdf_bytes:
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name="preprocessing_report.pdf",
                    mime="application/pdf",
                    use_container_width=False,
                    key="dl_pdf_btn",
                )

        # ── Insights text export ───────────────────────────────────────────
        if st.session_state.insights_result:
            st.markdown('<div class="sec-hdr">Export Insights as Text</div>', unsafe_allow_html=True)
            ins = st.session_state.insights_result
            lines = ["=== AI Data Preprocessing Report ===\n"]
            lines.append(f"Executive Summary:\n{ins.get('executive_summary','')}\n")
            lines.append("\nKey Insights:")
            for i, x in enumerate(ins.get("ai_insights", []), 1):
                lines.append(f"  {i}. {x}")
            lines.append("\nPatterns & Trends:")
            for p in ins.get("patterns", []):
                lines.append(f"  • {p}")
            lines.append("\nRecommendations:")
            for i, r in enumerate(ins.get("recommendations", []), 1):
                lines.append(f"  {i}. {r}")
            insights_txt = "\n".join(lines)
            st.download_button(
                "📋 Download Insights (.txt)",
                data=insights_txt.encode("utf-8"),
                file_name="insights.txt",
                mime="text/plain",
                use_container_width=False,
            )
            with st.expander("👁 Preview"):
                st.text(insights_txt)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Built with "
    "<b>Streamlit · FastAPI · LangGraph · Groq · Plotly · fpdf2</b>"
    "</small></center>",
    unsafe_allow_html=True,
)
