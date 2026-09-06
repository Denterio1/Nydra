"""
app.py — Nydra Web UI (Streamlit)

Run with:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import streamlit as st

st.set_page_config(
    page_title="Nydra — Data Inspection Agent",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "user" not in st.session_state:
    st.session_state["user"] = {"email": "guest@example.com", "name": "Guest", "plan": "professional"}

from src.security_vault import get_vault
from src.config_manager import get_manager

# Initialize Security Persistence Layer
vault = get_vault()
cm = get_manager(db_path="test.db", vault=vault)

def save_config_callback(key):
    """Callback to save a setting whenever it changes."""
    email = st.session_state.get("user", {}).get("email", "guest@example.com")
    if email and email != "guest@example.com":
        val = st.session_state.get(key)
        if val is not None:
            cm.save_config(email, key, val)

def load_user_persistence():
    """Load all saved settings for the current user into session state."""
    email = st.session_state["user"].get("email")
    if email and email != "guest@example.com":
        settings = cm.load_config(email)
        # Inject into session state if they exist in DB
        for key, value in settings.items():
            if key not in st.session_state or st.session_state[key] is None or st.session_state[key] == "":
                st.session_state[key] = value

load_user_persistence()

# ── Database Connector UI ─────────────────────────────────────────────────────

def render_database_connector():
    """Unified UI for connecting to and exploring databases."""
    st.markdown("### 🗄️ Database Connector")
    
    if "db_session" not in st.session_state:
        c1, c2 = st.columns(2)
        db_type = c1.selectbox("Database Type", ["sqlite", "postgresql", "mysql", "mariadb"])
        
        with st.expander("Connection Settings", expanded=True):
            if db_type == "sqlite":
                filepath = st.text_input("SQLite File Path", value="test.db")
                connect_btn = st.button("🔌 Connect to SQLite", type="primary")
                if connect_btn:
                    try:
                        connector = build_connector(db_type="sqlite", filepath=filepath)
                        ok, msg = connector.connect()
                        if ok:
                            st.session_state.db_session = DBSession(connector)
                            st.success(f"Connected to {filepath}")
                            st.rerun()
                        else:
                            st.error(msg)
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
            else:
                col_a, col_b = st.columns(2)
                host = col_a.text_input("Host", value="localhost")
                port = col_b.number_input("Port", value=5432 if db_type == "postgresql" else 3306)
                database = col_a.text_input("Database Name")
                username = col_b.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.button(f"🔌 Connect to {db_type.title()}", type="primary"):
                    try:
                        connector = build_connector(
                            db_type=db_type, host=host, port=port,
                            database=database, username=username, password=password
                        )
                        ok, msg = connector.connect()
                        if ok:
                            st.session_state.db_session = DBSession(connector)
                            st.success(f"Connected to {database}")
                            st.rerun()
                        else:
                            st.error(msg)
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
    else:
        # Already connected
        session = st.session_state.db_session
        status = session.status()
        
        st.success(f"✅ Connected to {status['db_type']}://{status['database']}")
        
        if st.button("🚫 Disconnect", type="secondary"):
            session.disconnect()
            del st.session_state["db_session"]
            if "db_data" in st.session_state: del st.session_state["db_data"]
            st.rerun()

        st.markdown("---")
        
        db_tabs = st.tabs(["📊 Table Browser", "💻 SQL Editor", "🤝 Relationships"])
        
        with db_tabs[0]:
            tables_df = session.browser.list_tables()
            if tables_df.empty:
                st.warning("No tables found in this database.")
            else:
                st.dataframe(tables_df, use_container_width=True, hide_index=True)
                
                selected_table = st.selectbox("Select Table to Analyse:", tables_df["Table"].tolist())
                
                col_btn1, col_btn2 = st.columns(2)
                if col_btn1.button("🔍 Preview Data", use_container_width=True):
                    st.session_state.db_preview = session.browser.preview(selected_table)
                
                if col_btn2.button("🚀 Load for Full Analysis", type="primary", use_container_width=True):
                    with st.spinner(f"Loading {selected_table}..."):
                        st.session_state.db_data = session.editor.run_table(selected_table)
                        st.success(f"✓ {selected_table} loaded! Nydra is ready.")
                        st.rerun()
                
                if "db_preview" in st.session_state:
                    st.markdown("#### Preview")
                    st.dataframe(st.session_state.db_preview, use_container_width=True)

        with db_tabs[1]:
            st.markdown("#### SQL Editor")
            sql_input = st.text_area("Write your SELECT query:", placeholder="SELECT * FROM table LIMIT 100", height=150)
            
            if st.button("▶ Run Query", type="primary"):
                if sql_input:
                    try:
                        with st.spinner("Executing..."):
                            res = session.editor.run(sql_input)
                            st.session_state.sql_result = res
                            st.success(f"✓ Query successful: {res['row_count']} rows in {res['duration_ms']:.0f}ms")
                    except Exception as e:
                        st.error(f"Query Error: {e}")
            
            if "sql_result" in st.session_state:
                res = st.session_state.sql_result
                st.dataframe(res["df"], use_container_width=True)
                if st.button("🚀 Analyse these results"):
                    st.session_state.db_data = res
                    st.rerun()

        with db_tabs[2]:
            st.markdown("#### Database Relationships")
            rels_df = session.relationships.relationship_summary()
            if rels_df.empty:
                st.info("No foreign key relationships detected.")
            else:
                st.dataframe(rels_df, use_container_width=True, hide_index=True)

import hashlib
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    st.error(f"Failed to import Plotly: {e}")
    st.info("Try to 'Reboot app' with 'Clear cache' in Streamlit Cloud management.")
    st.stop()
from io import StringIO, BytesIO

from src.data.advanced_outlier import OutlierAnalyzer, SmartDetector, detect_outliers
from src.data.db_converter import (
    MultiFileConverter, ConversionConfig, preview_schema, convert_to_bytes
)
from src.data.db_connector import build_connector
from src.data.db_query     import DBSession
from src.data.data_chat import get_chatbot
from src.ux import render_welcome, render_privacy_tab, render_audit_tab
from src.security import security, PrivacyManager
from src.auth import get_user_stats
from src.data.encoding_advisor  import encoding_advisor
from src.data.schema_validator  import validate_schema, infer_schema, schema_to_dict, schema_from_dict
from src.data.relationships     import detect_relationships
from src.data.ml_readiness      import ml_readiness
from src.data.preparator        import prepare_for_ml

from src.data.analyzer import full_report, detect_outliers
from src.data.loader       import load_file
from src.data.reliability  import load_bytes_resilient
from src.data.contracts    import evaluate_contracts
from src.data.auto_repair  import apply_safe_fixes
from src.data.audit_trail  import build_repair_audit
from src.data.cleaner      import handle_missing, remove_duplicates
from src.data.advanced_automl import AdvancedAutoML, run_automl as run_advanced_automl
from src.data.advanced_imputer import SmartImputer, AdvancedMultimodalImputer, ai_impute as run_ai_impute
from src.data.drift        import detect_drift
from src.core.agent        import Nydra
from src.core.human_analyst import run_human_analyst

# ── Image Version 6 Imports ──────────────────────────────────────────────────
try:
    from src.vision_nlp.image_loader       import ImageLoader
    from src.vision_nlp.image_quality      import analyze_dataset_quality, ImageQualityAnalyzer
    from src.vision_nlp.image_analyzer     import ImageAnalyzer
    from src.vision_nlp.image_report       import generate_report
    from src.vision_nlp.audit_orchestrator import AuditOrchestrator, run_image_audit
    IMAGE_V6_AVAILABLE = True
except ImportError:
    IMAGE_V6_AVAILABLE = False

# ── Cache functions ───────────────────────────────────────────────
def _file_hash(file) -> str:
    """Generate a stable MD5 hash for an uploaded file."""
    return hashlib.md5(file.getvalue()).hexdigest()

@st.cache_data
def _get_quality(df):
    from src.data.quality_score import DataQualityScorer
    return DataQualityScorer().score(df)

@st.cache_data
def _get_outliers(df):
    from src.data.advanced_outlier import OutlierAnalyzer
    return OutlierAnalyzer(verbose=False).analyze(df)

@st.cache_data
def _get_relationships(df_json, threshold=0.4):
    from src.data.relationships import detect_relationships
    # We pass data as a dict/json-compatible object for caching if possible, 
    # but st.cache_data handles DataFrames well too.
    df = pd.read_json(df_json) if isinstance(df_json, str) else df_json
    return detect_relationships({"df": df}, threshold=threshold)

@st.cache_data
def _get_privacy_report(df):
    from src.security import PrivacyManager
    pm = PrivacyManager()
    return pm.privacy_report(df)

@st.cache_data(show_spinner=False)
def _cached_analysis_v5(file_hash: str, file_bytes: bytes, file_name: str):
    """Run full analysis with resilient ingestion — cached by file hash."""
    from src.data.analyzer  import full_report
    from src.data.ml_readiness  import ml_readiness

    ingest = load_bytes_resilient(file_bytes, file_name)
    data = ingest["data"]
    df = data["df"]
    
    # Core analysis
    analysis = full_report(data)
    
    # We don't call detect_relationships here, we'll call the cached version later
    # to allow the UI to render parts of the page faster.
    
    # Outliers and ML Readiness are often needed for the header score
    outliers = _get_outliers(df)
    ml       = ml_readiness(data, outliers)
    
    return data, analysis, outliers, ml, ingest
def _load_uploaded(uploaded_file) -> dict:
    """Load an uploaded file into a data dict."""
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    ingest = load_bytes_resilient(file_bytes, uploaded_file.name)
    for warn in ingest["warnings"]:
        st.warning(warn)
    for risk in ingest["risk_flags"]:
        st.info(f"Risk flag: {risk}")
    return ingest["data"]


def _health_color(score: int) -> str:
    if score >= 85: return "#c8f06e"
    if score >= 65: return "#f0c46e"
    return "#f06e6e"


def _severity_color(sev: str) -> str:
    return {"none": "#c8f06e", "low": "#f0c46e", "medium": "#f0c46e", "high": "#f06e6e"}.get(sev, "#7a7875")


# ── Sidebar ───────────────────────────────────────────────────────────────────

user_email = st.session_state.get("user", {}).get("email", "guest@example.com")

with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 2rem'>
        <div style='font-family:DM Mono,monospace;font-size:11px;color:#7a7875;letter-spacing:0.12em;margin-bottom:0.5rem'>🩺 NYDRA</div>
        <div style='font-family:Fraunces,serif;font-size:1.4rem;font-weight:300;color:#e8e6e1'>Data Inspection<br><em style='color:#c8f06e'>Agent</em></div>
    </div>
    """, unsafe_allow_html=True)

    # User Profile
    with st.expander("👤 User Identity", expanded=user_email == "guest@example.com"):
        new_email = st.text_input("Email to sync settings", value=user_email)
        if new_email != user_email:
            st.session_state["user"]["email"] = new_email
            # Clear previous settings to force reload from new user
            for key in ["llm_api_key", "llm_provider", "llm_model", "missing_strategy", "remove_dupes", "run_ml", "run_rels", "analyst_profile"]:
                if key in st.session_state: del st.session_state[key]
            st.rerun()

    stats = get_user_stats(user_email)
    from src.ux import render_user_profile
    render_user_profile(stats, security.session_manager.get(user_email))
    
    st.markdown("---")

    st.markdown("### Upload your file")
    uploaded = st.file_uploader(
        "CSV, Excel, or JSON",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Settings")

    missing_strategy  = st.selectbox(
        "Missing value strategy",
        ["mean", "median", "mode", "drop", "knn", "mice", "missforest", "auto", "ensemble", "ffill", "bfill"],
        index=0,
        key="missing_strategy",
        on_change=save_config_callback, args=("missing_strategy",)
    )

    remove_dupes = st.checkbox("Remove duplicate rows", value=True, key="remove_dupes", on_change=save_config_callback, args=("remove_dupes",))
    run_ml       = st.checkbox("Run ML Readiness check", value=True, key="run_ml", on_change=save_config_callback, args=("run_ml",))
    run_rels     = st.checkbox("Detect column relationships", value=True, key="run_rels", on_change=save_config_callback, args=("run_rels",))
    analyst_profile = st.selectbox(
        "Analyst safety profile",
        ["conservative", "balanced", "aggressive"],
        index=1,
        help="Conservative favors safer recommendations. Aggressive favors faster modeling.",
        key="analyst_profile",
        on_change=save_config_callback, args=("analyst_profile",)
    )

    # ── AI Configuration ───────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Configuration")
    
    from src.data.data_chat import get_providers, make_config
    
    providers_info = get_providers()
    
    selected_provider_key = st.selectbox(
        "Provider",
        options=list(providers_info.keys()),
        format_func=lambda k: providers_info[k]["name"],
        key="llm_provider",
        on_change=save_config_callback, args=("llm_provider",)
    )
    
    provider_meta = providers_info[selected_provider_key]
    
    user_api_key = st.text_input(
        f"{provider_meta['name']} API Key",
        type="password",
        placeholder=provider_meta["key_hint"],
        help=provider_meta["notes"],
        key="llm_api_key",
        on_change=save_config_callback, args=("llm_api_key",)
    )
    
    selected_model = st.selectbox(
        "Model",
        options=provider_meta["models"],
        index=provider_meta["models"].index(provider_meta["recommended"]) if provider_meta["recommended"] in provider_meta["models"] else 0,
        key="llm_model",
        on_change=save_config_callback, args=("llm_model",)
    )

    if user_api_key or selected_provider_key == "ollama":
        st.success(f"🚀 AI Ready: {provider_meta['name']} mode active.")
        if user_email and user_email != "guest@example.com":
            st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:10px;color:#c8f06e;text-align:center'>🛡️ Securely Synced: {user_email}</div>", unsafe_allow_html=True)
    else:
        st.info("💡 Lite Mode: Enter an API key for full AI discussion.")

    st.markdown("---")
    st.markdown("### Drift Detection")
    uploaded_baseline = st.file_uploader(
        "Baseline file (optional)",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="visible",
        key="baseline",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:11px;color:#7a7875'>"
        "Nydra v0.5.0<br>Open Source</div>",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────

# Check if we have data from either upload or database
has_data = uploaded is not None or "db_data" in st.session_state
is_db_connected = "db_session" in st.session_state

# Initialize variables to avoid errors if no data is loaded
analysis = outliers = ml = rels = privacy_report = clean_data = analyst_report = contract_report = None
clean_log = {}
source_name = "No data loaded"

if not has_data:
    render_welcome()
    st.markdown("---")
    
    if is_db_connected:
        st.success("🔌 Connected to database! Please select a table below to begin analysis.")
        # Call the unified renderer directly here if connected but no table selected
        render_database_connector()
    else:
        st.info("👆 Upload a file in the sidebar or go to the **Database** tab below to begin.")
else:
    # ── Load & analyse ────────────────────────────────────────────────────────────
    if uploaded:
        with st.spinner("Analysing your data..."):
            file_hash = _file_hash(uploaded)
            uploaded.seek(0)
            file_bytes = uploaded.read()
            uploaded.seek(0)

            # 1. Use cached analysis
            data, analysis, outliers, ml, ingest_info = _cached_analysis_v5(
                file_hash,
                file_bytes,
                uploaded.name,
            )
            
            # 2. Lazy-load heavy components
            # We only run these if the user is actually on those tabs, but for now we'll 
            # just use the cached versions to speed up the main thread.
            rels = _get_relationships(data["df"]) if run_rels else []
            privacy_report = _get_privacy_report(data["df"])
            
            for warn in ingest_info["warnings"]:
                st.warning(warn)
            for risk in ingest_info["risk_flags"]:
                st.info(f"Risk flag: {risk}")
        
        # Security checks
        if user_email:
            allowed, reason = security.check_request(user_email, "analyse", uploaded.name)
            if not allowed:
                st.error(reason)
                st.stop()
                
        file_bytes = uploaded.getvalue()
        val_result = security.file_validator.validate(
            uploaded.name,
            len(file_bytes),
            file_bytes
        )
        if not val_result.is_valid:
            for err in val_result.errors:
                st.error(err)
            st.stop()
        for warn in val_result.warnings:
            st.warning(warn)
        
        source_name = uploaded.name
    else:
        # Use data from database
        data = st.session_state["db_data"]
        with st.spinner("Analysing database table..."):
            from src.data.analyzer      import full_report, detect_outliers
            from src.data.ml_readiness   import ml_readiness
            from src.data.relationships  import detect_relationships
            from src.security            import PrivacyManager

            analysis = full_report(data)
            outliers = detect_outliers(data)
            ml       = ml_readiness(data, outliers)
            rels     = detect_relationships(data, threshold=0.4)
            pm       = PrivacyManager()
            privacy_report = pm.privacy_report(data["df"])
        
        source_name = data.get("source", "Database Table")

    analyst_report = run_human_analyst(
        data=data,
        analysis=analysis,
        outliers=outliers,
        ml=ml,
        rels=rels,
        privacy_report=privacy_report,
        risk_profile=analyst_profile,
    )
    contract_report = evaluate_contracts(data)

    # ── Clean & Prepare ───────────────────────────────────────────────────────────
    clean_data = data
    clean_log  = {}
    if remove_dupes and analysis["duplicate_rows"] > 0:
        clean_data, n = remove_duplicates(clean_data)
        clean_log["duplicates_removed"] = n
    if sum(analysis["missing_values"].values()) > 0:
        from src.data.advanced_imputer import impute_data_dict
        clean_data, result = impute_data_dict(clean_data, strategy=missing_strategy)
        # Convert Advanced Imputer result to cleaning log format
        if missing_strategy == "drop":
             clean_log["missing"] = {"rows_dropped": result.n_imputed}
        else:
             clean_log["missing"] = {
                 col: {"filled": info["n_filled"], "replacement": info["sample_val"], "strategy": missing_strategy}
                 for col, info in result.col_changes.items()
             }


# ── Header ────────────────────────────────────────────────────────────────────

if has_data:
    col_title, col_score = st.columns([3, 1])

    with col_title:
        st.markdown(f"""
        <div class='main-title'>Report for<br><em>{source_name}</em></div>
        <div class='subtitle'>{analysis['shape']['rows']:,} rows · {analysis['shape']['columns']} columns · {source_name.split('.')[-1].upper()}</div>
        """, unsafe_allow_html=True)

    with col_score:
        if ml:
            score = ml["score"]
            color = _health_color(score)
            st.markdown(f"""
            <div style='text-align:right;padding-top:1rem'>
                <div style='font-family:Fraunces,serif;font-size:3.5rem;font-weight:600;color:{color};line-height:1'>{score}</div>
                <div style='font-family:DM Mono,monospace;font-size:11px;color:#7a7875'>ML Health Score / 100</div>
                <div style='width:100%;height:3px;background:rgba(255,255,255,0.07);border-radius:2px;margin-top:8px'>
                    <div style='width:{score}%;height:100%;background:{color};border-radius:2px'></div>
                </div>
                <div style='font-family:DM Mono,monospace;font-size:11px;color:{color};margin-top:4px'>Grade {ml["grade"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Overview metrics ──────────────────────────────────────────────────────────

    m1, m2, m3, m4, m5 = st.columns(5)
    total_missing = sum(analysis["missing_values"].values())

    def _metric(col, val, label, color):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-val' style='color:{color}'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    _metric(m1, f"{analysis['shape']['rows']:,}", "rows", "#5ce0c6")
    _metric(m2, analysis["shape"]["columns"],      "columns", "#5ce0c6")
    _metric(m3, total_missing,  "missing cells",  "#f06e6e" if total_missing > 0 else "#c8f06e")
    _metric(m4, analysis["duplicate_rows"], "duplicate rows", "#f06e6e" if analysis["duplicate_rows"] > 0 else "#c8f06e")
    outlier_cols = len(set(r.column for r in outliers.method_results if r.column and r.outlier_count > 0))
    _metric(m5, outlier_cols,  "cols with outliers", "#f0c46e" if outliers.n_outliers > 0 else "#c8f06e")

    st.markdown("<br>", unsafe_allow_html=True)


# ── Master Tabs ────────────────────────────────────────────────────────────────

master_tabs = st.tabs(["📊 Insights", "🛡️ Health & Quality", "🛠️ Preparation", "🔬 AI Audits", "🧠 Intelligence"])

# ── Master Tab 1: Insights ───────────────────────────────────────────────────
with master_tabs[0]:
    st.markdown("<div class='glass-card suite-insights'>", unsafe_allow_html=True)
    inner_tabs = st.tabs(["📊 Overview", "📈 Statistics", "🤖 ML Readiness", "🔗 Relationships", "🧬 Cognitive DNA"])
    
    # Overview
    with inner_tabs[0]:
        if not has_data:
            st.info("Please load data to see the Overview.")
        else:
            st.markdown("#### Data Preview")
            st.dataframe(data["df"].head(20), use_container_width=True)

            st.markdown("#### Column Types")
            type_data = {
                col: "Numeric" if s["type"] == "numeric" else "Text"
                for col, s in analysis["column_stats"].items()
            }
            fig = px.pie(
                names=list(type_data.values()),
                title="Column type distribution",
                color_discrete_map={"Numeric": "#5ce0c6", "Text": "#c8f06e"},
                hole=0.5,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
            st.plotly_chart(fig, use_container_width=True)

    # Statistics
    with inner_tabs[1]:
        if not has_data:
            st.info("Please load data to see Statistics.")
        else:
            st.markdown("#### Column Statistics")
            stats = analysis["column_stats"]

            num_cols = [col for col, s in stats.items() if s["type"] == "numeric"]
            cat_cols = [col for col, s in stats.items() if s["type"] == "text"]

            if num_cols:
                st.markdown("##### Numeric columns")
                num_df = pd.DataFrame([
                    {"Column": col, "Min": stats[col]["min"], "Max": stats[col]["max"],
                     "Mean": stats[col]["mean"], "Unique": stats[col]["unique"], "Count": stats[col]["count"]}
                    for col in num_cols
                ])
                st.dataframe(num_df, use_container_width=True)

                selected = st.selectbox("Select column to visualize", num_cols)
                fig = px.histogram(
                    data["df"], x=selected, nbins=20,
                    title=f"{selected} distribution",
                    color_discrete_sequence=["#5ce0c6"],
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
                st.plotly_chart(fig, use_container_width=True)

            if cat_cols:
                st.markdown("##### Categorical columns")
                cat_df = pd.DataFrame([
                    {"Column": col, "Unique Values": stats[col]["unique"],
                     "Most Common": stats[col]["most_common"], "Count": stats[col]["count"]}
                    for col in cat_cols
                ])
                st.dataframe(cat_df, use_container_width=True)

                selected_cat = st.selectbox("Select column to visualize", cat_cols, key="cat_select")
                vc = data["df"][selected_cat].value_counts().reset_index()
                vc.columns = [selected_cat, "count"]
                fig = px.bar(vc, x=selected_cat, y="count", title=f"{selected_cat} value counts",
                             color_discrete_sequence=["#c8f06e"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
                st.plotly_chart(fig, use_container_width=True)

    # ML Readiness
    with inner_tabs[2]:
        if not has_data:
            st.info("Please load data to see ML Readiness.")
        elif not ml:
            st.info("Enable ML Readiness check in the sidebar.")
        else:
            from src.data.quality_score import DataQualityScorer
            profile = _get_quality(data["df"])
            st.metric("Overall Quality Score", f"{profile.overall_score}/100")
            st.write(profile.grade_label)
            st.write(profile.health_badge)
            score = ml["score"]
            color = _health_color(score)

            st.markdown(f"""
            <div style='text-align:center;padding:2rem 0'>
                <div style='font-family:Fraunces,serif;font-size:5rem;font-weight:600;color:{color};line-height:1'>{score}</div>
                <div style='font-family:DM Mono,monospace;font-size:13px;color:#7a7875'>out of 100 — Grade {ml["grade"]}</div>
                <div style='font-size:15px;color:#e8e6e1;margin-top:1rem'>{ml["summary"]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Checks")
            for check in ml["checks"]:
                icon  = "✅" if check["status"] == "pass" else "⚠️" if check["status"] == "warn" else "❌"
                color_text = "#c8f06e" if check["status"] == "pass" else "#f0c46e" if check["status"] == "warn" else "#f06e6e"
                pts   = check["points"]
                max_p = check["max"]
                pct   = int(pts / max_p * 100)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{icon} **{check['name']}**  \n{check['detail']}")
                    st.progress(pct / 100)
                with col2:
                    st.markdown(f"<div style='text-align:right;font-family:DM Mono,monospace;color:{color_text};font-size:1.2rem;padding-top:0.5rem'>{pts}/{max_p}</div>", unsafe_allow_html=True)
                st.markdown("---")

    # Relationships
    with inner_tabs[3]:
        if not has_data:
            st.info("Please load data to see Relationships.")
        elif not run_rels:
            st.info("Enable relationship detection in the sidebar.")
        elif not rels:
            st.success("No strong relationships found between columns.")
        else:
            st.markdown(f"#### Found {len(rels)} relationship(s)")
            for r in rels:
                strength = r["strength"]
                color    = "#c8f06e" if strength >= 0.8 else "#f0c46e" if strength >= 0.6 else "#7a7875"
                level    = "Strong" if strength >= 0.8 else "Moderate" if strength >= 0.6 else "Weak"
                st.markdown(f"""
                **{r['col_a']}** ↔ **{r['col_b']}**  
                `{r['method']}` · {r['type']} · {r['direction']}
                """)
                st.progress(strength)
                st.markdown(f"<span style='color:{color};font-family:DM Mono,monospace;font-size:12px'>{level} ({strength})</span>", unsafe_allow_html=True)
                st.markdown("---")

            # Correlation heatmap for numeric columns
            num_cols = [c for c in data["df"].columns if pd.api.types.is_numeric_dtype(data["df"][c])]
            if len(num_cols) >= 2:
                st.markdown("#### Correlation Heatmap")
                corr = data["df"][num_cols].corr()
                fig  = px.imshow(
                    corr, text_auto=True, aspect="auto",
                    color_continuous_scale=["#f06e6e", "#17171a", "#c8f06e"],
                    title="Pearson correlation matrix",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
                st.plotly_chart(fig, use_container_width=True)

    # Cognitive DNA
    with inner_tabs[4]:
        if not has_data:
            st.info("Please load data to see Cognitive DNA.")
        else:
            st.markdown("#### 🧬 Cognitive Data DNA")
            st.markdown("---")
            
            try:
                from src.data.dna_memory import get_dna_manager
                from src.data.cognitive_dna import CognitiveDNA, DataDNA
                
                target_col_dna = st.selectbox("Target column (optional)", ["None"] + list(data["df"].columns), key="dna_target")
                target_col_dna = None if target_col_dna == "None" else target_col_dna
                
                if st.button("🧬 Generate & Analyze DNA"):
                    with st.spinner("Computing Cognitive DNA..."):
                        dna_obj = CognitiveDNA(data["df"], target_col=target_col_dna)
                        manager = get_dna_manager()
                        
                        dna = DataDNA(
                            statistical = dna_obj.statistical,
                            structural  = dna_obj.structural,
                            ml          = dna_obj.ml,
                            temporal    = dna_obj.temporal,
                            dna_hash    = dna_obj.dna_hash,
                            created_at  = dna_obj.created_at,
                            source      = source_name,
                            personality = dna_obj.personality,
                            tags        = dna_obj.tags
                        )
                        
                        analysis_result = manager.full_analysis(dna, source_name, ml_score=ml["score"] if ml else 0)
                        
                        col_dna1, col_dna2 = st.columns(2)
                        
                        with col_dna1:
                            st.markdown("##### 🆔 Identity")
                            st.metric("DNA Hash", analysis_result["dna_hash"])
                            st.markdown(f"**Personality:** {analysis_result['personality']}")
                            st.markdown(f"**Tags:** {', '.join(analysis_result['tags'])}")
                            st.markdown(f"**Total in DB:** {analysis_result['total_in_db']}")
                            
                            st.markdown("##### 📈 Evolution")
                            evo = analysis_result["evolution"]
                            st.metric("Version", evo.get("version", 1))
                            st.markdown(f"**Trend:** `{evo.get('trend', 'stable')}`")
                            st.markdown(f"**ML Trend:** `{evo.get('ml_trend', 'stable')}`")
                            if evo.get("changes"):
                                st.markdown("**Key Changes:**")
                                for ch in evo["changes"]:
                                    st.markdown(f"- {ch}")
                        
                        with col_dna2:
                            st.markdown("##### 🎯 Strategy Recommendation")
                            rec = analysis_result["recommendation"]
                            st.success(f"**{rec['strategy']}**")
                            st.progress(rec["confidence"])
                            st.markdown(f"**Confidence:** {int(rec['confidence']*100)}%")
                            st.markdown(f"**Explanation:** {rec['explanation']}")
                            if rec.get("based_on"):
                                st.markdown(f"**Based on:** {', '.join(rec['based_on'])}")

                        st.markdown("---")
                        st.markdown("##### 👯 Similar Datasets")
                        similar = analysis_result["similar"]
                        if not similar:
                            st.info("First time seeing this type of data. No similar datasets found yet.")
                        else:
                            for s in similar:
                                col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
                                col_s1.markdown(f"**{s['source']}**  \n*{s['personality']}*")
                                col_s2.metric("Similarity", f"{int(s['similarity']*100)}%")
                                col_s3.metric("ML Score", s["ml_score"])
                                st.markdown("---")
            except Exception as e:
                st.error(f"DNA Analysis Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Master Tab 2: Health & Quality ───────────────────────────────────────────
with master_tabs[1]:
    st.markdown("<div class='glass-card suite-health'>", unsafe_allow_html=True)
    health_tabs = st.tabs(["🔍 Quality", "🛡️ Privacy", "📉 Drift", "📋 Audit"])
    
    # Quality
    with health_tabs[0]:
        if not has_data:
            st.info("Please load data to see Quality analysis.")
        else:
            st.markdown("#### Missing Values")
            mv = analysis["missing_values"]
            if total_missing == 0:
                st.success("No missing values — data is complete!")
            else:
                mv_df = pd.DataFrame([
                    {"Column": col, "Missing": cnt, "Percentage": round(cnt / analysis["shape"]["rows"] * 100, 1)}
                    for col, cnt in mv.items()
                ])
                fig = px.bar(
                    mv_df, x="Column", y="Percentage",
                    color="Percentage",
                    color_continuous_scale=["#c8f06e", "#f0c46e", "#f06e6e"],
                    title="Missing values per column (%)",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(mv_df, use_container_width=True)

            st.markdown("#### Outliers")
            from src.data.advanced_outlier import OutlierAnalyzer
            analyzer       = OutlierAnalyzer(verbose=False)
            outlier_report = _get_outliers(data["df"])
            consensus      = outlier_report.ensemble.summary

            col1, col2, col3 = st.columns(3)
            col1.metric("Consensus Outliers", consensus["consensus_outliers"])
            col2.metric("Outlier Rate",       f"{consensus['consensus_rate']:.2%}")
            col3.metric("Smart Method",       outlier_report.smart_choice)

            st.dataframe(analyzer.to_dataframe(outlier_report), use_container_width=True)

            for rec in outlier_report.smart_result.details.get("smart_reasoning", []):
                st.caption(rec)

            st.markdown("#### Contract Violations")
            if contract_report and contract_report["counts"]["total"] > 0:
                c = contract_report["counts"]
                st.warning(f"{c['total']} violation(s): {c['high']} high, {c['medium']} medium.")
                for v in contract_report["violations"][:20]:
                    col_txt = f"[{v['column']}]" if v.get("column") else "[dataset]"
                    st.markdown(f"- **{v['level'].upper()}** {col_txt} {v['message']}")
                if st.button("🛠️ Apply Safe Fixes", key="apply_safe_fixes_quality"):
                    repaired, actions = apply_safe_fixes(data, contract_report, missing_strategy=missing_strategy)
                    audit = build_repair_audit(
                        source_name,
                        (len(data["df"]), len(data["df"].columns)),
                        (len(repaired["df"]), len(repaired["df"].columns)),
                        actions,
                    )
                    st.session_state["safe_repaired_data"] = repaired
                    st.session_state["safe_repair_audit"] = audit
                    st.success(f"Applied {len(actions)} safe action(s). See Cleaning tab for details.")
            else:
                st.success("No contract violations detected.")

    # Privacy
    with health_tabs[1]:
        if not has_data:
            st.info("Please load data to see the Privacy report.")
        else:
            render_privacy_tab(privacy_report, data["df"])

    # Drift
    with health_tabs[2]:
        if not has_data:
            st.info("Please load data to see Drift detection.")
        elif not uploaded_baseline:
            st.info("Upload a baseline file in the sidebar to detect drift.")
        else:
            with st.spinner("Detecting drift..."):
                baseline_data = _load_uploaded(uploaded_baseline)
                drift_result  = detect_drift(baseline_data, data)
                baseline_contract = evaluate_contracts(baseline_data)
                current_contract = evaluate_contracts(data)

            sev   = drift_result["severity"]
            color = _severity_color(sev)

            st.markdown(f"""
            <div style='text-align:center;padding:1.5rem 0'>
                <div style='font-family:Fraunces,serif;font-size:2.5rem;font-weight:600;color:{color}'>{sev.upper()}</div>
                <div style='font-family:DM Mono,monospace;font-size:13px;color:#7a7875'>Drift Severity</div>
                <div style='font-size:15px;color:#e8e6e1;margin-top:0.5rem'>{drift_result["summary"]}</div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            col1.metric("Baseline rows", drift_result["base_shape"]["rows"])
            col2.metric("Current rows",  drift_result["new_shape"]["rows"])
            col3, col4 = st.columns(2)
            col3.metric("Baseline contract violations", baseline_contract["counts"]["total"])
            col4.metric("Current contract violations", current_contract["counts"]["total"])

            if drift_result["drifted_columns"]:
                st.markdown("#### Drifted Columns")
                for d in drift_result["drifted_columns"]:
                    sev_c = "#f06e6e" if d["severity"] == "high" else "#f0c46e"
                    with st.expander(f"⚠️ {d['column']} — {d['severity'].upper()}"):
                        for issue in d["issues"]:
                            st.markdown(f"- {issue}")

            if drift_result["stable_columns"]:
                st.markdown("#### Stable Columns")
                st.success(", ".join(drift_result["stable_columns"]))

    # Audit
    with health_tabs[3]:
        if user_email:
            render_audit_tab(user_email)
        else:
            st.info("Login with your email in the sidebar to see your activity log.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Master Tab 3: Preparation ────────────────────────────────────────────────
with master_tabs[2]:
    st.markdown("<div class='glass-card suite-prep'>", unsafe_allow_html=True)
    prep_tabs = st.tabs(["🧹 Cleaning", "🔬 Lab", "📂 Multi-File", "🗄️ Database"])
    
    # Cleaning
    with prep_tabs[0]:
        if not has_data:
            st.info("Please load data to see Cleaning options.")
        else:
            repaired_data = st.session_state.get("safe_repaired_data")
            repair_audit = st.session_state.get("safe_repair_audit")
            if repaired_data is not None:
                st.markdown("#### Safe Repair Output")
                st.success(
                    f"Repaired shape: {repair_audit['after_shape']['rows']:,} rows x "
                    f"{repair_audit['after_shape']['cols']} cols."
                )
                if repair_audit["actions"]:
                    for act in repair_audit["actions"]:
                        st.markdown(f"- `{act['action']}`")
                st.dataframe(repaired_data["df"].head(20), use_container_width=True)
                repaired_csv = repaired_data["df"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download safe-repaired CSV",
                    data=repaired_csv,
                    file_name=f"{os.path.splitext(uploaded.name if uploaded else 'db')[0]}_safe_repaired.csv",
                    mime="text/csv",
                )
                st.markdown("---")

            st.markdown("#### Cleaning Log")
            if not clean_log:
                st.success("No cleaning was necessary.")
            else:
                if "duplicates_removed" in clean_log:
                    st.warning(f"Removed **{clean_log['duplicates_removed']}** duplicate row(s).")
                if "missing" in clean_log:
                    mv = clean_log["missing"]
                    if "rows_dropped" in mv:
                        st.warning(f"Dropped **{mv['rows_dropped']}** row(s) with missing values.")
                    else:
                        for col, info in mv.items():
                            st.info(f"**{col}**: filled {info['filled']} value(s) with `{info['replacement']}` ({info['strategy']})")

            st.markdown("#### Cleaned Data Preview")
            st.dataframe(clean_data["df"].head(20), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🤖 Advanced AI Multimodal Imputation")
            st.caption("Use Transformers, Vision AI, and Diffusion to fill complex gaps.")
            
            col_ai1, col_ai2 = st.columns(2)
            with col_ai1:
                ai_img_col = st.selectbox("Image Path Column (optional)", ["None"] + list(data["df"].columns))
                ai_img_col = None if ai_img_col == "None" else ai_img_col
            with col_ai2:
                ai_doc_col = st.selectbox("Document Path Column (optional)", ["None"] + list(data["df"].columns))
                ai_doc_col = None if ai_doc_col == "None" else ai_doc_col
                
            if st.button("🚀 Run AI Multimodal Impute", use_container_width=True):
                with st.spinner("Executing AI Imputation Pipeline (Transformers + Diffusion + Vision)..."):
                    try:
                        res = run_ai_impute(data["df"], image_col=ai_img_col, doc_col=ai_doc_col)
                        st.session_state["ai_cleaned_df"] = res.df_imputed
                        st.success(f"✓ AI Imputation Complete! Filled {res.n_imputed} cells using {res.method}.")
                        
                        if res.warnings:
                            for w in res.warnings: st.warning(w)
                    except Exception as e:
                        st.error(f"AI Imputation Error: {e}")

            if "ai_cleaned_df" in st.session_state:
                st.markdown("##### AI-Imputed Data Preview")
                st.dataframe(st.session_state["ai_cleaned_df"].head(20), use_container_width=True)
                
                # Comparison Plot
                from src.data.advanced_imputer import ImputationVisualizer
                viz = ImputationVisualizer()
                try:
                    fig = viz.plot_distributions(data["df"], st.session_state["ai_cleaned_df"])
                    st.pyplot(fig)
                except: pass

                ai_csv = st.session_state["ai_cleaned_df"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download AI-Imputed CSV",
                    ai_csv,
                    file_name=f"{os.path.splitext(source_name)[0]}_ai_fixed.csv",
                    mime="text/csv"
                )

            st.markdown("---")
            st.markdown("#### Download Cleaned File")
            csv_bytes = clean_data["df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download cleaned CSV",
                data=csv_bytes,
                file_name=f"{os.path.splitext(uploaded.name if uploaded else 'db')[0]}_cleaned.csv",
                mime="text/csv",
            )

    # Lab
    with prep_tabs[1]:
        if not has_data:
            st.info("Please load data to use the Lab.")
        else:
            st.markdown("## 🔬 Lab — Advanced Features")
            st.markdown("---")

            feature = st.selectbox("Choose a feature:", [
                "⚙️ Auto Feature Engineering",
                "🎯 Target Column Detection",
                "🕸️ Correlation Network Graph",
                "🔤 Smart Encoding Advisor",
                "📋 Data Schema Validator",
            ])

            st.markdown("---")

            # ── Auto Feature Engineering ──────────────────────────────────────────────
            if feature == "⚙️ Auto Feature Engineering":
                st.markdown("### ⚙️ Auto Feature Engineering")
                from src.data.preparator import prepare_for_ml

                prepared, log = prepare_for_ml(data, missing_strategy=missing_strategy)

                st.markdown("#### New Features Created")
                original_cols = set(data["df"].columns)
                new_cols = [c for c in prepared["df"].columns if c not in original_cols]

                if new_cols:
                    st.success(f"✓ Created {len(new_cols)} new feature(s)")
                    st.dataframe(prepared["df"][new_cols].head(10), use_container_width=True)
                else:
                    st.info("No new features generated for this dataset.")

                st.markdown("#### Full Prepared Dataset")
                st.dataframe(prepared["df"].head(20), use_container_width=True)

                csv_b = prepared["df"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Prepared Dataset",
                    data=csv_b,
                    file_name=f"{os.path.splitext(source_name)[0]}_features.csv",
                    mime="text/csv",
                )

            # ── Target Column Detection ───────────────────────────────────────────────
            elif feature == "🎯 Target Column Detection":
                st.markdown("### 🎯 Target Column Detection")
                from src.data.ml_readiness import ml_readiness
                from src.data.analyzer import detect_outliers

                outliers = analysis["outliers"]
                ml = ml_readiness(data, outliers)
                df = data["df"]

                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

                if not numeric_cols:
                    st.warning("No numeric columns found for target detection.")
                else:
                    scores = {}
                    for col in numeric_cols:
                        n_unique = df[col].nunique()
                        null_pct = df[col].isna().mean()
                        is_id = n_unique == len(df)
                        score = 0
                        if not is_id:
                            score += 30
                        if null_pct < 0.05:
                            score += 20
                        if n_unique > 5:
                            score += 20
                        if df[col].std() > 0:
                            score += 30
                        scores[col] = score

                    scores_df = pd.DataFrame([
                        {"Column": col, "Score": sc, "Unique": df[col].nunique(),
                         "Null %": f"{df[col].isna().mean():.1%}",
                         "Std": round(float(df[col].std()), 3)}
                        for col, sc in sorted(scores.items(), key=lambda x: -x[1])
                    ])

                    st.dataframe(scores_df, use_container_width=True)

                    best = max(scores, key=scores.get)
                    st.success(f"✓ Recommended target column: **{best}** (score {scores[best]}/100)")

                    fig = px.bar(
                        scores_df, x="Column", y="Score",
                        color="Score",
                        color_continuous_scale=["#f06e6e", "#f0c46e", "#c8f06e"],
                        title="Target Column Scores",
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
                    st.plotly_chart(fig, use_container_width=True)

            # ── Correlation Network Graph ─────────────────────────────────────────────
            elif feature == "🕸️ Correlation Network Graph":
                st.markdown("### 🕸️ Correlation Network Graph")
                from src.data.relationships import detect_relationships

                threshold = st.slider("Minimum correlation strength:", 0.1, 0.9, 0.4, 0.05)
                rels = detect_relationships(data, threshold=threshold)

                if not rels:
                    st.info(f"No relationships found above threshold {threshold}.")
                else:
                    rel_df = pd.DataFrame([{
                        "Column A": r["col_a"],
                        "Column B": r["col_b"],
                        "Strength": r["strength"],
                        "Direction": r["direction"],
                        "Method": r["method"],
                    } for r in rels])
                    st.dataframe(rel_df, use_container_width=True)

                    fig = px.scatter(
                        rel_df, x="Column A", y="Column B",
                        size="Strength", color="Strength",
                        color_continuous_scale=["#f06e6e", "#f0c46e", "#c8f06e"],
                        title="Column Relationships",
                        size_max=40,
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
                    st.plotly_chart(fig, use_container_width=True)

            # ── Smart Encoding Advisor ────────────────────────────────────────────────
            elif feature == "🔤 Smart Encoding Advisor":
                st.markdown("### 🔤 Smart Encoding Advisor")

                col1, col2 = st.columns(2)
                with col1:
                    model_type = st.selectbox("Model type:", ["tree", "linear", "neural"])
                with col2:
                    df = data["df"]
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    target_col = st.selectbox("Target column (optional):", ["None"] + numeric_cols)
                    target_col = None if target_col == "None" else target_col

                result = encoding_advisor(data, target_col=target_col, model_type=model_type)
                st.info(result["summary"])

                for r in result["columns"]:
                    risk_color = "green" if r["risk"] == "low" else "orange" if r["risk"] == "medium" else "red"
                    with st.expander(f"**{r['column']}** → {r['strategy']} ({r['cardinality']} cardinality)"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Unique Values", r["n_unique"])
                        col2.metric("Entropy", r["entropy"])
                        col3.metric("Imbalance", f"{r['imbalance']}x")
                        if r["target_corr"] is not None:
                            st.metric("Target Correlation", r["target_corr"])
                        st.markdown(f"**Reason:** {r['reason']}")
                        st.markdown(f"**sklearn:** `{r['sklearn_tip']}`")
                        if r["warnings"]:
                            for w in r["warnings"]:
                                st.warning(w)
                        st.code(r["code"], language="python")

                st.markdown("#### 📋 Pipeline Code")
                st.code(result["pipeline_code"], language="python")

            # ── Data Schema Validator ─────────────────────────────────────────────────
            elif feature == "📋 Data Schema Validator":
                st.markdown("### 📋 Data Schema Validator")

                mode = st.radio("Mode:", ["Auto-infer schema", "Upload schema JSON"], horizontal=True)

                if mode == "Upload schema JSON":
                    schema_file = st.file_uploader("Upload schema JSON", type=["json"])
                    if schema_file:
                        import json

                        schema = schema_from_dict(json.load(schema_file))
                    else:
                        st.info("Please upload a schema JSON file.")
                        st.stop()
                else:
                    schema = infer_schema(data)
                    schema_json = schema_to_dict(schema)
                    st.markdown("#### Inferred Schema")
                    st.json(schema_json)

                    import json

                    st.download_button(
                        label="⬇️ Download Schema JSON",
                        data=json.dumps(schema_json, indent=2).encode("utf-8"),
                        file_name=f"{os.path.splitext(source_name)[0]}_schema.json",
                        mime="application/json",
                    )

                result = validate_schema(data, schema)

                valid_color = "success" if result["valid"] else "error"
                if result["valid"]:
                    st.success(result["summary"])
                else:
                    st.error(result["summary"])

                for r in result["results"]:
                    if r["status"] == "pass":
                        with st.expander(f"✅ {r['column']}"):
                            for p in r["passed"]:
                                st.markdown(f"✓ {p}")
                    else:
                        with st.expander(f"❌ {r['column']}", expanded=True):
                            for e in r["errors"]:
                                st.error(e)
                            for w in r["warnings"]:
                                st.warning(w)

    # Multi-File
    with prep_tabs[2]:
        st.markdown("### 📂 Multi-File Analysis")
        st.caption("Upload multiple files and compare them side by side.")
        st.markdown("---")

        # ── File Upload ───────────────────────────────────────────────────────────
        multi_files = st.file_uploader(
            "Upload multiple files (CSV, Excel, JSON)",
            type=["csv", "xlsx", "xls", "json"],
            accept_multiple_files=True,
            key="multi_file_uploader",
        )

        if not multi_files:
            st.info("Upload 2 or more files to start comparison.")
        else:
            # ── Load all files ────────────────────────────────────────────────────
            all_data    = []
            all_reports = []
            all_ml      = []
            all_outliers= []
            all_contracts = []

            for f in multi_files:
                try:
                    f.seek(0)
                    ingest = load_bytes_resilient(f.read(), f.name)
                    data_multi = ingest["data"]
                    for warn in ingest["warnings"]:
                        st.warning(f"{f.name}: {warn}")
                    for risk in ingest["risk_flags"]:
                        st.info(f"{f.name}: {risk}")

                    from src.data.analyzer     import full_report, detect_outliers
                    from src.data.ml_readiness import ml_readiness

                    report_multi   = full_report(data_multi)
                    outliers_multi = detect_outliers(data_multi)
                    ml_multi       = ml_readiness(data_multi, outliers_multi)

                    all_data.append(data_multi)
                    all_reports.append(report_multi)
                    all_ml.append(ml_multi)
                    all_outliers.append(outliers_multi)
                    all_contracts.append(evaluate_contracts(data_multi))

                except Exception as e:
                    st.error(f"Error loading {f.name}: {e}")

            if len(all_data) < 1:
                st.warning("No valid files loaded.")
            else:
                # ── Summary Table ─────────────────────────────────────────────────
                st.markdown("#### 📊 Overview Comparison")

                summary_rows = []
                for i, (d, rep, m, outs, cont) in enumerate(
                    zip(all_data, all_reports, all_ml, all_outliers, all_contracts)
                ):
                    summary_rows.append({
                        "File":          d["source"],
                        "Rows":          rep["shape"]["rows"],
                        "Columns":       rep["shape"]["columns"],
                        "Missing":       sum(rep["missing_values"].values()),
                        "Duplicates":    rep["duplicate_rows"],
                        "Outlier Cols":  len(set(r.column for r in outs.method_results if r.column and r.outlier_count > 0)),
                        "ML Score":      m["score"],
                        "Grade":         m["grade"],
                        "Violations":    cont["counts"]["total"],
                    })

                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, use_container_width=True)

                # ── Best file ─────────────────────────────────────────────────────
                best_idx  = summary_df["ML Score"].idxmax()
                best_name = summary_df.loc[best_idx, "File"]
                best_score= summary_df.loc[best_idx, "ML Score"]
                st.success(f"🏆 Best file: **{best_name}** — ML Score {best_score}/100")

                st.markdown("---")

                # ── Charts ────────────────────────────────────────────────────────
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(
                        summary_df, x="File", y="ML Score",
                        color="ML Score",
                        color_continuous_scale=["#f06e6e", "#f0c46e", "#c8f06e"],
                        title="ML Readiness Score",
                        text="Grade",
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e8e6e1",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig2 = px.bar(
                        summary_df, x="File", y="Missing",
                        color="Missing",
                        color_continuous_scale=["#c8f06e", "#f0c46e", "#f06e6e"],
                        title="Missing Values",
                    )
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e8e6e1",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                col3, col4 = st.columns(2)

                with col3:
                    fig3 = px.bar(
                        summary_df, x="File", y="Rows",
                        color="Rows",
                        color_continuous_scale=["#5ce0c6", "#c8f06e"],
                        title="Row Count",
                    )
                    fig3.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e8e6e1",
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                with col4:
                    fig4 = px.bar(
                        summary_df, x="File", y="Outlier Cols",
                        color="Outlier Cols",
                        color_continuous_scale=["#c8f06e", "#f06e6e"],
                        title="Outlier Columns",
                    )
                    fig4.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e8e6e1",
                    )
                    st.plotly_chart(fig4, use_container_width=True)

                st.markdown("---")

                # ── Deep dive per file ────────────────────────────────────────────
                st.markdown("#### 🔍 Deep Dive")
                file_names  = [d["source"] for d in all_data]
                chosen_file = st.selectbox("Select file to inspect", file_names)
                chosen_idx  = file_names.index(chosen_file)
                chosen_data_m = all_data[chosen_idx]
                chosen_rep_m  = all_reports[chosen_idx]
                chosen_ml_m   = all_ml[chosen_idx]
                chosen_contract_m = all_contracts[chosen_idx]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows",        chosen_rep_m["shape"]["rows"])
                c2.metric("Columns",     chosen_rep_m["shape"]["columns"])
                c3.metric("ML Score",    f"{chosen_ml_m['score']}/100")
                c4.metric("Missing",     sum(chosen_rep_m["missing_values"].values()))
                st.metric("Contract violations", chosen_contract_m["counts"]["total"])

                if chosen_contract_m["counts"]["total"] > 0 and st.button("🛠️ Safe-fix selected file", key="multi_safe_fix_selected"):
                    repaired, actions = apply_safe_fixes(chosen_data_m, chosen_contract_m, missing_strategy="median")
                    st.success(f"Applied {len(actions)} action(s) on {chosen_file}.")
                    st.dataframe(repaired["df"].head(20), use_container_width=True)
                    csv_fixed = repaired["df"].to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download selected fixed file",
                        csv_fixed,
                        file_name=f"{os.path.splitext(chosen_file)[0]}_safe_fixed.csv",
                        mime="text/csv",
                        key="multi_safe_fix_download_selected",
                    )

                st.dataframe(chosen_data_m["df"].head(20), use_container_width=True)

                st.markdown("---")

                # ── Merge files ───────────────────────────────────────────────────
                st.markdown("#### 🔗 Merge Files")
                st.caption("Combine all files into one — files must have the same columns.")

                if st.button("🔗 Merge all files", use_container_width=True):
                    try:
                        merged = pd.concat(
                            [d["df"] for d in all_data],
                            ignore_index=True,
                        )
                        st.success(f"✓ Merged {len(all_data)} files → {len(merged):,} rows")
                        st.dataframe(merged.head(20), use_container_width=True)
                        csv = merged.to_csv(index=False).encode()
                        st.download_button(
                            "⬇️ Download merged file",
                            csv,
                            file_name="merged_data.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Merge failed: {e}")

                st.markdown("---")

                # ── Convert to SQLite ─────────────────────────────────────────────
                st.markdown("#### 🗄️ Convert All to SQLite")
                st.caption("Convert all uploaded files into a single optimized SQLite database.")

                if st.button("💾 Convert & Generate DB", use_container_width=True, key="multi_db_convert_btn"):
                    with st.spinner("Building SQLite Database..."):
                        try:
                            # Prepare files for converter: list of (bytes, filename)
                            to_convert = []
                            for f in multi_files:
                                f.seek(0)
                                to_convert.append((f.read(), f.name))
                            
                            db_bytes, report = convert_to_bytes(to_convert)
                            
                            st.success(f"✓ Database generated! {report.total_tables} tables, {report.total_rows:,} rows.")
                            
                            # Show report summary
                            st.dataframe(report.as_df(), use_container_width=True)
                            
                            st.download_button(
                                label="⬇️ Download SQLite Database (.db)",
                                data=db_bytes,
                                file_name="Nydra_converted.db",
                                mime="application/x-sqlite3",
                                key="multi_db_download",
                            )
                        except Exception as e:
                            st.error(f"Conversion failed: {e}")

                st.markdown("---")

                # ── Download all cleaned ──────────────────────────────────────────
                st.markdown("#### ⬇️ Download All Cleaned")

                for i, (f, data_dl) in enumerate(zip(multi_files, all_data)):
                    cleaned, _ = handle_missing(data_dl, strategy=missing_strategy)
                    if remove_dupes:
                        cleaned, _ = remove_duplicates(cleaned)
                    
                    csv = cleaned["df"].to_csv(index=False).encode()
                    st.download_button(
                        label=f"⬇️ {f.name} (cleaned)",
                        data=csv,
                        file_name=f"{os.path.splitext(f.name)[0]}_cleaned.csv",
                        mime="text/csv",
                        key=f"dl_multi_{i}",
                    )

    # Database
    with prep_tabs[3]:
        render_database_connector()
    st.markdown("</div>", unsafe_allow_html=True)

# ── Master Tab 4: AI Audits ──────────────────────────────────────────────────
with master_tabs[3]:
    st.markdown("<div class='glass-card suite-audits'>", unsafe_allow_html=True)
    audit_tabs = st.tabs(["🖼️ Image Audit", "📖 Text Audit", "⚖️ ML Training Audit"])
    
    # Image Audit
    with audit_tabs[0]:
        st.markdown("### 🖼️ Image Audit")
        st.caption("Professional end-to-end image dataset inspection: Quality, Diversity, Bias, and Automated Repair.")
        st.markdown("---")

        if not IMAGE_V6_AVAILABLE:
            st.error("Image modules (vision_nlp) not found or missing dependencies.")
        else:
            img_source = st.radio("Source:", ["Local Directory", "Individual Uploads"], horizontal=True, key="img_source_master")
            
            if img_source == "Local Directory":
                col_path, col_clean = st.columns([3, 1])
                with col_path:
                    target_dir = st.text_input("Directory path:", placeholder="e.g. data/images", key="img_dir_master")
                with col_clean:
                    do_clean = st.checkbox("Auto-Repair 🧹", help="Fix brightness, contrast, blur, etc. during audit.", key="img_clean_master")

                if st.button("▶ Run Professional Audit", key="run_image_audit_dir_master", type="primary", use_container_width=True):
                    if os.path.exists(target_dir):
                        with st.spinner("Executing Image Audit Pipeline..."):
                            orchestrator = AuditOrchestrator(target_dir=target_dir, verbose=False)
                            res = orchestrator.run_full_audit(clean=do_clean)
                            
                            if res["status"] == "success":
                                st.session_state["image_audit_results"] = res
                                st.success(f"✓ Audit complete for {len(res['loader_df'])} images!")
                            elif res["status"] == "empty":
                                st.warning(res["message"])
                            else:
                                st.error(f"Audit failed: {res.get('message')}")
                    else:
                        st.error("Directory not found.")
            else:
                uploaded_imgs = st.file_uploader("Upload images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="img_upload_master")
                if uploaded_imgs and st.button("▶ Run Audit on Uploads", key="run_image_audit_upload_master", type="primary", use_container_width=True):
                    st.info("Upload-based audit is coming soon. Please use 'Local Directory' for now.")

            if "image_audit_results" in st.session_state:
                res = st.session_state["image_audit_results"]
                
                # Metrics
                st.markdown("#### 📈 Dataset Overview")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Images Found", len(res["loader_df"]))
                
                q_score = res['quality_report'].mean_overall_score
                c2.metric("Mean Quality", f"{q_score:.1f}/100")
                
                n_classes = res["loader_df"]["class_name"].nunique() if "class_name" in res["loader_df"].columns else "N/A"
                c3.metric("Classes", n_classes)
                
                if res.get("cleaner_df") is not None:
                    n_cleaned = len(res["cleaner_df"][res["cleaner_df"]["cleaned"] == True])
                    c4.metric("Cleaned 🧹", n_cleaned)
                else:
                    c4.metric("Cleaning", "Disabled")

                # Quality Distribution
                st.markdown("#### 📊 Quality Distribution")
                fig = px.histogram(res["quality_df"], x="overall_score", nbins=20, 
                                   title="Overall Quality Score Distribution", 
                                   color_discrete_sequence=["#5ce0c6"],
                                   labels={"overall_score": "Quality Score (0-100)"})
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
                st.plotly_chart(fig, use_container_width=True)

                # Deep Insights from ImageAnalyzer
                if "analyzer_report" in res:
                    report = res["analyzer_report"]
                    st.markdown("#### 🧠 Deep Insights")
                    
                    c1, c2 = st.columns([1, 1])
                    
                    with c1:
                        if report.ml_readiness:
                            ml_i = report.ml_readiness
                            st.markdown(f"**ML Readiness Grade:** `{ml_i.grade}`")
                            st.metric("Readiness Score", f"{ml_i.final_score:.1f}/100", delta=ml_i.verdict)
                            if ml_i.priority_issues:
                                st.warning("**Priority Issues:**\n" + "\n".join([f"- {i}" for i in ml_i.priority_issues[:3]]))

                    with c2:
                        if report.class_balance:
                            cb = report.class_balance
                            fig_cb = px.pie(names=list(cb.class_counts.keys()), 
                                            values=list(cb.class_counts.values()), 
                                            title="Class Distribution",
                                            hole=0.4)
                            fig_cb.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1", showlegend=False)
                            st.plotly_chart(fig_cb, use_container_width=True)

                    if report.diversity_redundancy:
                        div = report.diversity_redundancy
                        col_div, col_red = st.columns(2)
                        col_div.info(f"**Diversity Index (Vendi):** `{div.vendi_score:.2f}`")
                        if div.near_duplicate_clusters:
                            col_red.error(f"**Redundancy:** Found {len(div.near_duplicate_clusters)} groups of duplicates!")
                        else:
                            col_red.success("**Redundancy:** No duplicates detected.")

                # Advanced Details
                with st.expander("🔍 View Detailed Audit Results"):
                    det_tab1, det_tab2, det_tab3 = st.tabs(["Quality Details", "Dataset Analysis", "Cleaning Log"])
                    
                    with det_tab1:
                        st.dataframe(res["quality_df"].head(100), use_container_width=True)
                    
                    with det_tab2:
                        st.json(res["analyzer_df"].to_dict(orient="records")[0] if not res["analyzer_df"].empty else {})
                    
                    with det_tab3:
                        if res.get("cleaner_df") is not None:
                            st.dataframe(res["cleaner_df"][res["cleaner_df"]["cleaned"] == True], use_container_width=True)
                        else:
                            st.info("Cleaning was not requested for this audit.")

                # Reports
                st.markdown("#### 🚀 Professional Reports")
                col_rep, col_dl = st.columns([1, 2])
                
                with col_rep:
                    if st.button("Generate Reports", key="gen_img_reports_master"):
                        with st.spinner("Generating PDF, HTML and JSON reports..."):
                            orchestrator = AuditOrchestrator(target_dir=target_dir, verbose=False)
                            orchestrator.results = res # Inject results
                            report_paths = orchestrator.generate_reports(output_dir="reports")
                            st.session_state["image_report_paths"] = report_paths
                
                with col_dl:
                    if "image_report_paths" in st.session_state:
                        paths = st.session_state["image_report_paths"]
                        dl_cols = st.columns(len(paths))
                        for idx, (fmt, p) in enumerate(paths.items()):
                            with open(p, "rb") as f:
                                dl_cols[idx].download_button(
                                    label=f"⬇️ {fmt.upper()}", 
                                    data=f, 
                                    file_name=os.path.basename(p), 
                                    key=f"dl_img_v6_{fmt}_master"
                                )

    # Text Audit
    with audit_tabs[1]:
        st.markdown("### 📖 Advanced Text Audit & Intelligence")
        st.caption("Multimodal text inspection: Sentiment, Readability, Keywords, PII Detection, and Noise Analysis.")
        st.markdown("---")

        text_input_mode = st.radio("Input Type:", ["Text Column from Dataset", "Raw Text Input"], horizontal=True, key="text_audit_mode_master")

        if text_input_mode == "Text Column from Dataset":
            if not has_data:
                st.info("Please load a dataset in the sidebar first.")
            else:
                df_t = data["df"]
                text_cols = [c for c in df_t.columns if not pd.api.types.is_numeric_dtype(df_t[c])]
                if not text_cols:
                    st.warning("No text columns found in this dataset.")
                else:
                    selected_col_t = st.selectbox("Select Text Column to Analyze:", text_cols, key="text_audit_col_select_master")
                    if st.button("▶ Run Full Text Audit", type="primary", use_container_width=True, key="run_text_audit_col_master"):
                        with st.spinner("Analyzing text corpus..."):
                            from src.vision_nlp.text_orchestrator import TextOrchestrator
                            orchestrator_t = TextOrchestrator()
                            sample_df_t = df_t.dropna(subset=[selected_col_t])
                            if len(sample_df_t) > 50:
                                sample_df_t = sample_df_t.sample(50)
                            
                            results_list = []
                            for val in sample_df_t[selected_col_t].astype(str):
                                results_list.append(orchestrator_t.analyze_text(val))
                            
                            st.session_state["text_audit_results"] = results_list
                            st.session_state["text_audit_column"] = selected_col_t
                            st.success(f"✓ Audited {len(results_list)} rows from '{selected_col_t}'")

        elif text_input_mode == "Raw Text Input":
            raw_text = st.text_area("Paste text here:", placeholder="e.g. I love this product...", height=200, key="text_audit_raw_input_master")
            if st.button("▶ Analyze Text", type="primary", use_container_width=True, key="run_text_audit_raw_master"):
                if raw_text:
                    with st.spinner("Analyzing..."):
                        from src.vision_nlp.text_orchestrator import TextOrchestrator
                        res_t = TextOrchestrator().analyze_text(raw_text)
                        st.session_state["text_audit_results"] = [res_t]
                        st.session_state["text_audit_column"] = "Manual Input"
                else:
                    st.error("Please enter some text.")

        if "text_audit_results" in st.session_state:
            results_t = st.session_state["text_audit_results"]
            st.markdown("---")
            
            if len(results_t) == 1:
                res_t = results_t[0]
                if res_t["status"] == "success":
                    c1, c2, c3 = st.columns(3)
                    intel_t = res_t["intelligence"]
                    with c1:
                        st.markdown("#### 🧠 Intelligence")
                        if "sentiment" in intel_t:
                            s_t = intel_t["sentiment"]["doc_level"]
                            st.metric("Sentiment", s_t.get("dominant", "Unknown"), delta=round(s_t.get("confidence", 0), 2))
                        if "readability" in intel_t:
                            st.markdown(f"**Readability:** `{intel_t['readability'].get('grade_label', 'Unknown')}`")

                    qual_t = res_t["quality"]
                    with c2:
                        st.markdown("#### 🔍 Quality")
                        st.metric("Quality Score", f"{qual_t['overall_score']:.1f}/100", delta=qual_t.get("verdict", "Unknown"))

                    with c3:
                        st.markdown("#### 🏷️ Key Topics")
                        if "keywords" in intel_t:
                            for kw in [k[0] for k in intel_t["keywords"][:5]]:
                                st.markdown(f"- {kw}")
                else:
                    st.error(f"Analysis failed: {res_t.get('message')}")
            else:
                scores_t = [r["quality"]["overall_score"] for r in results_t if r["status"] == "success"]
                avg_s_t = sum(scores_t)/len(scores_t) if scores_t else 0
                c1, c2 = st.columns(2)
                c1.metric("Avg Quality", f"{avg_s_t:.1f}/100")
                c2.metric("Total Rows", len(results_t))
                fig = px.histogram(x=scores_t, nbins=10, title="Quality Score Distribution", labels={'x': 'Score'})
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
                st.plotly_chart(fig, use_container_width=True)

    # ML Training Audit
    with audit_tabs[2]:
        st.markdown("### ⚖️ ML Training Data Audit")
        st.caption("Deep inspection of training/test splits for Bias, Data Leakage, and Label Quality.")
        st.markdown("---")

        if not has_data:
            st.info("Please load a dataset in the sidebar first.")
        else:
            df_a = data["df"]
            c1, c2 = st.columns(2)
            target_col_a = c1.selectbox("Select Target (Label) Column:", df_a.columns, key="audit_train_target_master")
            sensitive_cols_a = c2.multiselect("Select Sensitive Columns (for Bias Audit):", [c for c in df_a.columns if c != target_col_a], key="audit_train_sensitive_master")

            split_mode_a = st.radio("Test Data Source:", ["Auto-Split (80/20)", "Upload Test File"], horizontal=True, key="split_mode_audit_master")
            test_df_a = None
            if split_mode_a == "Auto-Split (80/20)":
                from sklearn.model_selection import train_test_split
                train_df_a, test_df_a = train_test_split(df_a, test_size=0.2, random_state=42)
            else:
                test_file_a = st.file_uploader("Upload Test CSV:", type=["csv"], key="test_file_audit_master")
                if test_file_a:
                    test_df_a = pd.read_csv(test_file_a)
                    train_df_a = df_a

            if st.button("▶ Run Full Training Audit", type="primary", use_container_width=True, key="run_training_audit_btn_master"):
                if test_df_a is not None:
                    with st.spinner("Running comprehensive audit (this may take a minute)..."):
                        doctor = Nydra()
                        res_a = doctor.audit_training_data(train_df_a, test_df_a, target_col_a, sensitive_cols_a)
                        st.session_state["training_audit_results"] = res_a
                else:
                    st.error("Please provide test data.")

        if "training_audit_results" in st.session_state:
            res_a = st.session_state["training_audit_results"]
            st.markdown("---")
            
            # 1. Summary Metrics
            summary_a = res_a.get("summary", {})
            c1, c2, c3 = st.columns(3)
            
            lq_score_a = summary_a.get("label_quality_score", 0)
            lk_score_a = summary_a.get("leakage_score", 0)
            bs_score_a = summary_a.get("bias_score", 100)
            
            c1.metric("Label Quality", f"{lq_score_a:.1f}/100")
            c2.metric("Leakage Risk", summary_a.get("leakage_risk", "N/A"), delta=f"{lk_score_a:.1f} score", delta_color="inverse")
            if "bias_score" in summary_a:
                c3.metric("Bias Score", f"{bs_score_a:.1f}/100", delta=summary_a.get("bias_verdict", "N/A"))

            # 2. Detailed Findings
            tab_lq, tab_lk, tab_bs = st.tabs(["🏷️ Label Quality", "🛡️ Data Leakage", "⚖️ Fairness & Bias"])
            
            with tab_lq:
                if "label_quality" in res_a:
                    lq_a = res_a["label_quality"]
                    st.markdown(f"#### Overall Score: `{lq_a.label_quality_score:.1f}/100`")
                    if lq_a.recommendations:
                        for rec in lq_a.recommendations:
                            st.warning(f"**[{rec.get('area', 'General')}]** {rec.get('issue', 'Issue detected')}\n\n*Action:* {rec.get('recommendation', 'Investigate')}")
                    else:
                        st.success("No critical label quality issues found.")
                else:
                    st.info("Label quality audit failed or not run.")

            with tab_lk:
                if "leakage" in res_a:
                    lk_a = res_a["leakage"]
                    score_a = lk_a.get("score", {})
                    st.markdown(f"#### Risk Level: `{score_a.get('label', 'Unknown')}`")
                    
                    leaking_features = [f.get("feature", "Unknown") for f in lk_a.get("all_findings", []) if f.get("feature") != "__overlap__"]
                    if leaking_features:
                        st.error(f"**Leaking Features detected:** {', '.join(set(leaking_features))}")
                    
                    overlap_results = lk_a.get("detector_results", {}).get("TrainTestOverlapDetector", {})
                    overlap_rate = overlap_results.get("exact_overlap_ratio", 0)
                    if overlap_rate > 0:
                        st.error(f"**Train/Test Overlap:** `{overlap_rate:.2%}` of test samples found in training set!")
                    
                    if not leaking_features and overlap_rate == 0:
                        st.success("No significant data leakage detected.")
                else:
                    st.info("Leakage audit failed or not run.")

            with tab_bs:
                if "bias" in res_a:
                    bs_a = res_a["bias"]
                    score_ba = bs_a.get("bias_score", {})
                    st.markdown(f"#### Fairness Verdict: `{score_ba.get('verdict', 'Unknown')}`")
                    
                    biased_classes = [f.get("attribute", "Unknown") for f in bs_a.get("findings", [])]
                    if biased_classes:
                        st.warning(f"**Potential Bias detected in:** {', '.join(set(biased_classes))}")
                    
                    per_detector_a = bs_a.get("per_detector", {})
                    if "StatisticalFairnessMetrics" in per_detector_a:
                        st.markdown("**Disparity Metrics by Group:**")
                        st.json(per_detector_a["StatisticalFairnessMetrics"])
                else:
                    st.info("Bias audit failed or no sensitive columns provided.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Master Tab 5: Intelligence ───────────────────────────────────────────────
with master_tabs[4]:
    st.markdown("<div class='glass-card suite-intel'>", unsafe_allow_html=True)
    intel_tabs = st.tabs(["💬 Data Chat", "🧠 Analyst", "🚀 ML Pipeline", "📄 Report"])
    
    # Data Chat
    with intel_tabs[0]:
        if not has_data:
            st.info("Please load data to Chat with your data.")
        else:
            # ── Session State Init ────────────────────────────────────────────
            session_id = user_email or "anonymous"
            if "chatbot" not in st.session_state:
                st.session_state.chatbot = get_chatbot(session_id)
            
            # Ensure data is loaded (only clears if data changes)
            st.session_state.chatbot.load_data(data)

            # ── Header ────────────────────────────────────────────────────────
            col_chat1, col_chat2, col_chat3 = st.columns([3, 1, 1])
            with col_chat1:
                st.markdown("#### 💬 AI Data Analyst")
            with col_chat2:
                if st.button("🗑️ Clear", use_container_width=True, key="clear_chat_v3"):
                    st.session_state.chatbot.clear_history()
                    st.rerun()
            with col_chat3:
                history_md = st.session_state.chatbot.export_history("markdown")
                st.download_button("⬇️ Export", data=history_md, file_name=f"chat_{session_id}.md", use_container_width=True, key="export_chat_v3")

            # ── Render Chat History ───────────────────────────────────────────
            for i, msg in enumerate(st.session_state.chatbot.get_history()):
                with st.chat_message(msg.role):
                    st.markdown(msg.content)
                    
                    if msg.result and msg.result.success:
                        r = msg.result
                        # Visuals
                        if r.chart_type == "bar" and isinstance(r.data, dict):
                            fig = px.bar(x=list(r.data.keys()), y=list(r.data.values()), color_discrete_sequence=["#c8f06e"])
                            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1", height=300)
                            st.plotly_chart(fig, use_container_width=True, key=f"chat_chart_{i}")
                        elif isinstance(r.data, pd.DataFrame):
                            st.dataframe(r.data, use_container_width=True)
                        # Code
                        if r.code:
                            with st.expander("📄 View AI Reasoning"):
                                st.code(r.code, language="python")

            # ── Chat Input ────────────────────────────────────────────────────
            if prompt := st.chat_input("Ask about your data..."):
                # Immediately show user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Process
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Get AI Config
                        use_key = st.session_state.get("llm_api_key")
                        provider = st.session_state.get("llm_provider", "groq")
                        model = st.session_state.get("llm_model")
                        from src.data.data_chat import make_config
                        cfg = make_config(provider=provider, model=model, api_key=use_key)
                        
                        # Run analysis
                        result = st.session_state.chatbot.chat_auto(prompt, cfg)
                        st.markdown(result.answer)
                        
                        # Rerun to update history properly
                        st.rerun()

            # ── Suggestions ───────────────────────────────────────────────────
            if not st.session_state.chatbot.get_history():
                suggestions = st.session_state.chatbot.get_suggested_questions()
                st.markdown("<div style='margin-top: 2rem'></div>", unsafe_allow_html=True)
                cols = st.columns(min(len(suggestions), 3))
                for i, q in enumerate(suggestions[:3]):
                    if cols[i].button(f"💡 {q}", key=f"suggest_{i}_v3", use_container_width=True):
                        st.session_state.pending_chat_prompt = q
                        st.rerun()

            if "pending_chat_prompt" in st.session_state:
                p = st.session_state.pop("pending_chat_prompt")
                use_key = st.session_state.get("llm_api_key")
                provider = st.session_state.get("llm_provider", "groq")
                model = st.session_state.get("llm_model")
                from src.data.data_chat import make_config
                cfg = make_config(provider=provider, model=model, api_key=use_key)
                st.session_state.chatbot.chat_auto(p, cfg)
                st.rerun()

    # Analyst
    with intel_tabs[1]:
        if not has_data:
            st.info("Please load data to run Analyst mode.")
        elif not analyst_report:
            st.warning("Analyst report is not available for this dataset.")
        else:
            risk_a = analyst_report["overall_risk"]
            risk_color_a = {"low": "#c8f06e", "medium": "#f0c46e", "high": "#f06e6e"}.get(risk_a, "#7a7875")

            st.markdown("### 🧠 Human Analyst Simulation")
            st.caption("Safety-first reasoning engine: observe -> hypothesize -> test -> decide")

            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:0.5px solid rgba(255,255,255,0.08);
                        border-radius:12px;padding:1rem 1.2rem;margin:0.5rem 0 1rem'>
                <div style='font-family:DM Mono,monospace;color:#7a7875;font-size:11px'>EXECUTIVE SUMMARY</div>
                <div style='color:#e8e6e1;font-size:14px;margin-top:6px'>{analyst_report["executive_summary"]}</div>
                <div style='margin-top:10px;color:{risk_color_a};font-family:DM Mono,monospace;font-size:12px'>
                    Overall Risk: {risk_a.upper()}
                </div>
                <div style='margin-top:4px;color:#9a9792;font-size:12px'>{analyst_report["preferred_mode"]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Reasoning Steps")
            for step in analyst_report["reasoning_steps"]:
                st.markdown(f"- {step}")

            st.markdown("#### Findings")
            for f in analyst_report["findings"]:
                sev_a = f["severity"]
                icon_a = "🚨" if sev_a == "high" else "⚠️" if sev_a == "medium" else "ℹ️"
                color_a = "#f06e6e" if sev_a == "high" else "#f0c46e" if sev_a == "medium" else "#c8f06e"
                st.markdown(
                    f"{icon_a} **{f['title']}**  \n"
                    f"{f['detail']}  \n"
                    f"<span style='color:{color_a};font-family:DM Mono,monospace;font-size:11px'>"
                    f"{sev_a.upper()} · confidence {f['confidence']}%</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")

            st.markdown("#### Recommended Actions")
            for idx, item in enumerate(analyst_report["action_plan"], start=1):
                sev_a = item["severity"]
                icon_a = "🔴" if sev_a == "high" else "🟡" if sev_a == "medium" else "🟢"
                st.markdown(f"{idx}. {icon_a} **{item['action']}**  \n{item['why']}")

    # ML Pipeline
    with intel_tabs[2]:
        if not has_data:
            st.info("Please load data to use ML Pipeline.")
        else:
            st.markdown("#### 🚀 ML Pipeline ")
            st.markdown("---")

            target_col_p = st.selectbox(
                "Select target column",
                options=data["df"].columns.tolist(),
                index=len(data["df"].columns) - 1,
                key="target_col_master"
            )

            task_type_p = st.selectbox(
                "Task type",
                ["auto", "classification", "regression"],
                key="task_type_master"
            )

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 🚀 Advanced Auto ML")
                st.caption("Bayesian Optimization (Optuna) + ASHA + Stacking")
                
                use_stacking = st.checkbox("Enable Model Stacking (Ensemble)", value=True, key="stacking_master")
                time_limit = st.slider("Time limit (seconds)", 60, 600, 300, 60, key="timeout_master")
                
                if st.button("▶ Run Advanced Auto ML", key="run_automl_master"):
                    if user_email:
                        allowed, reason = security.check_request(user_email, "run_automl", source_name)
                        if not allowed:
                            st.error(reason)
                            st.stop()
                    
                    with st.spinner("Optimizing 15+ models with Optuna..."):
                        try:
                            ml_result = run_advanced_automl(data, target_column=target_col_p, timeout=time_limit, enable_stacking=use_stacking)
                            st.success(f"🏆 Best Model: **{ml_result['best_model']}**")
                            c1, c2 = st.columns(2)
                            c1.metric("Best Score", f"{ml_result['best_score']:.4f}")
                            c2.metric("Trials Run", ml_result.get("n_trials", "N/A"))
                            st.info(f"**Recommendation:** {ml_result['recommendation']}")
                            
                            st.markdown("##### Model Leaderboard")
                            results_df = pd.DataFrame(ml_result["leaderboard"])
                            st.dataframe(results_df, use_container_width=True)
                            
                            fig = px.bar(results_df.head(10), x="model", y="score", color="score", color_continuous_scale="Viridis", title="Top 10 Model Performance")
                            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            if "pipeline_code" in ml_result:
                                st.download_button("⬇️ Download Best Pipeline (.py)", ml_result["pipeline_code"], file_name="best_model_pipeline.py", mime="text/plain", key="dl_pip_master")
                        except Exception as e:
                            st.error(f"AutoML Error: {e}")

            with col2:
                st.markdown("##### Feature Importance")
                if st.button("▶ Compute Importance", key="compute_fi_master"):
                    with st.spinner("Computing SHAP/Gini importance..."):
                        try:
                            from src.data.feature_importance import compute_feature_importance
                            fi_result = compute_feature_importance(data, target_col=target_col_p, task_type=task_type_p)
                            st.success(f"🏆 Top feature: **{fi_result['top_feature']}**")
                            fi_df = pd.DataFrame(fi_result["features"])
                            fig = px.bar(fi_df, x="pct", y="feature", orientation="h", color="pct", color_continuous_scale=["#f0c46e", "#c8f06e"], title="Feature importance (%)")
                            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e8e6e1", yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")

    # Report
    with intel_tabs[3]:
        if not has_data:
            st.info("Please load data to generate a report.")
        else:
            st.markdown("#### 📄 Smart Report Generator")
            col1, col2 = st.columns(2)
            dataset_name = col1.text_input("Dataset Name", value=source_name or "Dataset", key="ds_name_master")
            formats = col2.multiselect("Formats", ["pdf", "docx", "pptx"], default=["pdf", "docx", "pptx"], key="fmts_master")
            
            if st.button("🚀 Generate Reports", type="primary", key="gen_rep_master"):
                with st.spinner("Generating reports..."):
                    from src.data.quality_score import DataQualityScorer
                    from src.smart_report import SmartReport
                    profile_r = DataQualityScorer().score(data["df"])
                    reporter = SmartReport(output_dir="reports")
                    paths_r = reporter.generate_all(profile_r, data["df"], dataset_name, formats=formats)
                
                st.success(f"✅ {len(paths_r)} report(s) generated!")
                for fmt, path in paths_r.items():
                    with open(path, "rb") as f:
                        st.download_button(label=f"⬇️ {fmt.upper()}", data=f, file_name=f"{dataset_name}_quality.{fmt}", mime="application/octet-stream", key=f"dl_{fmt}_master")
    st.markdown("</div>", unsafe_allow_html=True)
