"""
ux.py — UX Components for Nydra

Features:
    - Welcome page with guided tour
    - Onboarding flow for new users
    - Tooltips and help system
    - Progress indicators
    - Sample data quick-start
    - Responsive feedback system
"""

from __future__ import annotations
import streamlit as st
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Welcome Page
# ══════════════════════════════════════════════════════════════════════════════

def render_welcome() -> None:
    """Render the welcome page when no file is uploaded."""
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem 2rem'>
        <div style='font-family:Fraunces,serif;font-size:3.5rem;font-weight:300;
                    color:var(--text-color);line-height:1.2'>
            Drop your data.<br>
            <em style='color:#c8f06e'>Get answers.</em>
        </div>
        <div style='font-size:1.1rem;color:rgba(255,255,255,0.5);
                    margin-top:1.5rem;font-family:DM Sans,sans-serif;font-weight:300'>
            The autonomous data inspection agent that cleans, analyses,<br>
            and prepares your data for Machine Learning — in seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3)

    features = [
        ("🔍", "Instant Inspection",
         "Missing values, duplicates, outliers, and statistics — all at once."),
        ("🧹", "Auto Cleaning",
         "Fill missing values, remove duplicates, and export clean data ready for use."),
        ("🤖", "ML Pipeline",
         "Auto ML baseline, feature importance, and a complete sklearn pipeline export."),
        ("🗄️", "Database Connector",
         "Connect to PostgreSQL, MySQL, or SQLite and analyse tables directly."),
        ("🧬", "Cognitive DNA",
         "A unique fingerprint of your dataset — detect similarity and drift over time."),
        ("💡", "AI Suggestions",
         "Connect any LLM (Groq, Gemini, OpenAI) for personalised data advice."),
    ]

    for i, (icon, title, desc) in enumerate(features):
        col = cols[i % 3]
        col.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:0.5px solid rgba(255,255,255,0.08);
                    border-radius:12px;padding:1.25rem;margin-bottom:12px;height:140px'>
            <div style='font-size:1.5rem;margin-bottom:8px'>{icon}</div>
            <div style='font-weight:500;font-size:14px;color:#e8e6e1;margin-bottom:6px'>{title}</div>
            <div style='font-size:12px;color:rgba(255,255,255,0.45);line-height:1.5'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Quick start
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;font-family:DM Mono,monospace;font-size:12px;
                color:rgba(255,255,255,0.3)'>
        ← Upload a file in the sidebar to get started &nbsp;|&nbsp;
        CSV · Excel · JSON · up to 1GB
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Privacy Tab
# ══════════════════════════════════════════════════════════════════════════════

def render_privacy_tab(privacy_report: dict, df: Any) -> None:
    """Render the privacy assessment tab."""
    from src.security import PrivacyManager, security

    pm = PrivacyManager()

    risk = privacy_report.get("risk_level", "low")
    risk_colors = {"low": "success", "medium": "warning", "high": "error"}
    risk_icons  = {"low": "✅", "medium": "⚠️", "high": "🚨"}

    getattr(st, risk_colors.get(risk, "info"))(
        f"{risk_icons.get(risk, '')} Privacy Risk: **{risk.upper()}** — "
        f"{privacy_report.get('affected_cols', 0)} column(s) with potential PII"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔍 PII Detection")
        pii_found = privacy_report.get("pii_found", {})

        if not pii_found:
            st.success("No PII detected in your dataset.")
        else:
            for pii_type, cols in pii_found.items():
                st.warning(f"**{pii_type.upper()}** found in: `{', '.join(cols)}`")

        st.markdown("---")
        st.markdown("#### 💡 Recommendations")
        recs = privacy_report.get("recommendations", [])
        if not recs:
            st.info("Your data looks clean — no privacy actions needed.")
        else:
            for rec in recs:
                st.markdown(f"• {rec}")

    with col2:
        st.markdown("#### 🛡️ Anonymization")

        if pii_found:
            all_pii_cols = list({col for cols in pii_found.values() for col in cols})
            selected_col = st.selectbox("Select column to anonymize", all_pii_cols)
            method = st.selectbox(
                "Anonymization method",
                ["hash", "mask", "fake_id", "drop"],
                help="hash=SHA256, mask=***, fake_id=ID_000001, drop=remove column"
            )

            if st.button("🛡️ Anonymize Column"):
                import pandas as pd
                df_copy = df.copy()
                df_copy[selected_col] = pm.anonymize_column(df_copy[selected_col], method)
                st.success(f"✓ Column '{selected_col}' anonymized using {method}")
                csv = df_copy.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download anonymized data",
                    data=csv,
                    file_name="data_anonymized.csv",
                    mime="text/csv",
                )
        else:
            st.info("No PII columns detected — anonymization not needed.")

    st.markdown("---")
    st.markdown("#### 🔒 Privacy Policy")
    st.markdown("""
    <div style='background:rgba(255,255,255,0.03);border:0.5px solid rgba(255,255,255,0.08);
                border-radius:12px;padding:1.25rem;font-family:DM Mono,monospace;font-size:12px;
                color:rgba(255,255,255,0.5);line-height:1.8'>
    ✓ &nbsp;Your data is processed <strong style='color:#c8f06e'>in memory only</strong><br>
    ✓ &nbsp;Nothing is stored on our servers<br>
    ✓ &nbsp;Files are deleted when your session ends<br>
    ✓ &nbsp;We never share your data with third parties<br>
    ✓ &nbsp;API keys are encrypted and never logged<br>
    ✓ &nbsp;Only usage metadata is stored (file name, action, timestamp)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Audit Tab
# ══════════════════════════════════════════════════════════════════════════════

def render_audit_tab(user_email: str) -> None:
    """Render the audit log tab for the current user."""
    from src.security import security
    import pandas as pd

    st.markdown("#### 📋 Your Activity Log")

    events = security.audit_logger.get_user_events(user_email)

    if not events:
        st.info("No activity recorded yet. Start by uploading a file!")
        return

    df = pd.DataFrame([{
        "Time":   e.timestamp,
        "Action": e.action,
        "File":   e.file_name or "—",
        "Status": "✅" if e.success else "❌",
    } for e in reversed(events[-50:])])

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Showing last {min(50, len(events))} of {len(events)} events")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar User Profile
# ══════════════════════════════════════════════════════════════════════════════

def render_user_profile(user_stats: dict, session: Any) -> None:
    """Render user profile in sidebar."""
    if not user_stats:
        return

    st.markdown(f"""
    <div style='background:rgba(200,240,110,0.05);border:0.5px solid rgba(200,240,110,0.2);
                border-radius:10px;padding:0.75rem;font-family:DM Mono,monospace;font-size:11px'>
        <div style='color:#c8f06e;font-weight:500;margin-bottom:6px'>
            👤 {user_stats.get('name', 'User')}
        </div>
        <div style='color:rgba(255,255,255,0.4);line-height:1.8'>
            Plan&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {user_stats.get('plan', 'free').upper()}<br>
            Sessions : {user_stats.get('total_sessions', 0)}<br>
            Files&nbsp;&nbsp;&nbsp;&nbsp;: {user_stats.get('files_analysed', 0)}<br>
            Avg ML&nbsp;&nbsp; : {user_stats.get('avg_ml_score', 0)}/100<br>
            Since&nbsp;&nbsp;&nbsp;&nbsp;: {user_stats.get('member_since', '—')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if session:
        st.markdown(f"""
        <div style='font-family:DM Mono,monospace;font-size:10px;
                    color:rgba(255,255,255,0.25);margin-top:6px;text-align:right'>
            Session: {session.duration_minutes}min active
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Notification System
# ══════════════════════════════════════════════════════════════════════════════

def notify_success(message: str, detail: str = "") -> None:
    st.success(f"✓ {message}" + (f"\n{detail}" if detail else ""))


def notify_warning(message: str, detail: str = "") -> None:
    st.warning(f"⚠ {message}" + (f"\n{detail}" if detail else ""))


def notify_error(message: str, detail: str = "") -> None:
    st.error(f"✗ {message}" + (f"\n{detail}" if detail else ""))


def notify_info(message: str) -> None:
    st.info(f"ℹ {message}")


# ══════════════════════════════════════════════════════════════════════════════
# Progress Tracker
# ══════════════════════════════════════════════════════════════════════════════

def render_analysis_progress(steps: list[str]) -> Any:
    """Render animated progress for multi-step analysis."""
    progress_bar = st.progress(0)
    status_text  = st.empty()

    def update(step_idx: int, message: str = "") -> None:
        pct = int((step_idx + 1) / len(steps) * 100)
        progress_bar.progress(pct)
        label = message or steps[step_idx]
        status_text.markdown(
            f"<div style='font-family:DM Mono,monospace;font-size:12px;"
            f"color:rgba(255,255,255,0.5)'>{label}...</div>",
            unsafe_allow_html=True
        )

    def complete() -> None:
        progress_bar.progress(100)
        status_text.empty()

    return update, complete


# ══════════════════════════════════════════════════════════════════════════════
# Help Tooltips
# ══════════════════════════════════════════════════════════════════════════════

TOOLTIPS = {
    "ml_score":      "Score from 0-100 indicating how ready your data is for ML training.",
    "missing":       "Cells with no value. High missing rates reduce ML model accuracy.",
    "duplicates":    "Identical rows that can bias your model during training.",
    "outliers":      "Values far outside the normal range that can distort model learning.",
    "drift":         "Statistical changes between two versions of the same dataset.",
    "dna":           "A unique fingerprint capturing the statistical identity of your dataset.",
    "smote":         "Synthetic Minority Over-sampling: generates artificial samples for minority class.",
    "stratify":      "Preserves class proportions when splitting data into train/test sets.",
    "shap":          "SHapley Additive exPlanations: explains why a model made a specific prediction.",
    "iqr":           "Interquartile Range: robust method to detect outliers using quartile distances.",
    "mcar":          "Missing Completely At Random: no pattern in which values are missing.",
    "mar":           "Missing At Random: missingness depends on other observed variables.",
    "mnar":          "Missing Not At Random: missingness depends on the missing value itself.",
}


def help_text(key: str) -> str:
    return TOOLTIPS.get(key, "")
