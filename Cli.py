"""
cli.py — Nydra Full CLI
==============================
Covers every operation the agent can perform:

  python cli.py inspect    <file>
  python cli.py clean      <file>
  python cli.py stats      <file>
  python cli.py missing    <file>
  python cli.py duplicates <file>
  python cli.py outliers   <file>
  python cli.py export     <file>
  python cli.py report     <file>
  python cli.py ml         <file>
  python cli.py prepare    <file>
  python cli.py relations  <file>
  python cli.py suggest    <file>
  python cli.py memory     [list|compare <file>|clear]
  python cli.py dna        <file>
  python cli.py image      <directory>
  python cli.py drift      <baseline> <current>
  python cli.py interactive
"""

from __future__ import annotations
import pandas as pd
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


from src.data.advanced_outlier import OutlierAnalyzer
from src.data.advanced_outlier import detect_outliers, full_outlier_report
from src.data.db_connector import build_connector
from src.data.db_query     import DBSession
from src.data.dna_memory import get_dna_manager
from src.data.cognitive_dna import CognitiveDNA
from src.data.pipeline_export import export_pipeline
from src.data.imbalance_detector import detect_imbalance
from src.data.split_advisor import advise_split
from src.data.feature_importance import compute_feature_importance
from src.data.advanced_automl import run_automl as run_advanced_automl
from src.data.advanced_imputer import ai_impute as run_ai_impute, SmartImputer
from src.data.schema_validator import validate_schema, infer_schema, schema_to_dict, schema_from_dict
from src.data.encoding_advisor import encoding_advisor
from src.data.correlation_network import build_correlation_network
from src.data.target_detector import detect_target
from src.data.feature_engineer import engineer_features
from src.data.loader        import load_csv
from src.data.analyzer      import full_report, shape, missing_values, duplicate_rows, basic_stats, detect_outliers
from src.data.cleaner       import handle_missing, remove_duplicates
from src.core.agent         import Nydra
from src.report             import generate_html_report
from src.data.ml_readiness  import ml_readiness
from src.data.relationships import detect_relationships
from src.data.preparator    import prepare_for_ml
from src.data.drift         import detect_drift
from src.data.memory        import save_snapshot, get_history, compare_last_two, get_all_files, clear_history, clear_all
from src.data.ai_suggestions import get_ai_suggestions

# ── Image Version 6 Imports ──────────────────────────────────────────────────
try:
    from src.vision_nlp.image_loader  import ImageLoader
    from src.vision_nlp.image_quality import analyze_dataset_quality
    from src.vision_nlp.image_analyzer import ImageAnalyzer
    from src.vision_nlp.image_report   import generate_report
    from src.vision_nlp.audit_orchestrator import AuditOrchestrator
    IMAGE_V6_AVAILABLE = True
except ImportError:
    IMAGE_V6_AVAILABLE = False

# ── Terminal colours ──────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
WHITE  = "\033[97m"

def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"

SEPARATOR = c("─" * 60, DIM)


# ── Shared helpers ────────────────────────────────────────────────────────────

def banner() -> None:
    print()
    print(c("╔══════════════════════════════════════════════════════════╗", CYAN))
    print(c("║", CYAN) + c("        🩺  Nydra — Data Inspection Agent         ", BOLD + WHITE) + c("║", CYAN))
    print(c("╚══════════════════════════════════════════════════════════╝", CYAN))
    print()


def usage() -> None:
    banner()
    print(c("  COMMANDS", BOLD + CYAN))
    print()
    print(c("  ── Core Inspection ──────────────────────────────────", CYAN))
    core_cmds = [
        ("inspect   <file>",   "Full inspection: issues + cleaning + stats + report"),
        ("clean     <file>",   "Clean data (remove dupes + fill missing)"),
        ("ai-impute <file>",   "🚀 Multimodal AI Impute (Transformers/Diffusion/Vision)"),
        ("stats     <file>",   "Show column statistics only"),
        ("missing   <file>",   "Show missing value counts per column"),
        ("duplicates <file>",  "Detect and show duplicate rows"),
        ("outliers  <file>",   "Detect outliers using IQR method"),
        ("report    <file>",   "Generate professional HTML report"),
        ("export    <file>",   "Inspect + export cleaned CSV to disk"),
    ]
    for cmd, desc in core_cmds:
        print(f"  {c('python cli.py', DIM)} {c(cmd, YELLOW)}  {c(desc, DIM)}")


    print()
    print(c("  ── Machine Learning & Prep ──────────────────────────", CYAN))
    ml_cmds = [
        ("ml        <file>",   "ML Readiness Score & detailed checks"),
        ("prepare   <file>",   "Prepare data for ML (encode + scale)"),
        ("target    <file>",   "Auto-detect best target column for ML"),
        ("split     <file>",   "Train/Test split advisor & code generator"),
        ("imbalance <file>",   "Class imbalance detector & strategy advisor"),
        ("encoding  <file>",   "Smart Encoding Advisor for categorical data"),
        ("importance <file>",  "Feature Importance using SHAP values"),
        ("automl    <file>",   "Run 5 models to find the best baseline"),
        ("pipeline  <file>",   "Export full Sklearn/Imblearn pipeline code"),
    ]
    for cmd, desc in ml_cmds:
        print(f"  {c('python cli.py', DIM)} {c(cmd, YELLOW)}  {c(desc, DIM)}")

    print()
    print(c("  ── Advanced Analysis ────────────────────────────────", CYAN))
    adv_cmds = [
        ("relations <file>",   "Detect column relationships & correlations"),
        ("network   <file>",   "Build & visualize correlation network"),
        ("engineer  <file>",   "Auto Feature Engineering (Date/Text/Num)"),
        ("schema    <file>",   "Schema Validator (Infer/Validate/Export)"),
        ("dna       <file>",   "Cognitive Data DNA (Statistical identity)"),
        ("image     <dir>",    "🖼️ Image Audit: load + quality + analyze + report"),
        ("text      <val>",    "📖 Text Audit: sentiment + quality + keywords"),
        ("audit-train <tr,ts>", "⚖️ ML Training Audit: Bias + Leakage + Labels"),
        ("drift     <base> <new>", "Detect data drift between two files"),
        ("suggest   <file>",   "AI-powered smart suggestions (needs .env)"),
        ("memory    [list|compare|clear]", "Track & compare inspection history"),
        ("interactive",        "Guided interactive mode — no flags needed"),
    ]
    for cmd, desc in adv_cmds:
        print(f"  {c('python cli.py', DIM)} {c(cmd, YELLOW)}  {c(desc, DIM)}")

    print()
    print(c("  OPTIONS", BOLD + CYAN))
    opts = [
        ("--strategy mean",   "Fill missing values with column mean  (default)"),
        ("--strategy median", "Fill missing values with column median"),
        ("--strategy mode",   "Fill missing values with most frequent value"),
        ("--strategy drop",   "Drop rows that contain any missing value"),
        ("--no-dedup",        "Do NOT remove duplicate rows"),
    ]
    for opt, desc in opts:
        print(f"  {c(opt, YELLOW)}  {c(desc, DIM)}")
    print()


def require_file(args: list[str]) -> str:
    if not args:
        print(c("  ✗ Error: no file path provided.", RED))
        sys.exit(1)
    path = args[0]
    if not os.path.exists(path):
        print(c(f"  ✗ File not found: {path}", RED))
        sys.exit(1)
    return path


def parse_options(args: list[str]) -> dict:
    opts = {"strategy": "mean", "dedup": True}
    i = 0
    while i < len(args):
        if args[i] == "--strategy" and i + 1 < len(args):
            opts["strategy"] = args[i + 1]
            i += 2
        elif args[i] == "--no-dedup":
            opts["dedup"] = False
            i += 1
        else:
            i += 1
    return opts


def safe_load(path: str) -> dict:
    try:
        return load_csv(path)
        # Save to DNA memory and get full analysis
        manager = get_dna_manager()
        analysis_result = manager.full_analysis(dna, path, ml_score=0)

        print(c("  ── DNA Memory ───────────────────────────────────────", CYAN))
        print(f"  Total datasets in memory : {c(str(analysis_result['total_in_db']), WHITE)}")

        similar = analysis_result["similar"]
        if similar:
            print(f"  Similar datasets found   : {c(str(len(similar)), YELLOW)}")
            for s in similar[:3]:
                print(f"     {c(s['source'], WHITE)} — {c(str(int(s['similarity'] * 100)) + '%', GREEN)} similar")
        else:
            print(f"  Similar datasets found   : {c('0 (first time)', DIM)}")

        rec = analysis_result["recommendation"]
        print()
        print(c("  ── Auto Strategy Recommendation ─────────────────────", CYAN))
        print(f"  Source     : {c(rec['source'], YELLOW)}")
        print(f"  Confidence : {c(str(int(rec['confidence'] * 100)) + '%', GREEN)}")
        print(f"  Strategy   : {c(str(rec['strategy']), WHITE)}")
        print(f"  Explanation: {c(rec['explanation'], DIM)}")

        evo = analysis_result["evolution"]
        if not evo.get("is_new"):
            print()
            print(c("  ── Evolution ────────────────────────────────────────", CYAN))
            print(f"  Version : {c(str(evo.get('version', 1)), WHITE)}")
            print(f"  Trend   : {c(evo.get('trend', 'unknown'), YELLOW)}")
            print(f"  ML Trend: {c(evo.get('ml_trend', 'unknown'), GREEN)}")
            if evo.get("changes"):
                for ch in evo["changes"][:3]:
                    print(f"  {c('→', CYAN)} {c(ch, WHITE)}")
    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))
    print()


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_inspect(args: list[str]) -> None:
    path = require_file(args)
    opts = parse_options(args[1:])
    banner()
    print(c(f"  Inspecting: {path}", CYAN))
    print(SEPARATOR)
    doctor = Nydra(remove_dupes=opts["dedup"], missing_strategy=opts["strategy"])
    result = doctor.inspect(path)
    print(result["summary"])

    outliers = detect_outliers(result["clean_data"])
    ml       = ml_readiness(result["clean_data"], outliers)
    save_snapshot(path, result["raw_analysis"], ml, outliers)
    print(c("  ✓ Snapshot saved to memory.", DIM))


def cmd_clean(args: list[str]) -> None:
    path = require_file(args)
    opts = parse_options(args[1:])
    banner()
    print(c(f"  Cleaning: {path}", CYAN))
    print(c(f"  Strategy: {opts['strategy']}  |  Remove dupes: {opts['dedup']}", DIM))
    print(SEPARATOR)

    data          = safe_load(path)
    original_rows = len(data["df"])
    dupes_removed = 0

    if opts["dedup"]:
        data, dupes_removed = remove_duplicates(data)

    total_missing = sum(v for v in missing_values(data).values())
    changes = {}
    if total_missing > 0:
        data, changes = handle_missing(data, strategy=opts["strategy"])

    print()
    print(c("  ── Results ──────────────────────────────────────────", CYAN))
    print(f"  Original rows   : {c(str(original_rows), WHITE)}")
    print(f"  Rows after clean: {c(str(len(data['df'])), GREEN)}")

    if dupes_removed:
        print(f"  Duplicates removed : {c(str(dupes_removed), YELLOW)}")
    else:
        print(f"  Duplicates removed : {c('0 (none found)', DIM)}")

    if changes:
        print()
        print(c("  ── Missing Values Fixed ─────────────────────────────", CYAN))
        if "rows_dropped" in changes:
            print(f"  Rows dropped: {c(str(changes['rows_dropped']), YELLOW)}")
        else:
            for col, info in changes.items():
                print(f"  {c(col, WHITE)}: filled {c(str(info['filled']), YELLOW)} "
                      f"value(s) → {c(str(info['replacement']), GREEN)}")
    else:
        print(f"\n  {c('✓ No missing values found.', GREEN)}")
    print()


def cmd_stats(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Column Statistics: {path}", CYAN))
    print(SEPARATOR)
    data  = safe_load(path)
    stats = basic_stats(data)
    shp   = shape(data)
    print(f"\n  {c(str(shp['rows']), WHITE)} rows  ×  {c(str(shp['columns']), WHITE)} columns\n")
    for col, s in stats.items():
        print(c(f"  ┌─ {col}", CYAN))
        print(f"  │  type   : {c(s['type'], YELLOW)}")
        print(f"  │  unique : {c(str(s['unique']), WHITE)}")
        print(f"  │  count  : {c(str(s['count']), WHITE)}")
        if s["type"] == "numeric":
            print(f"  │  min    : {c(str(s['min']), GREEN)}")
            print(f"  │  max    : {c(str(s['max']), GREEN)}")
            print(f"  │  mean   : {c(str(s['mean']), GREEN)}")
        else:
            print(f"  │  top    : {c(str(s['most_common']), GREEN)}")
        print(c("  └" + "─" * 40, DIM))
    print()


def cmd_missing(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Missing Values: {path}", CYAN))
    print(SEPARATOR)
    data  = safe_load(path)
    mv    = missing_values(data)
    total = sum(mv.values())
    rows  = len(data["df"])
    print()
    if total == 0:
        print(c("  ✓ No missing values found — data is complete!", GREEN))
    else:
        print(f"  {c('Column', BOLD):<30} {c('Missing', BOLD):<12} {c('%', BOLD)}")
        print(c("  " + "─" * 50, DIM))
        for col, cnt in mv.items():
            pct  = cnt / rows * 100
            flag = c(f"{cnt}", YELLOW if cnt > 0 else GREEN)
            bar  = c("█" * int(pct / 5), YELLOW if cnt > 0 else GREEN)
            print(f"  {col:<28} {flag:<12} {pct:5.1f}%  {bar}")
    print(f"\n  Total missing cells: {c(str(total), YELLOW if total else GREEN)}")
    print()


def cmd_duplicates(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Duplicate Detection: {path}", CYAN))
    print(SEPARATOR)
    data  = safe_load(path)
    df    = data["df"]
    count = duplicate_rows(data)
    print()
    if count == 0:
        print(c("  ✓ No duplicate rows found.", GREEN))
    else:
        print(c(f"  ⚠  Found {count} duplicate row(s):", YELLOW))
        print()
        dupes = df[df.duplicated(keep=False)]
        for idx, row in dupes.iterrows():
            vals = "  |  ".join(f"{col}={row[col]}" for col in df.columns[:4])
            print(c(f"  Row {idx + 1}: {vals} ...", YELLOW))
    print()


def cmd_outliers(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Advanced Outlier Detection: {path}", CYAN))
    print(SEPARATOR)

    data     = safe_load(path)
    analyzer = OutlierAnalyzer(verbose=False)
    report   = analyzer.analyze(data["df"])
    analyzer.print_report(report)

def cmd_report(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Smart Report Generator: {path}", CYAN))
    print(SEPARATOR)

    from src.data.quality_score import DataQualityScorer
    from src.smart_report import SmartReport
    from pathlib import Path

    df       = safe_load(path)["df"] 
    profile  = DataQualityScorer().score(df)
    reporter = SmartReport(output_dir="reports")
    paths    = reporter.generate_all(profile, df, Path(path).stem)

    print()
    for fmt, p in paths.items():
        print(c(f"  ✓ {fmt.upper()}: {p}", GREEN))
    print()

def cmd_export(args: list[str]) -> None:
    path = require_file(args)
    opts = parse_options(args[1:])
    banner()
    print(c(f"  Export: {path}", CYAN))
    print(SEPARATOR)
    doctor = Nydra(remove_dupes=opts["dedup"], missing_strategy=opts["strategy"])
    result = doctor.inspect(path)
    print(result["summary"])
    base, ext = os.path.splitext(path)
    out_path  = f"{base}_cleaned{ext}"
    result["clean_data"]["df"].to_csv(out_path, index=False, encoding="utf-8")
    print(c(f"  ✓ Cleaned file saved to: {out_path}", GREEN))
    print()


def cmd_report(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Smart Report Generator: {path}", CYAN))
    print(SEPARATOR)

    from src.data.quality_score import DataQualityScorer
    from src.smart_report import SmartReport
    from pathlib import Path

    df       = safe_load(path)["df"]
    profile  = DataQualityScorer().score(df)
    reporter = SmartReport(output_dir="reports")
    paths    = reporter.generate_all(profile, df, Path(path).stem)

    print()
    for fmt, p in paths.items():
        print(c(f"  ✓ {fmt.upper()}: {p}", GREEN))
    print()


def cmd_ml(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  ML Readiness Check: {path}", CYAN))
    print(SEPARATOR)

    data     = safe_load(path)
    outliers = detect_outliers(data)
    result   = ml_readiness(data, outliers)

    score       = result["score"]
    grade       = result["grade"]
    score_color = GREEN if score >= 75 else YELLOW if score >= 50 else RED

    print()
    print(f"  {c('ML Readiness Score', BOLD)}  {c(str(score), score_color + BOLD)}{c('/100', DIM)}  {c(f'[ {grade} ]', score_color)}")
    print()
    print(f"  {c(result['summary'], DIM)}")
    print()

    for check in result["checks"]:
        if check["status"] == "pass":
            icon  = c("✓", GREEN)
            color = DIM
        elif check["status"] == "warn":
            icon  = c("⚠", YELLOW)
            color = YELLOW
        else:
            icon  = c("✗", RED)
            color = RED

        bar_w   = int(check["points"] / check["max"] * 20)
        bar     = c("█" * bar_w + "░" * (20 - bar_w), color)
        pts_str = c(f"{check['points']}/{check['max']}", color)

        print(f"  {icon}  {c(check['name'], BOLD):<30} {pts_str}  {bar}")
        print(f"     {c(check['detail'], DIM)}")
        print()


def cmd_prepare(args: list[str]) -> None:
    path = require_file(args)
    opts = parse_options(args[1:])
    banner()
    print(c(f"  Preparing for ML: {path}", CYAN))
    print(SEPARATOR)

    data     = safe_load(path)
    prepared, log = prepare_for_ml(data, missing_strategy=opts["strategy"], scale=True, encode=True)

    print()
    print(c("  ── Pipeline Results ─────────────────────────────────", CYAN))
    print(f"  Duplicates removed : {c(str(log['duplicates_removed']), YELLOW)}")

    if log.get("missing_filled"):
        for col, info in log["missing_filled"].items():
            print(f"  {c(col, WHITE)}: filled {c(str(info['filled']), YELLOW)} value(s) → {c(str(info['value']), GREEN)}")

    if log.get("columns_dropped"):
        for col, reason in log["columns_dropped"]:
            print(f"  Dropped {c(col, YELLOW)} ({reason})")

    if log.get("encoded"):
        print(f"  Encoded {c(str(len(log['encoded'])), YELLOW)} text column(s)")

    if log.get("scaled"):
        print(f"  Scaled  {c(str(len(log['scaled'])), YELLOW)} numeric column(s)")

    shp = log["final_shape"]
    print()
    print(f"  {c('Final shape:', BOLD)} {c(str(shp['rows']), GREEN)} rows × {c(str(shp['columns']), GREEN)} columns")

    base, ext = os.path.splitext(path)
    out_path  = f"{base}_ready.csv"
    prepared["df"].to_csv(out_path, index=False, encoding="utf-8")

    print()
    print(c(f"  ✓ ML-ready file saved to: {out_path}", GREEN))
    print()


def cmd_relationships(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Column Relationships: {path}", CYAN))
    print(SEPARATOR)

    data    = safe_load(path)
    results = detect_relationships(data, threshold=0.4)

    print()
    if not results:
        print(c("  ✓ No strong relationships found between columns.", GREEN))
    else:
        print(c(f"  Found {len(results)} relationship(s):\n", CYAN))
        for r in results:
            strength = r["strength"]
            if strength >= 0.8:
                level = c("● strong", GREEN)
            elif strength >= 0.6:
                level = c("● moderate", YELLOW)
            else:
                level = c("● weak", DIM)

            bar_w = int(strength * 20)
            bar   = c("█" * bar_w + "░" * (20 - bar_w), GREEN if strength >= 0.8 else YELLOW)

            print(f"  {c(r['col_a'], WHITE)} ↔ {c(r['col_b'], WHITE)}")
            print(f"     {level}  {bar}  {c(str(strength), WHITE)}")
            print(f"     {c(r['method'], DIM)} · {c(r['type'], DIM)} · {c(r['direction'], DIM)}")
            print()
    print()

def cmd_encoding(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Smart Encoding Advisor: {path}", CYAN))
    print(SEPARATOR)

    data = safe_load(path)

    # detect target column
    df = data["df"]
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_cols     = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    print()
    print(c("  Model type:", BOLD))
    print(f"  1. Tree / Boosting   (XGBoost, RandomForest, LightGBM)")
    print(f"  2. Linear            (LogisticRegression, LinearSVC)")
    print(f"  3. Neural Network    (MLP, Keras)")
    model_choice = input(c("\n  Your choice (1-3, default=1): ", YELLOW)).strip() or "1"
    model_map = {"1": "tree", "2": "linear", "3": "neural"}
    model_type = model_map.get(model_choice, "tree")

    target_col = None
    if numeric_cols:
        print()
        print(c("  Numeric columns (potential targets):", BOLD))
        for i, col in enumerate(numeric_cols, 1):
            print(f"  {i}. {col}")
        print(f"  0. No target column")
        t_choice = input(c("\n  Select target column (0 to skip): ", YELLOW)).strip()
        if t_choice.isdigit() and 1 <= int(t_choice) <= len(numeric_cols):
            target_col = numeric_cols[int(t_choice) - 1]

    print()
    print(c("  Analyzing...", DIM))

    result = encoding_advisor(data, target_col=target_col, model_type=model_type)

    print()
    print(c(f"  {result['summary']}", CYAN))
    print()

    for r in result["columns"]:
        risk_color = GREEN if r["risk"] == "low" else YELLOW if r["risk"] == "medium" else RED

        print(c(f"  ┌─ {r['column']}", CYAN))
        print(f"  │  strategy   : {c(r['strategy'], risk_color + BOLD)}")
        print(f"  │  cardinality: {c(str(r['n_unique']), WHITE)} unique ({r['cardinality']})")
        print(f"  │  entropy    : {c(str(r['entropy']), DIM)}")
        print(f"  │  imbalance  : {c(str(r['imbalance']) + 'x', DIM)}")
        if r["target_corr"] is not None:
            print(f"  │  target corr: {c(str(r['target_corr']), YELLOW)}")
        if r["is_ordinal"]:
            print(f"  │  order      : {c(' < '.join(r['ordinal_scale']), GREEN)}")
        print(f"  │  reason     : {c(r['reason'], DIM)}")
        print(f"  │  sklearn    : {c(r['sklearn_tip'], DIM)}")
        if r["warnings"]:
            for w in r["warnings"]:
                print(f"  │  {c('⚠ ' + w, YELLOW)}")
        print(c("  └" + "─" * 50, DIM))
        print()

    print(c("  ── Pipeline Code ────────────────────────────────────", CYAN))
    print()
    for line in result["pipeline_code"].splitlines():
        print(f"  {c(line, WHITE)}")
    print()

def cmd_schema(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Schema Validator: {path}", CYAN))
    print(SEPARATOR)

    data = safe_load(path)

    print()
    print(c("  What do you want to do?", BOLD))
    print(f"  1. Auto-infer schema from data and validate")
    print(f"  2. Load schema from JSON file")
    print(f"  3. Export inferred schema to JSON")
    choice = input(c("\n  Your choice (1-3): ", YELLOW)).strip()

    if choice == "3":
        schema   = infer_schema(data)
        base, _  = os.path.splitext(path)
        out_path = f"{base}_schema.json"
        with open(out_path, "w", encoding="utf-8") as f:
            import json
            json.dump(schema_to_dict(schema), f, indent=2)
        print(c(f"\n  ✓ Schema exported to: {out_path}", GREEN))
        print(c("  Edit it and re-run with option 2 to validate.", DIM))
        print()
        return

    if choice == "2":
        schema_path = input(c("  Schema JSON file path: ", YELLOW)).strip().strip('"')
        try:
            import json
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = schema_from_dict(json.load(f))
        except Exception as e:
            print(c(f"\n  ✗ Could not load schema: {e}", RED))
            return
    else:
        schema = infer_schema(data)

    result = validate_schema(data, schema)

    print()
    valid_color = GREEN if result["valid"] else RED
    print(c(f"  {result['summary']}", valid_color))
    print()
    print(f"  Coverage : {c(str(result['coverage']) + '%', CYAN)} of columns validated")
    if result["uncovered"]:
        print(f"  Uncovered: {c(', '.join(result['uncovered']), DIM)}")
    print()

    for r in result["results"]:
        if r["status"] == "pass":
            icon  = c("✓", GREEN)
            color = GREEN
        elif r["status"] == "missing":
            icon  = c("?", YELLOW)
            color = YELLOW
        else:
            icon  = c("✗", RED)
            color = RED

        print(f"  {icon}  {c(r['column'], color + BOLD)}")

        for p in r["passed"]:
            print(f"     {c('✓ ' + p, DIM)}")
        for w in r["warnings"]:
            print(f"     {c('⚠ ' + w, YELLOW)}")
        for e in r["errors"]:
            print(f"     {c('✗ ' + e, RED)}")
        print()
    print()


def cmd_suggest(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  AI Suggestions: {path}", CYAN))
    print(SEPARATOR)

    api_key  = os.environ.get("NYDRA_API_KEY", "")
    base_url = os.environ.get("NYDRA_BASE_URL", "https://api.groq.com/openai/v1")
    model    = os.environ.get("NYDRA_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        print(c("\n  ✗ No API key found.", RED))
        print(c("  Add these to your .env file:\n", DIM))
        print(c("  NYDRA_API_KEY=your_key", YELLOW))
        print(c("  NYDRA_BASE_URL=https://api.groq.com/openai/v1", YELLOW))
        print(c("  NYDRA_MODEL=llama-3.3-70b-versatile", YELLOW))
        print(c("\n  Supported providers: Groq (free), Gemini (free), OpenRouter, OpenAI", DIM))
        return

    data     = safe_load(path)
    if "rows" not in data:
        data["rows"] = data["df"].to_dict("records")
    analysis = full_report(data)
    outliers = detect_outliers(data)
    rels     = detect_relationships(data, threshold=0.4)
    ml       = ml_readiness(data, outliers)

    print()
    print(c(f"  Provider : {base_url}", DIM))
    print(c(f"  Model    : {model}", DIM))
    print(c("  Thinking...\n", DIM))

    try:
        import requests
        suggestions = get_ai_suggestions(
            api_key       = api_key,
            base_url      = base_url,
            model         = model,
            source        = path,
            shape         = analysis["shape"],
            missing       = analysis["missing_values"],
            dupes         = analysis["duplicate_rows"],
            stats         = analysis["column_stats"],
            outliers      = outliers,
            relationships = rels,
            ml_score      = ml,
        )
        print(c("  ── AI Suggestions ───────────────────────────────────", CYAN))
        print()
        for line in suggestions.splitlines():
            if line.strip():
                print(f"  {c(line, WHITE)}")
    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))
    print()


def cmd_memory(args: list[str]) -> None:
    banner()
    print(c("  Data Memory", CYAN))
    print(SEPARATOR)
    print()

    sub = args[0].lower() if args else "list"

    if sub == "list":
        files = get_all_files()
        if not files:
            print(c("  No files in memory yet.", DIM))
            print(c("  Run 'inspect' on any file to start tracking.", DIM))
        else:
            print(c(f"  {len(files)} file(s) tracked:\n", CYAN))
            for f in files:
                score_color = GREEN if f["best_score"] >= 75 else YELLOW if f["best_score"] >= 50 else RED
                print(f"  {c(f['file_name'], WHITE)}")
                print(f"     inspections : {c(str(f['inspections']), CYAN)}")
                print(f"     last seen   : {c(f['last_seen'][:19].replace('T', ' '), DIM)}")
                print(f"     best score  : {c(str(f['best_score']), score_color)}/100")
                print()

    elif sub == "compare":
        if len(args) < 2:
            print(c("  Usage: python cli.py memory compare <file>", RED))
            return
        path    = args[1]
        changes = compare_last_two(path)
        if not changes:
            print(c("  Need at least 2 inspections to compare.", YELLOW))
            print(c("  Run 'inspect' on this file again later.", DIM))
            return

        print(c(f"  Comparing last 2 inspections of: {changes['file']}\n", CYAN))
        print(f"  {c('Previous', DIM)} : {changes['old_time']}")
        print(f"  {c('Current',  DIM)} : {changes['new_time']}\n")

        def _show(label, info):
            delta = info["delta"]
            if delta == 0:
                icon, col = "→", DIM
            elif info["improved"]:
                icon, col = "↑", GREEN
            else:
                icon, col = "↓", RED
            sign = "+" if delta > 0 else ""
            print(f"  {label:<20} {info['old']} → {c(str(info['new']), col)}  {c(f'{sign}{delta}', col)}  {icon}")

        _show("Rows",         changes["rows"])
        _show("Columns",      changes["columns"])
        _show("Missing",      changes["missing"])
        _show("Duplicates",   changes["duplicates"])
        _show("ML Score",     changes["ml_score"])
        _show("Outlier cols", changes["outliers"])

    elif sub == "clear":
        if len(args) < 2:
            n = clear_all()
            print(c(f"  ✓ Cleared all memory ({n} snapshots).", GREEN))
        else:
            n = clear_history(args[1])
            print(c(f"  ✓ Cleared {n} snapshot(s) for {args[1]}.", GREEN))

    else:
        print(c(f"  Unknown sub-command: {sub}", RED))
        print(c("  Usage: memory list | memory compare <file> | memory clear", DIM))

    print()


def cmd_drift(args: list[str]) -> None:
    if len(args) < 2:
        print(c("\n  Usage: python cli.py drift <baseline_file> <current_file>\n", RED))
        return

    base_path = args[0]
    curr_path = args[1]

    banner()
    print(c(f"  Drift Detection", CYAN))
    print(c(f"  Baseline : {base_path}", DIM))
    print(c(f"  Current  : {curr_path}", DIM))
    print(SEPARATOR)

    base   = safe_load(base_path)
    curr   = safe_load(curr_path)
    result = detect_drift(base, curr)

    sev_color = GREEN if result["severity"] == "none" else YELLOW if result["severity"] == "low" else RED

    print()
    print(f"  {c('Severity', BOLD)} : {c(result['severity'].upper(), sev_color)}")
    print(f"  {c('Summary',  BOLD)} : {c(result['summary'], DIM)}")
    print()
    print(f"  Baseline : {result['base_shape']['rows']} rows × {result['base_shape']['cols']} cols")
    print(f"  Current  : {result['new_shape']['rows']} rows × {result['new_shape']['cols']} cols")
    print()

    if result["drifted_columns"]:
        print(c(f"  ── Drifted Columns ({len(result['drifted_columns'])}) ─────────────────────────", RED))
        print()
        for d in result["drifted_columns"]:
            sev = d["severity"]
            sc  = RED if sev == "high" else YELLOW if sev == "medium" else DIM
            print(c(f"  ┌─ {d['column']}  [{sev}]", sc))
            for issue in d["issues"]:
                print(f"  │  {c(issue, WHITE)}")
            print(c("  └" + "─" * 40, DIM))
            print()

    if result["stable_columns"]:
        print(c(f"  ── Stable Columns ({len(result['stable_columns'])}) ──────────────────────────", GREEN))
        print(f"  {c(', '.join(result['stable_columns']), DIM)}")

    print()

def cmd_engineer(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Auto Feature Engineering: {path}", CYAN))
    print(SEPARATOR)

    data     = safe_load(path)
    enriched, log = engineer_features(data)

    total = sum(len(v) for v in log.values())
    print()
    print(c(f"  Generated {total} new feature(s):\n", CYAN))

    if log["date"]:
        print(c("  ── Date Features ────────────────────────────────────", CYAN))
        for f in log["date"]:
            print(f"  {c('+', GREEN)} {f}")
        print()

    if log["numeric"]:
        print(c("  ── Numeric Features ─────────────────────────────────", CYAN))
        for f in log["numeric"]:
            print(f"  {c('+', GREEN)} {f}")
        print()

    if log["interaction"]:
        print(c("  ── Interaction Features ─────────────────────────────", CYAN))
        for f in log["interaction"]:
            print(f"  {c('+', GREEN)} {f}")
        print()

    if log["text"]:
        print(c("  ── Text Features ────────────────────────────────────", CYAN))
        for f in log["text"]:
            print(f"  {c('+', GREEN)} {f}")
        print()

    base, ext = os.path.splitext(path)
    out_path  = f"{base}_engineered.csv"
    enriched["df"].to_csv(out_path, index=False, encoding="utf-8")

    print(c(f"  ✓ Saved to: {out_path}", GREEN))
    print(f"  Original : {len(data['df'].columns)} columns")
    print(f"  New      : {c(str(len(enriched['df'].columns)), GREEN)} columns")
    print()


def cmd_text(args: list[str]) -> None:
    if not args:
        print(c("\n  Usage: python cli.py text \"Your text here...\"", RED))
        print(c("         python cli.py text <filepath> [--col <column_name>]\n", DIM))
        return

    input_val = args[0]
    col_name = None
    if "--col" in args:
        idx = args.index("--col")
        if idx + 1 < len(args):
            col_name = args[idx + 1]

    banner()
    print(c("  📖 Advanced Text Audit & Intelligence", CYAN))
    print(SEPARATOR)

    doctor = Nydra()

    try:
        # Check if input is a file
        if os.path.isfile(input_val):
            print(c(f"  📂 Loading file: {input_val}", DIM))
            if input_val.endswith(".csv"):
                df = pd.read_csv(input_val)
                print(c(f"  📊 Analyzing column: {col_name or 'first detected text column'}", DIM))
                res = doctor.audit_text(df, text_column=col_name)
            else:
                # Assume it's a raw text file
                with open(input_val, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                res = doctor.audit_text(text)
        else:
            # Assume it's raw text
            res = doctor.audit_text(input_val)

        if res["status"] != "success":
            print(c(f"  ✗ Analysis failed: {res.get('message')}", RED))
            return

        # Display results
        if "intelligence" in res:
            intel = res["intelligence"]
            print(c("  ── Intelligence ─────────────────────────────────────", CYAN))
            if "sentiment" in intel:
                sent = intel["sentiment"]["doc_level"]
                score = sent["score"]
                sc = GREEN if score > 0.2 else RED if score < -0.2 else YELLOW
                print(f"  {c('Sentiment', BOLD):<15} : {c(sent['label'].upper(), sc)} ({score:.2f})")

            if "style" in intel:
                style = intel["style"]
                print(f"  {c('Complexity', BOLD):<15} : {c(style['readability_grade'], WHITE)}")
                print(f"  {c('Tone', BOLD):<15} : {c(style['dominant_tone'].upper(), YELLOW)}")

            if "keywords" in intel and intel["keywords"]:
                kw = [k[0] for k in intel["keywords"][:5]]
                print(f"  {c('Keywords', BOLD):<15} : {c(', '.join(kw), CYAN)}")
            print()

        if "quality" in res:
            qual = res["quality"]
            print(c("  ── Quality Audit ────────────────────────────────────", CYAN))
            score = qual["overall_score"]
            sc = GREEN if score > 80 else YELLOW if score > 60 else RED
            print(f"  {c('Quality Score', BOLD):<15} : {c(f'{score:.1f}/100', sc)} [{c(qual['verdict'], sc)}]")

            if qual.get("priority_issues"):
                print(c("\n  Top Issues:", RED))
                for issue in qual["priority_issues"][:3]:
                    print(f"  {c('!', RED)} {issue['message']} ({issue['severity']})")
            print()

    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))
        import traceback
        traceback.print_exc()


def cmd_audit_train(args: list[str]) -> None:
    if len(args) < 2:
        print(c("\n  Usage: python cli.py audit-train <train_file> <test_file> [--target <col>] [--sensitive <col1,col2>]\n", RED))
        return

    train_path = args[0]
    test_path = args[1]
    target_col = "target"
    sensitive_cols = None

    if "--target" in args:
        idx = args.index("--target")
        if idx + 1 < len(args):
            target_col = args[idx + 1]

    if "--sensitive" in args:
        idx = args.index("--sensitive")
        if idx + 1 < len(args):
            sensitive_cols = args[idx + 1].split(",")

    banner()
    print(c("  ⚖️  ML Training Data Audit (Bias, Leakage, Label Quality)", CYAN))
    print(SEPARATOR)

    try:
        print(c(f"  📂 Loading Train: {train_path}", DIM))
        print(c(f"  📂 Loading Test : {test_path}", DIM))
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        doctor = Nydra()
        res = doctor.audit_training_data(train_df, test_df, target_col, sensitive_cols)

        if res["status"] != "success":
            print(c(f"  ✗ Audit failed: {res.get('message')}", RED))
            return

        # Summary
        summary = res["summary"]
        print(c("  ── Summary ──────────────────────────────────────────", CYAN))
        lq_score = summary.get("label_quality_score", 0)
        print(f"  {c('Label Quality', BOLD):<18} : {c(f'{lq_score:.1f}/100', GREEN)}")
        lk_score = summary.get("leakage_score", 0)
        lk_risk = summary.get('leakage_risk', 'N/A')
        print(f"  {c('Leakage Score', BOLD):<18} : {c(f'{lk_score:.1f}/100', YELLOW)} [{lk_risk}]")
        if "bias_score" in summary:
            bs_score = summary.get("bias_score", 0)
            bs_verdict = summary.get('bias_verdict', 'N/A')
            print(f"  {c('Bias Score', BOLD):<18} : {c(f'{bs_score:.1f}/100', RED)} [{bs_verdict}]")
        print()

        # Detailed Label Quality
        if "label_quality" in res:
            lq = res["label_quality"]
            print(c("  ── Label Quality Issues ─────────────────────────────", CYAN))
            if lq.priority_recommendations:
                for rec in lq.priority_recommendations[:3]:
                    print(f"  {c('!', RED)} [{rec['area']}] {rec['issue']}")
            else:
                print(c("  ✓ No critical label quality issues detected.", GREEN))
            print()

        # Detailed Leakage
        if "leakage" in res and isinstance(res["leakage"], dict):
            lk = res["leakage"]
            print(c("  ── Data Leakage Findings ────────────────────────────", CYAN))
            findings = lk.get("all_findings", [])
            leakers = [f["feature"] for f in findings if f.get("feature")]
            if leakers:
                print(f"  {c('⚠', RED)} Found leaking features: {', '.join(list(set(leakers))[:5])}")
            
            # Check overlap from detector results
            overlap_res = lk.get("detector_results", {}).get("TrainTestOverlapDetector", {})
            overlap_rate = overlap_res.get("overlap_rate", 0)
            if overlap_rate > 0:
                print(f"  {c('⚠', RED)} Train/Test overlap: {overlap_rate:.2%}")
            
            if not leakers and overlap_rate == 0:
                print(c("  ✓ No significant leakage detected.", GREEN))
            print()

        # Detailed Bias
        if "bias" in res and isinstance(res["bias"], dict):
            bs = res["bias"]
            print(c("  ── Bias & Fairness Findings ─────────────────────────", CYAN))
            findings = bs.get("findings", [])
            if findings:
                biased = [f["attribute"] for f in findings if f.get("attribute")]
                print(f"  {c('⚠', RED)} Potential bias in: {', '.join(list(set(biased)))}")
            else:
                print(c("  ✓ No significant bias detected.", GREEN))
            print()

    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))
        import traceback
        traceback.print_exc()


def cmd_target(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Target Column Detection: {path}", CYAN))
    print(SEPARATOR)

    data   = safe_load(path)
    result = detect_target(data)

    conf_color = GREEN if result["confidence"] == "high" else YELLOW if result["confidence"] == "medium" else RED
    task_color = CYAN if result["task_type"] == "classification" else YELLOW if result["task_type"] == "regression" else DIM

    print()
    print(f"  {c('Recommended target', BOLD)} : {c(result['recommended'], GREEN)}")
    print(f"  {c('Task type', BOLD)}          : {c(result['task_type'].upper(), task_color)}")
    print(f"  {c('Confidence', BOLD)}         : {c(result['confidence'].upper(), conf_color)}")
    print(f"  {c('Reason', BOLD)}             : {c(result['reason'], DIM)}")
    print()

    print(c("  ── All Candidates ───────────────────────────────────", CYAN))
    print()
    for i, cand in enumerate(result["candidates"]):
        rank  = c(f"#{i+1}", YELLOW if i == 0 else DIM)
        score = c(str(cand["score"]), GREEN if i == 0 else DIM)
        print(f"  {rank}  {c(cand['column'], WHITE if i == 0 else DIM):<25} score={score}")
        if cand["signals"]:
            print(f"      {c(' | '.join(cand['signals'][:2]), DIM)}")
        print()

def cmd_network(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Correlation Network: {path}", CYAN))
    print(SEPARATOR)

    data   = safe_load(path)
    result = build_correlation_network(data, threshold=0.3)

    print()
    print(f"  {c(result['summary'], DIM)}")
    print()
    print(f"  Nodes : {c(str(len(result['nodes'])), CYAN)} columns")
    print(f"  Edges : {c(str(result['stats']['total_edges']), CYAN)} relationships")
    print(f"  Strong: {c(str(result['stats']['strong']), GREEN)}  "
          f"Moderate: {c(str(result['stats']['moderate']), YELLOW)}  "
          f"Weak: {c(str(result['stats']['weak']), DIM)}")
    print()

    if result["edges"]:
        print(c("  ── Relationships (sorted by strength) ───────────────", CYAN))
        print()
        for e in result["edges"]:
            strength = e["strength"]
            if strength >= 0.7:
                color = GREEN
                level = "strong"
            elif strength >= 0.5:
                color = YELLOW
                level = "moderate"
            else:
                color = DIM
                level = "weak"

            bar = c("█" * int(strength * 20) + "░" * (20 - int(strength * 20)), color)
            print(f"  {c(e['source'], WHITE)} ↔ {c(e['target'], WHITE)}")
            print(f"     {bar}  {c(str(strength), color)}  {c(level, color)}  {c(e['method'], DIM)}")
            print()
    else:
        print(c("  ✓ No significant relationships found.", GREEN))
    print()

def cmd_automl(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Advanced Auto ML (Optuna): {path}", CYAN))
    print(SEPARATOR)

    data = safe_load(path)

    print()
    target = input(c("  Target column name: ", YELLOW)).strip()
    limit  = input(c("  Time limit in seconds (default=300): ", YELLOW)).strip() or "300"
    stack  = input(c("  Use Model Stacking? (y/n, default=y): ", YELLOW)).strip().lower() != "n"

    print()
    print(c(f"  Optimizing 15+ models for {limit}s...", DIM))
    print()

    try:
        result = run_advanced_automl(data, target_col=target, time_limit=int(limit), use_stacking=stack)

        print(c(f"  ── Best Model ───────────────────────────────────────", CYAN))
        print(f"  🏆 {c(result['best_model'], GREEN + BOLD)}")
        print(f"  Score : {c('{:.4f}'.format(result['best_score']), GREEN)}")
        print()

        print(c("  ── Model Leaderboard ────────────────────────────────", CYAN))
        print()
        for i, r in enumerate(result["leaderboard"][:10]):
            score = r["score"]
            bar_w = int(score * 20)
            bar   = c("█" * bar_w + "░" * (20 - bar_w), GREEN if i==0 else DIM)
            print(f"  #{i+1:<2} {c(r['model'], WHITE if i==0 else DIM):<25} {bar} {c(f'{score:.4f}', WHITE)}")

        print()
        print(c("  ── Recommendation ───────────────────────────────────", CYAN))
        print(f"  {c(result['recommendation'], WHITE)}")

        if "pipeline_code" in result:
            base, _ = os.path.splitext(path)
            out_path = f"{base}_best_pipeline.py"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result["pipeline_code"])
            print()
            print(c(f"  ✓ Exported best pipeline to: {out_path}", GREEN))

    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))

    print()

def cmd_ai_impute(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  🚀 AI Multimodal Impute: {path}", CYAN))
    print(SEPARATOR)

    data = safe_load(path)
    print()
    img_col = input(c("  Image path column (optional): ", YELLOW)).strip() or None
    doc_col = input(c("  Doc path column (optional): ", YELLOW)).strip() or None

    print()
    print(c("  Running Transformers + Diffusion + Vision Pipeline...", DIM))
    print()

    try:
        res = run_ai_impute(data["df"], image_col=img_col, doc_col=doc_col)
        
        print(c("  ── Imputation Results ───────────────────────────────", CYAN))
        print(f"  Method    : {c(res.method, GREEN)}")
        print(f"  Cells fixed: {c(str(res.n_imputed), YELLOW)}")
        print(f"  Duration  : {c(f'{res.duration_ms/1000:.1f}s', DIM)}")
        print()

        if res.warnings:
            print(c("  ── Warnings ─────────────────────────────────────────", YELLOW))
            for w in res.warnings:
                print(f"  {c('⚠', YELLOW)} {w}")
            print()

        base, ext = os.path.splitext(path)
        out_path  = f"{base}_ai_fixed{ext}"
        res.df_imputed.to_csv(out_path, index=False)
        print(c(f"  ✓ AI-fixed file saved to: {out_path}", GREEN))

    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))

    print()

def cmd_importance(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Feature Importance (SHAP): {path}", CYAN))
    print(SEPARATOR)

    data   = safe_load(path)
    print()
    target = input(c("  Target column name: ", YELLOW)).strip()
    task   = input(c("  Task type (auto/classification/regression): ", YELLOW)).strip() or "auto"

    print()
    print(c("  Computing SHAP values...", DIM))
    print()

    try:
        result = compute_feature_importance(data, target_col=target, task_type=task)

        print(c(f"  ── Feature Importance [{result['method'].upper()}] ────────────────────", CYAN))
        print(f"  Target  : {c(result['target'], CYAN)}")
        print(f"  Method  : {c(result['method'].upper(), YELLOW)}")
        print(f"  Summary : {c(result['summary'], DIM)}")
        print()

        for i, f in enumerate(result["features"]):
            pct   = f["pct"]
            color = GREEN if pct >= 20 else YELLOW if pct >= 10 else DIM
            bar_w = int(pct / 5)
            bar   = c("█" * bar_w + "░" * (20 - bar_w), color)
            rank  = c(f"#{i+1}", YELLOW if i == 0 else DIM)

            print(f"  {rank:<5} {c(f['feature'], WHITE if i < 3 else DIM):<25} "
                  f"{bar}  {c(f'{pct:.1f}%', color)}")

        print()
        print(c(f"  🏆 Most important: {result['top_feature']}", GREEN))

    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))

    print()

def cmd_split(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Train/Test Split Advisor: {path}", CYAN))
    print(SEPARATOR)

    data   = safe_load(path)
    print()
    target = input(c("  Target column name: ", YELLOW)).strip()
    task   = input(c("  Task type (auto/classification/regression): ", YELLOW)).strip() or "auto"
    print()

    try:
        result = advise_split(data, target_col=target, task_type=task)

        print(c(f"  ── Recommendation ───────────────────────────────────", CYAN))
        print()
        print(f"  {c('Strategy', BOLD)}  : {c(result['strategy'], GREEN)}")
        print(f"  {c('Task', BOLD)}      : {c(result['task_type'], CYAN)}")
        print(f"  {c('Dataset', BOLD)}   : {c(str(result['n_rows']), WHITE)} rows")
        print(f"  {c('Train', BOLD)}     : {c(str(int(result['train_size']*100))+'%', GREEN)}")
        print(f"  {c('Test', BOLD)}      : {c(str(int(result['test_size']*100))+'%', YELLOW)}")
        print(f"  {c('CV Folds', BOLD)}  : {c(str(result['cv_folds']), WHITE)}")
        print(f"  {c('Stratify', BOLD)}  : {c(str(result['stratify']), GREEN if result['stratify'] else DIM)}")
        print()

        if result["warnings"]:
            print(c("  ── Warnings ─────────────────────────────────────────", YELLOW))
            for w in result["warnings"]:
                print(f"  {c('⚠', YELLOW)} {c(w, WHITE)}")
            print()

        if result["recommendations"]:
            print(c("  ── Recommendations ──────────────────────────────────", CYAN))
            for r in result["recommendations"]:
                print(f"  {c('→', CYAN)} {c(r, WHITE)}")
            print()

        print(c("  ── Ready-to-use Code ────────────────────────────────", CYAN))
        print()
        for line in result["code"].splitlines():
            print(f"  {c(line, DIM)}")

    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))

    print()

def cmd_imbalance(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Class Imbalance Detector: {path}", CYAN))
    print(SEPARATOR)

    data   = safe_load(path)
    print()
    target = input(c("  Target column name: ", YELLOW)).strip()
    print()

    try:
        result = detect_imbalance(data, target_col=target)

        sev_color = {
            "none":     GREEN,
            "mild":     YELLOW,
            "moderate": YELLOW,
            "severe":   RED,
            "extreme":  RED,
        }.get(result["severity"], DIM)

        print(c("  ── Class Distribution ───────────────────────────────", CYAN))
        print()
        for cls, info in result["class_dist"].items():
            bar_w = int(info["pct"] / 5)
            bar   = c("█" * bar_w + "░" * (20 - bar_w), GREEN if info["pct"] > 30 else YELLOW)
            print(f"  {c(str(cls), WHITE):<20} {bar}  {c(str(info['count']), WHITE)} ({info['pct']}%)")
        print()

        print(f"  {c('Severity', BOLD)}        : {c(result['severity'].upper(), sev_color)}")
        print(f"  {c('Imbalance ratio', BOLD)}  : {c(str(result['imbalance_ratio'])+':1', sev_color)}")
        print(f"  {c('Majority class', BOLD)}   : {c(result['majority_class'], WHITE)}")
        print(f"  {c('Minority class', BOLD)}   : {c(result['minority_class'], WHITE)}")
        print()

        if result["warnings"]:
            print(c("  ── Warnings ─────────────────────────────────────────", YELLOW))
            for w in result["warnings"]:
                print(f"  {c('⚠', YELLOW)} {c(w, WHITE)}")
            print()

        print(c("  ── Recommended Strategy ─────────────────────────────", CYAN))
        print(f"  {c('🏆 ' + result['recommended'], GREEN)}")
        print()

        print(c("  ── All Strategies ───────────────────────────────────", CYAN))
        for s in result["strategies"]:
            rec = c(" ← recommended", GREEN) if s["recommended"] else ""
            print(f"\n  {c(s['name'], WHITE + BOLD)}{rec}")
            print(f"  {c(s['description'], DIM)}")
            print(f"  Best for: {c(s['best_for'], DIM)}")

        print()
        print(c("  ── Ready-to-use Code ────────────────────────────────", CYAN))
        print(f"  {c('Showing code for: ' + result['recommended'], DIM)}\n")
        for line in result["code"].get(result["recommended"], "").splitlines():
            print(f"  {c(line, DIM)}")

    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))

    print()

def cmd_pipeline(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  Pipeline Export: {path}", CYAN))
    print(SEPARATOR)

    data = safe_load(path)
    print()
    target = input(c("  Target column name: ", YELLOW)).strip()
    task   = input(c("  Task type (auto/classification/regression): ", YELLOW)).strip() or "auto"

    print()
    print(c("  Available models:", BOLD))
    models = {
        "1": "random_forest",
        "2": "gradient_boosting",
        "3": "logistic",
        "4": "svm",
        "5": "knn",
    }
    for k, v in models.items():
        print(f"  {c(k, YELLOW)}. {v.replace('_', ' ').title()}")

    model_choice = input(c("\n  Choose model (1-5, default=1): ", YELLOW)).strip() or "1"
    model_name   = models.get(model_choice, "random_forest")

    imbalance = input(c("  Handle class imbalance with SMOTE? (y/n, default=n): ", YELLOW)).strip().lower() == "y"

    base, _ = os.path.splitext(path)
    out_path = f"{base}_pipeline.py"

    print()
    print(c("  Generating pipeline...", DIM))

    try:
        result = export_pipeline(
            data,
            target_col       = target,
            task_type        = task,
            model_name       = model_name,
            handle_imbalance = imbalance,
            output_path      = out_path,
        )

        print()
        print(c("  ── Pipeline Summary ─────────────────────────────────", CYAN))
        print(f"  Task       : {c(result['task_type'].upper(), CYAN)}")
        print(f"  Target     : {c(result['target'], GREEN)}")
        print(f"  Model      : {c(result['model'].replace('_',' ').title(), WHITE)}")
        print(f"  Features   : {c(str(result['n_features']), WHITE)} "
              f"({c(str(result['n_numeric']), CYAN)} numeric, "
              f"{c(str(result['n_categorical']), YELLOW)} categorical)")
        print(f"  SMOTE      : {c('Yes', GREEN) if imbalance else c('No', DIM)}")
        print()
        print(c(f"  ✓ Pipeline saved to: {out_path}", GREEN))
        print(c("  ✓ Open the file, change 'your_data.csv' and run it!", CYAN))

    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))

    print()

def cmd_connect(args: list[str]) -> None:
    banner()
    print(c("  🗄️  Database Connector", CYAN))
    print(SEPARATOR)
    print()

    db_type = input(c("  DB type (postgresql/mysql/sqlite): ", YELLOW)).strip().lower()

    if db_type == "sqlite":
        filepath = input(c("  File path: ", YELLOW)).strip()
        kwargs   = {"db_type": "sqlite", "filepath": filepath}
    else:
        host     = input(c("  Host (localhost): ", YELLOW)).strip() or "localhost"
        port     = input(c("  Port: ", YELLOW)).strip()
        database = input(c("  Database: ", YELLOW)).strip()
        username = input(c("  Username: ", YELLOW)).strip()
        password = input(c("  Password: ", YELLOW)).strip()
        kwargs   = {
            "db_type":  db_type,
            "host":     host,
            "port":     int(port) if port else (5432 if db_type == "postgresql" else 3306),
            "database": database,
            "username": username,
            "password": password,
        }

    print()
    print(c("  Connecting...", DIM))

    try:
        connector = build_connector(**kwargs)
        ok, msg   = connector.connect()

        if not ok:
            print(c(f"  ✗ {msg}", RED))
            return

        print(c(f"  ✓ {msg}", GREEN))
        session = DBSession(connector)

        # Show tables
        tables = session.connector.get_tables()
        print()
        print(c(f"  Found {len(tables)} table(s):\n", CYAN))
        for t in tables:
            print(f"  {c(t['table'], WHITE):<30} {c(str(t['rows']), YELLOW)} rows")

        print()
        print(c("  ── Commands ─────────────────────────────────────────", CYAN))
        print(c("  tables           ← show all tables", DIM))
        print(c("  analyse <table>  ← full Nydra analysis", DIM))
        print(c("  schema <table>   ← show table structure", DIM))
        print(c("  export <table>   ← export table to CSV", DIM))
        print(c("  clean <table>    ← analyse + clean data", DIM))
        print(c("  sql              ← run custom SQL query", DIM))
        print(c("  exit             ← disconnect", DIM))

        # Interactive loop
        while True:
            cmd = input(c("  db> ", YELLOW)).strip()

            if cmd == "exit":
                session.disconnect()
                print(c("  Disconnected.", DIM))
                break

            elif cmd.startswith("analyse "):
                table = cmd.split(" ", 1)[1].strip()
                print(c(f"  Loading {table}...", DIM))
                try:
                    data = session.editor.run_table(table, limit=10_000)
                    from src.data.analyzer     import full_report, detect_outliers
                    from src.data.ml_readiness import ml_readiness
                    analysis = full_report(data)
                    outliers = detect_outliers(data)
                    ml = ml_readiness(data, outliers if isinstance(outliers, dict) else {})
                    print(c(f"  ✓ {len(data['df'])} rows loaded", GREEN))
                    print(f"  ML Score : {c(str(ml['score']), GREEN)}/100")
                    print(f"  Missing  : {c(str(sum(analysis['missing_values'].values())), YELLOW)}")
                    print(f"  Dupes    : {c(str(analysis['duplicate_rows']), YELLOW)}")
                except Exception as e:
                    print(c(f"  ✗ {e}", RED))

            elif cmd == "sql":
                sql = input(c("  SQL> ", YELLOW)).strip()
                try:
                    result = session.editor.run(sql)
                    print(c(f"  ✓ {result['rows']} rows in {result['duration_ms']:.0f}ms", GREEN))
                    print(result["df"].head(10).to_string())
                except Exception as e:
                    print(c(f"  ✗ {e}", RED))
            elif cmd == "tables":
                tables = session.connector.get_tables()
                if not tables:
                    print(c("  No tables found.", DIM))
                else:
                    print()
                    for t in tables:
                        print(f"  {c(t['table'], WHITE):<30} {c(str(t['rows']), YELLOW)} rows  {c(str(t['columns']), DIM)} cols")
                    print()

            elif cmd.startswith("schema "):
                table = cmd.split(" ", 1)[1].strip()
                try:
                    info = session.connector.get_table_info(table)
                    if not info:
                        print(c(f"  Table '{table}' not found.", RED))
                    else:
                        print()
                        print(c(f"  {table}", BOLD))
                        print(f"  Rows: {c(str(info.row_count), YELLOW)}")
                        print()
                        for col in info.columns:
                            pk  = c(" PK", GREEN) if col.primary_key else ""
                            fk  = c(f" FK→{col.foreign_key}", DIM) if col.foreign_key else ""
                            null= c(" nullable", DIM) if col.nullable else ""
                            print(f"  {c(col.name, WHITE):<25} {c(col.dtype, CYAN)}{pk}{fk}{null}")
                        print()
                except Exception as e:
                    print(c(f"  ✗ {e}", RED))

            elif cmd.startswith("export "):
                table = cmd.split(" ", 1)[1].strip()
                try:
                    data     = session.editor.run_table(table, limit=100_000)
                    out_path = f"{table}_export.csv"
                    data["df"].to_csv(out_path, index=False, encoding="utf-8")
                    print(c(f"  ✓ Exported {len(data['df']):,} rows to {out_path}", GREEN))
                except Exception as e:
                    print(c(f"  ✗ {e}", RED))

            elif cmd.startswith("clean "):
                table = cmd.split(" ", 1)[1].strip()
                try:
                    data     = session.editor.run_table(table, limit=50_000)
                    from src.data.analyzer     import full_report, detect_outliers
                    from src.data.ml_readiness import ml_readiness
                    from src.data.cleaner      import handle_missing, remove_duplicates
                    analysis = full_report(data)
                    outliers = detect_outliers(data)
                    ml       = ml_readiness(data, outliers)
                    cleaned, _ = handle_missing(data, strategy="mean")
                    cleaned, _ = remove_duplicates(cleaned)
                    out_path   = f"{table}_cleaned.csv"
                    cleaned["df"].to_csv(out_path, index=False)
                    print()
                    print(c(f"  ── Analysis: {table} ──────────────────────────────", CYAN))
                    print(f"  ML Score  : {c(str(ml['score']), GREEN)}/100")
                    print(f"  Missing   : {c(str(sum(analysis['missing_values'].values())), YELLOW)}")
                    print(f"  Duplicates: {c(str(analysis['duplicate_rows']), YELLOW)}")
                    # Handle OutlierReport object or dict
                    if hasattr(outliers, "method_results"):
                        outlier_cnt = len(set(r.column for r in outliers.method_results if r.column and r.outlier_count > 0))
                    else:
                        outlier_cnt = len(outliers)
                    print(f"  Outlier cols: {c(str(outlier_cnt), YELLOW)}")
                    print(c(f"  ✓ Cleaned file saved: {out_path}", GREEN))
                    print()
                except Exception as e:
                    print(c(f"  ✗ {e}", RED))        

            else:
                print(c("  Unknown command. Try: analyse <table> | sql | exit", DIM))

    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))

    print()    

def cmd_dna(args: list[str]) -> None:
    path = require_file(args)
    banner()
    print(c(f"  🧬 Cognitive Data DNA: {path}", CYAN))
    print(SEPARATOR)

    data   = safe_load(path)
    print()
    target = input(c("  Target column (optional, press Enter to skip): ", YELLOW)).strip() or None
    print()
    print(c("  Computing DNA...", DIM))
    print()

    try:
        dna = CognitiveDNA(data, target_col=target)

        print(c("  ── Identity ─────────────────────────────────────────", CYAN))
        print(f"  {c('DNA Hash', BOLD)}    : {c(dna.short_hash, GREEN)}")
        print(f"  {c('Personality', BOLD)} : {c(dna.personality, YELLOW)}")
        print(f"  {c('Tags', BOLD)}        : {c(', '.join(dna.tags), CYAN)}")
        print(f"  {c('Created', BOLD)}     : {c(dna.created_at[:19], DIM)}")
        print()

        print(c("  ── Statistical DNA ──────────────────────────────────", CYAN))
        print(f"  Correlation hash : {c(dna.statistical.correlation_hash, WHITE)}")
        top_entropy = sorted(
            zip(data['df'].columns, dna.statistical.entropy_vector),
            key=lambda x: x[1], reverse=True
        )[:3]
        print(f"  Top entropy cols : {c(', '.join(f'{col}({e:.2f})' for col, e in top_entropy), WHITE)}")
        print()

        print(c("  ── Structural DNA ───────────────────────────────────", CYAN))
        print(f"  Schema hash      : {c(dna.structural.schema_hash, WHITE)}")
        print(f"  Missing mechanism: {c(dna.structural.missing_mechanism, YELLOW if dna.structural.missing_mechanism != 'NONE' else GREEN)}")
        print(f"  Outlier density  : {c(str(dna.structural.outlier_density), WHITE)}")
        print(f"  Duplicate ratio  : {c(str(dna.structural.duplicate_ratio), WHITE)}")
        print()

        print(c("  ── ML DNA ───────────────────────────────────────────", CYAN))
        sig_color = GREEN if dna.ml.target_signal == "strong" else YELLOW if dna.ml.target_signal == "medium" else RED
        print(f"  Task             : {c(dna.ml.recommended_task, CYAN)}")
        print(f"  Target signal    : {c(dna.ml.target_signal, sig_color)}")
        print(f"  Separability     : {c(str(dna.ml.separability_score), WHITE)}")
        print(f"  Redundancy       : {c(str(dna.ml.feature_redundancy), WHITE)}")
        print(f"  Complexity       : {c(str(dna.ml.complexity_score), WHITE)}")
        print()

        print(c("  ── Temporal DNA ─────────────────────────────────────", CYAN))
        print(f"  Has temporal     : {c(str(dna.temporal.has_temporal), GREEN if dna.temporal.has_temporal else DIM)}")
        if dna.temporal.has_temporal:
            print(f"  Temporal cols    : {c(', '.join(dna.temporal.temporal_columns), WHITE)}")
            print(f"  Trend detected   : {c(str(dna.temporal.trend_detected), WHITE)}")
            print(f"  Seasonality      : {c(str(dna.temporal.seasonality_score), WHITE)}")

    except Exception as e:
        print(c(f"  ✗ Error: {e}", RED))

    print()

def cmd_image(args: list[str]) -> None:
    if not IMAGE_V6_AVAILABLE:
        print(c("  ✗ Error: Image modules (vision_nlp) not found or missing dependencies.", RED))
        return

    if not args:
        print(c("  ✗ Error: no directory path provided.", RED))
        sys.exit(1)
    
    path = args[0]
    do_clean = "--clean" in args
    
    if not os.path.exists(path):
        print(c(f"  ✗ Path not found: {path}", RED))
        sys.exit(1)

    banner()
    print(c(f"  🖼️  Professional Image Audit: {path}", CYAN))
    if do_clean:
        print(c("  🧹 Auto-Repair mode enabled", YELLOW))
    print(SEPARATOR)

    try:
        orchestrator = AuditOrchestrator(target_dir=path, verbose=False)
        
        print(c("  1/4 Running Full Pipeline (Load, Quality, Analyze)...", DIM))
        res = orchestrator.run_full_audit(clean=do_clean)
        
        if res["status"] != "success":
            print(c(f"  ✗ Audit failed: {res.get('message')}", RED))
            return

        print(f"  ✓ Found {c(str(len(res['loader_df'])), GREEN)} valid images")
        mean_score = res["quality_report"].mean_overall_score
        print(f"  ✓ Mean Quality Score: {c(f'{mean_score:.1f}/100', GREEN)}")
        print()

        # Deep Insights from ImageAnalyzer
        if "analyzer_report" in res:
            report = res["analyzer_report"]
            print(c("  ── Deep Insights ────────────────────────────────────", CYAN))

            # 1. Class Balance
            if report.class_balance:
                cb = report.class_balance
                imb = "Balanced" if cb.imbalance_ratio < 1.5 else "Moderate Imbalance" if cb.imbalance_ratio < 3 else "Severe Imbalance"
                color = GREEN if imb == "Balanced" else YELLOW if "Moderate" in imb else RED
                print(f"  {c('Class Balance', BOLD):<18} : {c(imb, color)} (Ratio: {cb.imbalance_ratio:.2f})")

            # 2. Diversity
            if report.diversity_redundancy:
                div = report.diversity_redundancy
                print(f"  {c('Diversity Index', BOLD):<18} : {c(f'{div.vendi_score:.2f}', WHITE)} (Vendi Score)")
                if div.near_duplicate_clusters:
                    print(f"  {c('Redundancy', BOLD):<18} : {c(f'⚠️ Found {len(div.near_duplicate_clusters)} groups of duplicates', YELLOW)}")

            # 3. Outliers
            if report.outlier_detection:
                out = report.outlier_detection
                if out.outlier_images:
                    print(f"  {c('Outliers', BOLD):<18} : {c(f'Found {len(out.outlier_images)} abnormal images ({out.outlier_rate:.1%})', RED)}")
                else:
                    print(f"  {c('Outliers', BOLD):<18} : {c('None detected', GREEN)}")

            # 4. ML Readiness
            if report.ml_readiness:
                ml = report.ml_readiness
                sc = GREEN if ml.final_score > 80 else YELLOW if ml.final_score > 60 else RED
                print(f"  {c('ML Readiness', BOLD):<18} : {c(f'{ml.final_score:.1f}/100', sc)} [{c(ml.grade, sc)} - {ml.verdict}]")
                if ml.priority_issues:
                    print(c("\n  Priority Fixes:", RED))
                    for issue in ml.priority_issues[:3]:
                        print(f"  {c('!', RED)} {issue}")
            print()

        if do_clean and "cleaner_df" in res:
            cdf = res["cleaner_df"]
            if not cdf.empty and "cleaned" in cdf.columns:
                n_cleaned = len(cdf[cdf["cleaned"] == True])
                print(f"  ✓ Auto-Repair: {c(str(n_cleaned), GREEN)} images fixed")
            else:
                print(f"  ✓ Auto-Repair: {c('0', GREEN)} images fixed (no repairs needed or possible)")
            print()

        print(c("  3/4 Generating professional reports...", DIM))
        report_paths = orchestrator.generate_reports(output_dir="reports")

        print()
        print(c("  ── Reports Generated ────────────────────────────────", CYAN))
        for fmt, p in report_paths.items():
            print(f"  ✓ {fmt.upper():<5} : {c(str(p), GREEN)}")
        print()

    except Exception as e:
        print(c(f"  ✗ Unexpected Error: {e}", RED))
        import traceback
        traceback.print_exc()

def cmd_quality(args):
    import pandas as pd
    from src.data.quality_score import DataQualityScorer
    df      = pd.read_csv(args[0])
    scorer  = DataQualityScorer()
    profile = scorer.score(df)
    scorer.print_report(profile)    


def cmd_interactive() -> None:
    banner()
    print(c("  Interactive Mode — type 'exit' anytime to quit.", CYAN))
    print(SEPARATOR)

    while True:
        print()

        while True:
            path = input(c("  File path — CSV, Excel, JSON (or 'exit'): ", YELLOW)).strip().strip('"')
            if path.lower() == "exit":
                print(c("\n  Goodbye! 👋\n", GREEN))
                return
            if os.path.exists(path):
                break
            print(c(f"  ✗ File not found: {path}. Try again.", RED))

        print()
        print(c("  What do you want to do?", BOLD))
        options = {
            "1":  ("Full inspection report",     "inspect"),
            "2":  ("Show column statistics",     "stats"),
            "3":  ("Show missing values",        "missing"),
            "4":  ("Detect duplicates",          "duplicates"),
            "5":  ("Detect outliers",            "outliers"),
            "6":  ("Clean and save to new file", "export"),
            "7":  ("Generate HTML report",       "report"),
            "8":  ("🚀 Multimodal AI Impute",    "ai-impute"),
            "9":  ("ML Readiness Score",         "ml"),
            "10": ("Prepare data for ML",        "prepare"),
            "11": ("Column relationships",       "relations"),
            "12": ("AI smart suggestions",       "suggest"),
            "13": ("Data memory & history",      "memory"),
            "14": ("Drift detection",            "drift"),
            "15": ("Auto feature engineering",   "engineer"),
            "16": ("Target column detection",    "target"),
            "17": ("Correlation network",        "network"),
            "18": ("Smart Encoding Advisor",     "encoding"),
            "19": ("Schema Validator",           "schema"),
            "20": ("🏆 Advanced Auto ML (Optuna)", "automl"),
            "21": ("Feature importance (SHAP)",  "importance"),
            "22": ("Train/test split advisor",   "split"),
            "23": ("Class imbalance detector",   "imbalance"),
            "24": ("Pipeline export (sklearn)",  "pipeline"),
            "25": ("Cognitive Data DNA",         "dna"),
            "26": ("🖼️ Image Audit",           "image"),
            "27": ("Database connector",         "connect"),

        }
        for key, (label, _) in options.items():
            print(f"  {c(key, YELLOW)}. {label}")

        choice = input(c("\n  Your choice (1-27): ", YELLOW)).strip()
        action = options.get(choice, ("", "inspect"))[1]

        strategy = "mean"
        if action in ("inspect", "export", "report", "prepare", "suggest"):
            print()
            print(c("  Missing value strategy:", BOLD))
            strategies = {"1": "mean", "2": "median", "3": "mode", "4": "drop"}
            for k, v in strategies.items():
                print(f"  {c(k, YELLOW)}. {v}")
            sc = input(c("\n  Your choice (1-4, default=1): ", YELLOW)).strip()
            strategy = strategies.get(sc, "mean")

        print()
        print(SEPARATOR)

        file_args = [path]
        if action == "inspect":
            file_args += ["--strategy", strategy]
            cmd_inspect(file_args)
        elif action == "stats":
            cmd_stats(file_args)
        elif action == "missing":
            cmd_missing(file_args)
        elif action == "duplicates":
            cmd_duplicates(file_args)
        elif action == "outliers":
            cmd_outliers(file_args)
        elif action == "export":
            file_args += ["--strategy", strategy]
            cmd_export(file_args)
        elif action == "report":
            file_args += ["--strategy", strategy]
            cmd_report(file_args)
        elif action == "ml":
            cmd_ml(file_args)
        elif action == "quality":  
            from src.data.quality_score import DataQualityScorer
            import pandas as pd
            df = pd.read_csv(file_args[0])
            scorer  = DataQualityScorer()
            profile = scorer.score(df)
            scorer.print_report(profile)    
        elif action == "prepare":
            file_args += ["--strategy", strategy]
            cmd_prepare(file_args)
        elif action == "relations":
            cmd_relationships(file_args)
        elif action == "suggest":
            cmd_suggest(file_args)
        elif action == "memory":
            print(c("  Memory sub-commands:", BOLD))
            print(f"  1. List tracked files")
            print(f"  2. Compare last 2 inspections of current file")
            print(f"  3. Clear history for current file")
            m_choice = input(c("\n  Your choice (1-3, default=1): ", YELLOW)).strip() or "1"
            if m_choice == "2":
                cmd_memory(["compare", path])
            elif m_choice == "3":
                cmd_memory(["clear", path])
            else:
                cmd_memory(["list"])
        elif action == "drift":
            print(c("\n  Drift Detection requires a baseline file.", BOLD))
            base_p = input(c("  Baseline file path: ", YELLOW)).strip().strip('"')
            if os.path.exists(base_p):
                cmd_drift([base_p, path])
            else:
                print(c(f"  ✗ Baseline file not found: {base_p}", RED))
        elif action == "engineer":
            cmd_engineer(file_args)
        elif action == "target":
            cmd_target(file_args)
        elif action == "network":
            cmd_network(file_args)
        elif action == "encoding":
            cmd_encoding(file_args)
        elif action == "schema":
            cmd_schema(file_args)
        elif action == "automl":
            cmd_automl(file_args)
        elif action == "importance":
            cmd_importance(file_args)
        elif action =="split":
            cmd_split(file_args)
        elif action == "imbalance":
            cmd_imbalance(file_args)
        elif action == "pipeline":
            cmd_pipeline(file_args)
        elif action == "dna":
            cmd_dna(file_args)
        elif action == "image":
            cmd_image(file_args)
        elif action == "connect":
            cmd_connect([])    

        print(c("  ✓ Done! Ready for next operation.", GREEN))
        print(SEPARATOR)


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "inspect":    cmd_inspect,
    "clean":      cmd_clean,
    "stats":      cmd_stats,
    "missing":    cmd_missing,
    "duplicates": cmd_duplicates,
    "outliers":   cmd_outliers,
    "export":     cmd_export,
    "report":     cmd_report,
    "ml":         cmd_ml,
    "prepare":    cmd_prepare,
    "relations":  cmd_relationships,
    "suggest":    cmd_suggest,
    "memory":     cmd_memory,
    "drift":      cmd_drift,
    "engineer":   cmd_engineer,
    "target":     cmd_target,
    "network":    cmd_network,
    "encoding":   cmd_encoding,
    "schema":     cmd_schema,
    "automl":     cmd_automl,
    "importance": cmd_importance,
    "split":      cmd_split,
    "imbalance":  cmd_imbalance,
    "pipeline":   cmd_pipeline,
    "dna":        cmd_dna,
    "image":      cmd_image,
    "text":       cmd_text,
    "audit-train": cmd_audit_train,
    "connect": cmd_connect,
    "quality" : cmd_quality,
    "report" : cmd_report,  
}


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        usage()
        return

    cmd = args[0].lower()

    if cmd == "interactive":
        cmd_interactive()
        return

    if cmd == "memory":
        cmd_memory(args[1:])
        return

    if cmd == "drift":
        cmd_drift(args[1:])
        return

    if cmd not in COMMANDS:
        print(c(f"\n  ✗ Unknown command: '{cmd}'", RED))
        usage()
        sys.exit(1)

    COMMANDS[cmd](args[1:])


if __name__ == "__main__":
    main()
