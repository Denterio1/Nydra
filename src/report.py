"""
report.py — Generate a professional HTML report for nydra.
"""

from __future__ import annotations
import os
from typing import Any


def generate_html_report(result: dict[str, Any], out_path: str) -> None:
    raw       = result.get("raw_analysis", {})
    shape     = raw.get("shape", {})
    missing   = raw.get("missing_values", {})
    dupes     = raw.get("duplicate_rows", 0)
    stats     = raw.get("column_stats", {})
    outliers  = result.get("outliers", {})
    cleaning  = result.get("cleaning_log", {})
    source    = result.get("source", "unknown")

    # Health score
    score = 100
    missing_cols = {k: v for k, v in missing.items() if v > 0}
    score -= len(missing_cols) * 5
    score -= min(dupes * 2, 20)
    
    # Handle OutlierReport object or dict
    if hasattr(outliers, "method_results"):
        outlier_cols_count = len(set(r.column for r in outliers.method_results if r.column and r.outlier_count > 0))
    else:
        outlier_cols_count = len(outliers)
        
    score -= outlier_cols_count * 5
    score = max(0, min(100, score))

    if score >= 80:
        grade_color = "#c8f06e"
        grade_label = "Good"
    elif score >= 60:
        grade_color = "#f0c46e"
        grade_label = "Fair"
    else:
        grade_color = "#f06e6e"
        grade_label = "Poor"

    # Missing values rows
    missing_rows = ""
    for col, cnt in missing.items():
        if cnt > 0:
            pct = round(cnt / shape.get("rows", 1) * 100, 1)
            bar = int(pct / 5)
            missing_rows += f"""
            <tr>
              <td>{col}</td>
              <td>{cnt}</td>
              <td>{pct}%</td>
              <td><div class="bar" style="width:{min(pct*3,100)}%"></div></td>
            </tr>"""

    # Outlier rows
    outlier_rows = ""
    for col, info in outliers.items():
        outlier_rows += f"""
        <tr>
          <td>{col}</td>
          <td>{info['count']}</td>
          <td>{info['lower']} → {info['upper']}</td>
          <td>{str(info['values'][:3])}{'...' if len(info['values']) > 3 else ''}</td>
        </tr>"""

    # Column stats cards
    stat_cards = ""
    for col, s in stats.items():
        if s["type"] == "numeric":
            detail = f"min {s['min']} / max {s['max']} / mean {s['mean']}"
        else:
            detail = f"most common: <b>{s.get('most_common','—')}</b>"
        stat_cards += f"""
        <div class="card">
          <div class="card-title">{col}</div>
          <div class="card-badge">{'numeric' if s['type']=='numeric' else 'text'}</div>
          <div class="card-detail">{detail}</div>
          <div class="card-detail">unique: {s['unique']}</div>
        </div>"""

    # Cleaning log rows
    clean_rows = ""
    if isinstance(cleaning, dict):
        filled = cleaning.get("missing_filled", {})
        for col, info in filled.items():
            clean_rows += f"<li>'{col}': filled {info.get('filled','?')} value(s) → {info.get('value','?')}</li>"
        dr = cleaning.get("duplicates_removed", 0)
        if dr:
            clean_rows += f"<li>Removed {dr} duplicate row(s)</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nydra Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',sans-serif;background:#0f0f11;color:#e8e6e1;padding:2rem}}
  h1{{font-size:2rem;color:#c8f06e;margin-bottom:.25rem}}
  .sub{{color:#888;font-size:.9rem;margin-bottom:2rem}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:2rem}}
  .metric{{background:#17171a;border-radius:12px;padding:1.25rem;text-align:center}}
  .metric-val{{font-size:2rem;font-weight:700;color:#c8f06e}}
  .metric-lbl{{font-size:.8rem;color:#888;margin-top:.25rem}}
  .score-val{{font-size:3rem;font-weight:700;color:{grade_color}}}
  section{{margin-bottom:2rem}}
  h2{{font-size:1.1rem;color:#c8f06e;border-bottom:1px solid #2a2a2e;padding-bottom:.5rem;margin-bottom:1rem}}
  table{{width:100%;border-collapse:collapse;background:#17171a;border-radius:8px;overflow:hidden}}
  th{{background:#1e1e22;padding:.75rem 1rem;text-align:left;font-size:.8rem;color:#888;text-transform:uppercase}}
  td{{padding:.65rem 1rem;border-bottom:1px solid #2a2a2e;font-size:.9rem}}
  .bar{{height:8px;background:#c8f06e;border-radius:4px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1rem}}
  .card{{background:#17171a;border-radius:10px;padding:1rem}}
  .card-title{{font-weight:600;margin-bottom:.35rem}}
  .card-badge{{display:inline-block;font-size:.7rem;padding:.2rem .5rem;border-radius:20px;background:#2a2a2e;color:#888;margin-bottom:.5rem}}
  .card-detail{{font-size:.82rem;color:#aaa;margin-top:.2rem}}
  ul{{padding-left:1.25rem;color:#aaa;font-size:.9rem;line-height:1.8}}
  footer{{margin-top:3rem;text-align:center;color:#444;font-size:.8rem}}
</style>
</head>
<body>
<h1>🩺 Nydra Report</h1>
<div class="sub">File: {os.path.basename(source)}</div>

<div class="grid">
  <div class="metric"><div class="score-val">{score}</div><div class="metric-lbl">Health Score / 100 — {grade_label}</div></div>
  <div class="metric"><div class="metric-val">{shape.get('rows','?')}</div><div class="metric-lbl">Rows</div></div>
  <div class="metric"><div class="metric-val">{shape.get('columns','?')}</div><div class="metric-lbl">Columns</div></div>
  <div class="metric"><div class="metric-val">{len(missing_cols)}</div><div class="metric-lbl">Cols with Missing</div></div>
  <div class="metric"><div class="metric-val">{dupes}</div><div class="metric-lbl">Duplicate Rows</div></div>
  <div class="metric"><div class="metric-val">{outlier_cols_count}</div><div class="metric-lbl">Outlier Cols</div></div>
</div>

{'<section><h2>⚠️ Missing Values</h2><table><thead><tr><th>Column</th><th>Missing</th><th>%</th><th>Bar</th></tr></thead><tbody>' + missing_rows + '</tbody></table></section>' if missing_rows else ''}

{'<section><h2>🔴 Outliers</h2><table><thead><tr><th>Column</th><th>Count</th><th>Safe Range</th><th>Values</th></tr></thead><tbody>' + outlier_rows + '</tbody></table></section>' if outlier_rows else ''}

<section><h2>📊 Column Statistics</h2><div class="cards">{stat_cards}</div></section>

{'<section><h2>🧹 Cleaning Actions</h2><ul>' + clean_rows + '</ul></section>' if clean_rows else ''}

<footer>Generated by Nydra</footer>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
