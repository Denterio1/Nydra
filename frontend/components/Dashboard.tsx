"use client";

/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║           NYDRA — Dashboard.tsx                                 ║
 * ║   Real-time job progress + full results visualization           ║
 * ║                                                                  ║
 * ║  ✅ Live WebSocket progress via useNydraWS                      ║
 * ║  ✅ Step-by-step pipeline tracker                               ║
 * ║  ✅ Quality score radial gauge                                  ║
 * ║  ✅ Issues list with severity badges                            ║
 * ║  ✅ Column stats table                                          ║
 * ║  ✅ Warnings feed                                               ║
 * ║  ✅ Result download + report generation                         ║
 * ║  ✅ Skeleton loading states                                     ║
 * ║  ✅ Terminal / dark industrial aesthetic                        ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */

import React, { useEffect, useRef, useState } from "react";
import {
  useJobProgress,
  useNydraWS,
  useWsConnectionIndicator,
  UseNydraWSOptions,
  type WSResultPayload,
  type WSWarningPayload,
  type ConnectionState,
} from "../hooks/useNydraWS";

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

interface ColumnStat {
  name: string;
  dtype: string;
  inferred_type: string;
  missing: number;
  missing_pct: number;
  unique: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  top_values?: { value: string; count: number }[];
}

interface Issue {
  issue_id: string;
  code: string;
  title: string;
  description: string;
  severity: "critical" | "high" | "medium" | "low" | "info";
  affected_columns: string[];
  suggestion: string;
  auto_fixable: boolean;
}

interface QualityDimension {
  name: string;
  score: number;
  weight: number;
  issues: string[];
}

interface JobResultData {
  overview?: {
    rows: number;
    columns: number;
    file_size_mb: number;
    file_type?: string;
    total_missing: number;
    missing_pct: number;
    total_duplicates: number;
    duplicate_pct: number;
    column_stats: ColumnStat[];
  };
  quality?: {
    overall_score: number;
    verdict: "ready" | "needs_work" | "not_ready";
    grade: "A" | "B" | "C" | "D" | "F";
    dimensions: QualityDimension[];
    critical_issues: string[];
    fix_checklist: string[];
    estimated_model_impact: string;
  };
  issues?: Issue[];
  warnings?: string[];
}

export interface DashboardProps {
  jobId: string | null;
  token: string;
  goal: string;
  filename: string;
  onNewAnalysis?: () => void;
  workspace?: "DIAGNOSTICS" | "VISION_LAB" | "TEXT_INTELLIGENCE" | "REPAIR_SHOP";
  apiUrl?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const SEVERITY_CONFIG = {
  critical: { color: "#f43f5e", bg: "rgba(244,63,94,0.08)", label: "CRITICAL", icon: "⬟" },
  high:     { color: "#f97316", bg: "rgba(249,115,22,0.08)", label: "HIGH",     icon: "⬡" },
  medium:   { color: "#f59e0b", bg: "rgba(245,158,11,0.08)", label: "MEDIUM",   icon: "◈" },
  low:      { color: "#3b82f6", bg: "rgba(59,130,246,0.08)", label: "LOW",      icon: "◇" },
  info:     { color: "#64748b", bg: "rgba(100,116,139,0.08)", label: "INFO",    icon: "○" },
};

const VERDICT_CONFIG = {
  ready:      { color: "#22c55e", label: "READY FOR ML",  icon: "✓" },
  needs_work: { color: "#f59e0b", label: "NEEDS WORK",    icon: "⚠" },
  not_ready:  { color: "#f43f5e", label: "NOT READY",     icon: "✕" },
};

const GRADE_COLORS = { A: "#22c55e", B: "#84cc16", C: "#f59e0b", D: "#f97316", F: "#f43f5e" };

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

function Skeleton({ w = "100%", h = 16, radius = 4 }: { w?: string | number; h?: number; radius?: number }) {
  return (
    <div style={{
      width: w, height: h, borderRadius: radius,
      background: "linear-gradient(90deg, rgba(30,41,59,0.8) 25%, rgba(51,65,85,0.5) 50%, rgba(30,41,59,0.8) 75%)",
      backgroundSize: "200% 100%",
      animation: "nydra-shimmer 1.5s infinite",
    }} />
  );
}

function ConnectionBadge({ state, retryCount }: { state: ConnectionState; retryCount: number }) {
  const indicator = useWsConnectionIndicator(state);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{
        width: 6, height: 6, borderRadius: "50%",
        background: indicator.color,
        boxShadow: indicator.pulsing ? `0 0 6px ${indicator.color}` : "none",
        animation: indicator.pulsing ? "nydra-pulse 1.5s ease-in-out infinite" : "none",
      }} />
      <span style={{
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 9, fontWeight: 700, letterSpacing: 1.5,
        color: indicator.color,
      }}>
        {indicator.label}
        {retryCount > 0 && ` (retry ${retryCount})`}
      </span>
    </div>
  );
}

function RadialGauge({ score, grade }: { score: number; grade: string }) {
  const radius  = 54;
  const circ    = 2 * Math.PI * radius;
  const offset  = circ - (score / 100) * circ;
  const color   = GRADE_COLORS[grade as keyof typeof GRADE_COLORS] ?? "#00d4aa";

  return (
    <div style={{ position: "relative", width: 140, height: 140 }}>
      <svg width={140} height={140} style={{ transform: "rotate(-90deg)" }}>
        {/* Track */}
        <circle cx={70} cy={70} r={radius} fill="none"
          stroke="rgba(51,65,85,0.4)" strokeWidth={8} />
        {/* Progress */}
        <circle cx={70} cy={70} r={radius} fill="none"
          stroke={color} strokeWidth={8}
          strokeDasharray={circ}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{
            transition: "stroke-dashoffset 1s ease-out",
            filter: `drop-shadow(0 0 8px ${color}88)`,
          }}
        />
      </svg>
      {/* Center text */}
      <div style={{
        position: "absolute", inset: 0,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
      }}>
        <span style={{
          fontFamily: "Syne, sans-serif",
          fontSize: 32, fontWeight: 800, color,
          lineHeight: 1,
        }}>
          {score}
        </span>
        <span style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 10, color: "#475569", marginTop: 2,
        }}>
          / 100
        </span>
        <span style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 16, fontWeight: 700, color,
          marginTop: 2,
        }}>
          {grade}
        </span>
      </div>
    </div>
  );
}

function StepTracker({ steps, currentStep, isDone }: {
  steps: { name: string; status: string }[];
  currentStep: string;
  isDone: boolean;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {steps.map((step, i) => {
        const stepDone  = step.status === "done";
        const isActive  = step.name === currentStep;
        const color     = stepDone ? "#22c55e" : isActive ? "#00d4aa" : "#1e293b";
        const textColor = stepDone ? "#22c55e" : isActive ? "#00d4aa" : "#334155";

        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, position: "relative" }}>
            {/* Node */}
            <div style={{
              width: 20, height: 20, borderRadius: "50%", flexShrink: 0,
              background: stepDone ? "rgba(34,197,94,0.12)" : isActive ? "rgba(0,212,170,0.12)" : "rgba(15,23,42,0.6)",
              border: `1.5px solid ${color}`,
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.3s",
            }}>
              {stepDone
                ? <span style={{ color: "#22c55e", fontSize: 10 }}>✓</span>
                : isActive
                ? <div style={{
                    width: 6, height: 6, borderRadius: "50%",
                    background: "#00d4aa",
                    animation: "nydra-pulse 1s ease-in-out infinite",
                  }} />
                : <div style={{ width: 4, height: 4, borderRadius: "50%", background: "#1e293b" }} />
              }
            </div>
            {/* Connector line */}
            {i < steps.length - 1 && (
              <div style={{
                position: "absolute",
                left: 9, top: 20,
                width: 1.5, height: 4,
                background: stepDone ? "rgba(34,197,94,0.4)" : "rgba(30,41,59,0.8)",
              }} />
            )}
            {/* Label */}
            <span style={{
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 11, color: textColor,
              transition: "color 0.3s",
              animation: isActive ? "nydra-pulse 1.5s ease-in-out infinite" : "none",
            }}>
              {step.name}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ProgressSection({ progress, currentStep, elapsedMs, etaMs, isConnected, retryCount }: {
  progress: number;
  currentStep: string;
  elapsedMs: number;
  etaMs: number | null;
  isConnected: boolean;
  retryCount: number;
}) {
  const formatTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    return m > 0 ? `${m}m ${s % 60}s` : `${s}s`;
  };

  return (
    <div style={{
      background: "rgba(8,14,27,0.8)",
      border: "1px solid rgba(0,212,170,0.15)",
      borderRadius: 10, padding: 20,
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 10, fontWeight: 700, letterSpacing: 2,
          color: "#00d4aa",
        }}>
          ANALYSIS IN PROGRESS
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 10, color: "#475569" }}>
            {formatTime(elapsedMs)} elapsed
          </span>
          {etaMs && (
            <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 10, color: "#334155" }}>
              ~{formatTime(etaMs)} remaining
            </span>
          )}
        </div>
      </div>

      {/* Main progress bar */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
          <span style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 11, color: "#94a3b8",
          }}>
            {currentStep || "Initializing..."}
          </span>
          <span style={{
            fontFamily: "Syne, sans-serif",
            fontSize: 18, fontWeight: 800, color: "#00d4aa",
          }}>
            {progress}%
          </span>
        </div>
        <div style={{
          width: "100%", height: 6, background: "rgba(15,23,42,0.8)",
          borderRadius: 3, overflow: "hidden",
        }}>
          <div style={{
            height: "100%", width: `${progress}%`,
            background: "linear-gradient(90deg, #00d4aa, #7c3aed)",
            borderRadius: 3,
            boxShadow: "0 0 12px rgba(0,212,170,0.4)",
            transition: "width 0.4s ease-out",
          }} />
        </div>
      </div>

      {/* Connection + retry info */}
      {retryCount > 0 && (
        <div style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 10, color: "#f97316",
          padding: "6px 10px", borderRadius: 4,
          background: "rgba(249,115,22,0.06)",
          border: "1px solid rgba(249,115,22,0.2)",
        }}>
          ↻ Reconnecting... (attempt {retryCount})
        </div>
      )}
    </div>
  );
}

function IssueCard({ issue }: { issue: Issue }) {
  const cfg     = SEVERITY_CONFIG[issue.severity];
  const [open, setOpen] = useState(false);

  return (
    <div style={{
      background: cfg.bg,
      border: `1px solid ${cfg.color}22`,
      borderLeft: `3px solid ${cfg.color}`,
      borderRadius: 6, padding: "10px 14px",
      cursor: "pointer",
    }} onClick={() => setOpen(!open)}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
        <span style={{ color: cfg.color, fontSize: 12, marginTop: 1, flexShrink: 0 }}>
          {cfg.icon}
        </span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <span style={{
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 9, fontWeight: 700, letterSpacing: 1.5,
              color: cfg.color, flexShrink: 0,
            }}>
              {cfg.label}
            </span>
            <span style={{
              fontFamily: "Syne, sans-serif",
              fontSize: 13, fontWeight: 600, color: "#e2e8f0",
            }}>
              {issue.title}
            </span>
            {issue.auto_fixable && (
              <span style={{
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 9, color: "#22c55e", letterSpacing: 1,
                padding: "1px 6px", borderRadius: 3,
                background: "rgba(34,197,94,0.1)",
                border: "1px solid rgba(34,197,94,0.2)",
              }}>
                AUTO-FIX
              </span>
            )}
          </div>
          {open && (
            <div style={{ marginTop: 8 }}>
              <p style={{
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 11, color: "#64748b", margin: "0 0 8px 0",
              }}>
                {issue.description}
              </p>
              <div style={{
                padding: "8px 10px", borderRadius: 4,
                background: "rgba(0,212,170,0.05)",
                border: "1px solid rgba(0,212,170,0.1)",
              }}>
                <span style={{
                  fontFamily: "JetBrains Mono, monospace",
                  fontSize: 10, color: "#00d4aa",
                }}>
                  ⬡ {issue.suggestion}
                </span>
              </div>
              {issue.affected_columns.length > 0 && (
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 8 }}>
                  {issue.affected_columns.map((col) => (
                    <span key={col} style={{
                      fontFamily: "JetBrains Mono, monospace",
                      fontSize: 9, color: "#475569",
                      padding: "1px 6px", borderRadius: 3,
                      background: "rgba(71,85,105,0.2)",
                      border: "1px solid rgba(71,85,105,0.3)",
                    }}>
                      {col}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
        <span style={{ color: "#334155", fontSize: 10, flexShrink: 0 }}>
          {open ? "▲" : "▼"}
        </span>
      </div>
    </div>
  );
}

function QualitySection({ quality }: { quality: NonNullable<JobResultData["quality"]> }) {
  const verdict = VERDICT_CONFIG[quality.verdict];

  return (
    <div style={{
      background: "rgba(8,14,27,0.8)",
      border: "1px solid rgba(51,65,85,0.5)",
      borderRadius: 10, padding: 20,
    }}>
      <div style={{
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 10, fontWeight: 700, letterSpacing: 2,
        color: "#475569", marginBottom: 20,
      }}>
        DATA QUALITY SCORE
      </div>

      {/* Gauge + verdict side by side */}
      <div style={{ display: "flex", gap: 24, alignItems: "center", marginBottom: 24 }}>
        <RadialGauge score={quality.overall_score} grade={quality.grade} />
        <div style={{ flex: 1 }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            padding: "6px 14px", borderRadius: 6, marginBottom: 12,
            background: `${verdict.color}12`,
            border: `1px solid ${verdict.color}33`,
          }}>
            <span style={{ color: verdict.color, fontSize: 14 }}>{verdict.icon}</span>
            <span style={{
              fontFamily: "Syne, sans-serif",
              fontSize: 14, fontWeight: 700, color: verdict.color,
            }}>
              {verdict.label}
            </span>
          </div>
          <p style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 10, color: "#475569", margin: 0,
          }}>
            {quality.estimated_model_impact}
          </p>
        </div>
      </div>

      {/* Dimension bars */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {quality.dimensions.map((dim) => (
          <div key={dim.name}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
              <span style={{
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 10, color: "#64748b", letterSpacing: 0.5,
              }}>
                {dim.name.toUpperCase()}
              </span>
              <span style={{
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 10, fontWeight: 700,
                color: dim.score >= 80 ? "#22c55e" : dim.score >= 60 ? "#f59e0b" : "#f43f5e",
              }}>
                {dim.score}
              </span>
            </div>
            <div style={{
              width: "100%", height: 3,
              background: "rgba(15,23,42,0.8)", borderRadius: 2,
            }}>
              <div style={{
                height: "100%", width: `${dim.score}%`,
                background: dim.score >= 80 ? "#22c55e" : dim.score >= 60 ? "#f59e0b" : "#f43f5e",
                borderRadius: 2,
                transition: "width 1s ease-out",
              }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DistributionsView({ jobId, token, apiBase }: { jobId?: string | null; token?: string | null; apiBase: string }) {
  const mono = "JetBrains Mono, monospace";
  const card = { background: "rgba(8,14,27,0.8)", border: "1px solid rgba(51,65,85,0.5)", borderRadius: 10, padding: 16 } as const;
  const label = { fontSize: 10, letterSpacing: "0.12em", color: "#64748b", fontFamily: mono } as const;
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!jobId) { setLoading(false); return; }
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`${apiBase}/api/v1/jobs/${jobId}/distributions`, {
      headers: { Authorization: `Bearer ${token ?? ""}` },
    })
      .then(async (res) => {
        if (res.status === 202) throw new Error("Job is still running. Reopen this tab in a moment.");
        if (!res.ok) throw new Error(`Request failed (${res.status})`);
        return res.json();
      })
      .then((json) => { if (!cancelled) setData(json); })
      .catch((e) => { if (!cancelled) setError(e.message || "Failed to load distributions"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [jobId, token, apiBase]);

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: 40, border: "1px dashed rgba(51,65,85,0.5)", borderRadius: 10 }}>
        <span style={{ fontSize: 10, color: "#64748b", fontFamily: mono }}>
          FITTING DISTRIBUTIONS... (first load can take up to a minute)
        </span>
      </div>
    );
  }
  if (error) {
    return (
      <div style={{ ...card, color: "#f87171", fontSize: 11, fontFamily: mono }}>{error}</div>
    );
  }
  const cols: any[] = data?.columns ?? [];
  if (cols.length === 0) {
    return (
      <div style={{ ...card, color: "#64748b", fontSize: 11, fontFamily: mono }}>
        {data?.note ?? "No numeric columns suitable for distribution analysis."}
      </div>
    );
  }

  const badge = (text: string, color: string) => (
    <span style={{ fontSize: 9, letterSpacing: "0.08em", fontFamily: mono, color, border: `1px solid ${color}55`, borderRadius: 4, padding: "2px 6px", marginRight: 6 }}>
      {text}
    </span>
  );
  const fmt = (v: any, d = 2) => (typeof v === "number" ? v.toFixed(d) : "-");

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 14 }}>
      {cols.map((c) => {
        const counts: number[] = c.histogram?.counts ?? [];
        const edges: number[] = c.histogram?.edges ?? [];
        const max = Math.max(1, ...counts);
        const W = 300, H = 90;
        const bw = counts.length ? W / counts.length : W;
        return (
          <div key={c.column} style={card}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
              <span style={{ fontSize: 13, color: "#e2e8f0", fontFamily: mono }}>{c.column}</span>
              <span style={label}>n = {c.n}</span>
            </div>

            <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
              {counts.map((n, i) => {
                const h = (n / max) * (H - 4);
                return <rect key={i} x={i * bw + 1} y={H - h} width={Math.max(1, bw - 2)} height={h} fill="#38bdf8" opacity={0.85} rx={1} />;
              })}
            </svg>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 2, marginBottom: 10 }}>
              <span style={label}>{edges.length ? fmt(edges[0], 1) : ""}</span>
              <span style={label}>{edges.length ? fmt(edges[edges.length - 1], 1) : ""}</span>
            </div>

            <div style={{ marginBottom: 8 }}>
              {c.normality && badge(c.normality.is_normal ? "NORMAL" : "NOT NORMAL", c.normality.is_normal ? "#34d399" : "#fbbf24")}
              {c.shape?.skewness_type && badge(String(c.shape.skewness_type).replace(/_/g, " ").toUpperCase(), "#94a3b8")}
              {c.shape?.possibly_bimodal && badge("BIMODAL?", "#a78bfa")}
            </div>

            <div style={{ fontSize: 11, color: "#94a3b8", fontFamily: mono, lineHeight: 1.7 }}>
              {c.best_fit && <div>best fit: <span style={{ color: "#e2e8f0" }}>{c.best_fit.distribution}</span> (KS p = {fmt(c.best_fit.ks_p, 3)})</div>}
              {c.shape && <div>skew {fmt(c.shape.skewness)} | excess kurtosis {fmt(c.shape.kurtosis_excess)}</div>}
              {c.transform && (
                <div>
                  transform: {c.transform.needed
                    ? <span style={{ color: "#fbbf24" }}>{c.transform.best ?? "none"} (skew after {fmt(c.transform.skewness_after)})</span>
                    : <span style={{ color: "#34d399" }}>not needed</span>}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function OutliersView({ data }: { data?: any }) {
  const mono = "JetBrains Mono, monospace";
  const card = { background: "rgba(8,14,27,0.8)", border: "1px solid rgba(51,65,85,0.5)", borderRadius: 10, padding: 16 } as const;
  const label = { fontFamily: mono, fontSize: 10, fontWeight: 700, letterSpacing: 2, color: "#475569", marginBottom: 12 } as const;
  if (!data || data.error) {
    return (
      <div style={{ ...card, textAlign: "center", padding: 40 }}>
        <span style={{ fontFamily: mono, fontSize: 10, color: "#475569" }}>NO OUTLIER DATA FOR THIS JOB.</span>
      </div>
    );
  }
  const sevColor: Record<string, string> = { critical: "#f43f5e", high: "#f97316", medium: "#f59e0b", low: "#3b82f6", none: "#22c55e" };
  const color = sevColor[data.severity] ?? "#64748b";
  const breakdown = Object.entries(data.report?.method_breakdown ?? {})
    .map(([k, v]: [string, any]) => {
      const parts = k.split("|");
      return { method: parts[0], column: parts[1] ?? k, count: v?.outlier_count ?? 0, rate: v?.outlier_rate ?? 0 };
    })
    .filter((b) => b.count > 0)
    .sort((a, b) => b.count - a.count);
  const maxCount = Math.max(1, ...breakdown.map((b) => b.count));
  const stats = [
    { label: "OUTLIERS", value: String(data.n_outliers ?? 0), color },
    { label: "RATE", value: `${((data.outlier_rate ?? 0) * 100).toFixed(1)}%`, color },
    { label: "METHOD", value: String(data.method_used ?? "-").toUpperCase(), color: "#7c3aed" },
    { label: "SEVERITY", value: String(data.severity ?? "none").toUpperCase(), color },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
        {stats.map((st) => (
          <div key={st.label} style={{ ...card, padding: "12px 14px" }}>
            <div style={{ fontFamily: mono, fontSize: 9, letterSpacing: 2, color: "#475569" }}>{st.label}</div>
            <div style={{ fontFamily: mono, fontSize: 18, fontWeight: 700, color: st.color, marginTop: 4 }}>{st.value}</div>
          </div>
        ))}
      </div>

      {breakdown.length > 0 && (
        <div style={card}>
          <div style={label}>PER-COLUMN BREAKDOWN</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {breakdown.map((b, i) => (
              <div key={i} style={{ display: "grid", gridTemplateColumns: "160px 1fr 110px", gap: 10, alignItems: "center" }}>
                <span style={{ fontFamily: mono, fontSize: 11, color: "#94a3b8" }}>{b.column}</span>
                <div style={{ background: "rgba(51,65,85,0.3)", borderRadius: 3, height: 8 }}>
                  <div style={{ width: `${(b.count / maxCount) * 100}%`, height: 8, borderRadius: 3, background: color }} />
                </div>
                <span style={{ fontFamily: mono, fontSize: 10, color: "#64748b", textAlign: "right" }}>
                  {b.count} ({(b.rate * 100).toFixed(1)}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.preview?.rows?.length > 0 && (
        <div style={{ ...card, overflowX: "auto" }}>
          <div style={label}>OUTLIER ROWS (FIRST {data.preview.rows.length})</div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: mono, fontSize: 11 }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "4px 8px", color: "#475569" }}>ROW</th>
                {data.preview.columns.map((c: string) => (
                  <th key={c} style={{ textAlign: "left", padding: "4px 8px", color: "#475569" }}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.preview.rows.map((r: any, i: number) => (
                <tr key={i} style={{ borderTop: "1px solid rgba(51,65,85,0.3)" }}>
                  <td style={{ padding: "4px 8px", color: color }}>{r._row}</td>
                  {data.preview.columns.map((c: string) => (
                    <td key={c} style={{ padding: "4px 8px", color: "#94a3b8" }}>{r[c] === null || r[c] === undefined ? "-" : String(r[c])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {Array.isArray(data.recommendations) && data.recommendations.length > 0 && (
        <div style={card}>
          <div style={label}>RECOMMENDATIONS</div>
          {data.recommendations.map((rec: string, i: number) => (
            <div key={i} style={{ fontFamily: mono, fontSize: 11, color: "#94a3b8", padding: "4px 0" }}>{rec}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function OverviewStats({ overview }: { overview: NonNullable<JobResultData["overview"]> }) {
  const stats = [
    { label: "ROWS",       value: overview.rows.toLocaleString(),           color: "#00d4aa" },
    { label: "COLUMNS",    value: overview.columns.toString(),               color: "#7c3aed" },
    { label: "MISSING",    value: `${overview.missing_pct.toFixed(1)}%`,    color: overview.missing_pct > 10 ? "#f43f5e" : "#22c55e" },
    { label: "DUPLICATES", value: `${overview.duplicate_pct.toFixed(1)}%`,  color: overview.duplicate_pct > 5 ? "#f97316" : "#22c55e" },
    { label: "SIZE",       value: `${overview.file_size_mb.toFixed(1)} MB`, color: "#64748b" },
  ];

  return (
    <div style={{
      display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8,
    }}>
      {stats.map((s) => (
        <div key={s.label} style={{
          background: "rgba(8,14,27,0.8)",
          border: "1px solid rgba(51,65,85,0.4)",
          borderRadius: 8, padding: "12px 10px", textAlign: "center",
        }}>
          <div style={{
            fontFamily: "Syne, sans-serif",
            fontSize: 20, fontWeight: 800, color: s.color,
          }}>
            {s.value}
          </div>
          <div style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 8, color: "#334155", letterSpacing: 1.5, marginTop: 4,
          }}>
            {s.label}
          </div>
        </div>
      ))}
    </div>
  );
}

function OverviewView({ overview }: { overview: NonNullable<JobResultData["overview"]> }) {
  const mono = "JetBrains Mono, monospace";
  const card = { background: "rgba(8,14,27,0.8)", border: "1px solid rgba(51,65,85,0.5)", borderRadius: 10, padding: 16 } as const;
  const label = { fontSize: 10, letterSpacing: "0.12em", color: "#64748b", fontFamily: mono } as const;
  const cols = overview.column_stats ?? [];

  const withMissing = cols.filter((c) => c.missing > 0).sort((a, b) => b.missing_pct - a.missing_pct).slice(0, 10);

  const typeCounts: Record<string, number> = {};
  cols.forEach((c) => {
    const t = String(c.inferred_type || c.dtype || "unknown");
    typeCounts[t] = (typeCounts[t] ?? 0) + 1;
  });
  const typeEntries = Object.entries(typeCounts).sort((a, b) => b[1] - a[1]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      
      <div style={card}>
        <div style={{ ...label, marginBottom: 10 }}>MISSING VALUES BY COLUMN</div>
        {withMissing.length === 0 ? (
          <div style={{ fontSize: 11, color: "#22c55e", fontFamily: mono }}>No missing values in any column.</div>
        ) : (
          withMissing.map((c) => (
            <div key={c.name} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <span style={{ width: 130, fontSize: 11, color: "#cbd5e1", fontFamily: mono, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.name}</span>
              <div style={{ flex: 1, height: 8, background: "rgba(51,65,85,0.4)", borderRadius: 4 }}>
                <div style={{ width: `${Math.min(100, c.missing_pct)}%`, height: "100%", borderRadius: 4, background: c.missing_pct > 20 ? "#f43f5e" : c.missing_pct > 5 ? "#f97316" : "#fbbf24" }} />
              </div>
              <span style={{ width: 90, textAlign: "right", fontSize: 10, color: "#94a3b8", fontFamily: mono }}>
                {c.missing_pct.toFixed(1)}% ({c.missing})
              </span>
            </div>
          ))
        )}
      </div>

      <div style={card}>
        <div style={{ ...label, marginBottom: 10 }}>COLUMN TYPES</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {typeEntries.map(([t, n]) => (
            <span key={t} style={{ fontSize: 11, fontFamily: mono, color: "#e2e8f0", border: "1px solid rgba(51,65,85,0.6)", borderRadius: 6, padding: "4px 10px" }}>
              {t}: <span style={{ color: "#00d4aa" }}>{n}</span>
            </span>
          ))}
        </div>
        <div style={{ ...label, marginTop: 12 }}>
          {overview.total_duplicates > 0
            ? `${overview.total_duplicates.toLocaleString()} duplicate rows (${overview.duplicate_pct.toFixed(1)}%)`
            : "No duplicate rows."}
          {" | open the COLUMNS tab for the full table"}
        </div>
      </div>
    </div>
  );
}

function RepairResultView({ info, jobId, token, apiBase }: { info?: any; jobId?: string | null; token?: string | null; apiBase: string }) {
  const mono = "JetBrains Mono, monospace";
  const card = { background: "rgba(8,14,27,0.8)", border: "1px solid rgba(51,65,85,0.5)", borderRadius: 10, padding: 16 } as const;
  const label = { fontSize: 10, letterSpacing: "0.12em", color: "#64748b", fontFamily: mono } as const;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  if (!info) {
    return (
      <div style={{ ...card, color: "#64748b", fontSize: 11, fontFamily: mono }}>
        No repaired file for this job. Upload a file while the Repair Shop workspace is selected.
      </div>
    );
  }

  const download = async () => {
    if (!jobId) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${apiBase}/api/v1/jobs/${jobId}/cleaned`, {
        headers: { Authorization: `Bearer ${token ?? ""}` },
      });
      if (!res.ok) throw new Error(`Download failed (${res.status})`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "cleaned_data.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setErr(e.message || "Download failed");
    } finally {
      setBusy(false);
    }
  };

  const stats = [
    { label: "ROWS", value: info.rows_before != null ? `${info.rows_before} -> ${info.rows_after}` : String(info.rows_after), color: "#00d4aa" },
    { label: "MISSING FILLED", value: String(info.missing_filled ?? 0), color: "#38bdf8" },
    { label: "DUPLICATES REMOVED", value: String(info.duplicates_removed ?? 0), color: "#f97316" },
    { label: "COLUMNS", value: String(info.columns ?? "-"), color: "#7c3aed" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
        {stats.map((s) => (
          <div key={s.label} style={{ ...card, padding: "12px 10px", textAlign: "center" }}>
            <div style={{ fontFamily: "Syne, sans-serif", fontSize: 20, fontWeight: 800, color: s.color }}>{s.value}</div>
            <div style={{ ...label, fontSize: 8, marginTop: 4 }}>{s.label}</div>
          </div>
        ))}
      </div>

      <div style={{ ...card, fontSize: 11, color: "#94a3b8", fontFamily: mono, lineHeight: 1.8 }}>
        {info.impute_method && <div>imputation method: <span style={{ color: "#e2e8f0" }}>{info.impute_method}</span></div>}
        <div>outliers are <span style={{ color: "#e2e8f0" }}>kept</span> in the file; review them in Diagnostics, Outliers tab</div>
        {info.formula_cells_neutralized > 0 && (
          <div style={{ color: "#fbbf24" }}>
            {info.formula_cells_neutralized} cell(s) starting with = + - @ were prefixed with a quote to prevent spreadsheet formula injection
          </div>
        )}
      </div>

      <div>
        <button
          onClick={download}
          disabled={busy}
          style={{
            padding: "10px 18px", borderRadius: 6, cursor: busy ? "wait" : "pointer",
            background: "rgba(0,212,170,0.12)", border: "1px solid rgba(0,212,170,0.5)",
            color: "#00d4aa", fontFamily: mono, fontSize: 11, letterSpacing: "0.1em",
          }}
        >
          {busy ? "DOWNLOADING..." : "DOWNLOAD CLEANED CSV"}
        </button>
        {err && <span style={{ marginLeft: 12, fontSize: 11, color: "#f87171", fontFamily: mono }}>{err}</span>}
      </div>
    </div>
  );
}

function ColumnTable({ columns }: { columns: ColumnStat[] }) {
  const [sortBy, setSortBy] = useState<"name" | "missing_pct">("missing_pct");
  const sorted = [...columns].sort((a, b) =>
    sortBy === "missing_pct" ? b.missing_pct - a.missing_pct : a.name.localeCompare(b.name)
  );

  const typeColor: Record<string, string> = {
    numeric: "#00d4aa", categorical: "#f59e0b",
    datetime: "#a78bfa", text: "#3b82f6", boolean: "#22c55e", unknown: "#475569",
  };

  return (
    <div style={{
      background: "rgba(8,14,27,0.8)",
      border: "1px solid rgba(51,65,85,0.5)",
      borderRadius: 10, overflow: "hidden",
    }}>
      <div style={{
        padding: "12px 16px", display: "flex",
        justifyContent: "space-between", alignItems: "center",
        borderBottom: "1px solid rgba(30,41,59,0.8)",
      }}>
        <span style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 10, fontWeight: 700, letterSpacing: 2, color: "#475569",
        }}>
          COLUMN PROFILE ({columns.length})
        </span>
        <div style={{ display: "flex", gap: 6 }}>
          {(["missing_pct", "name"] as const).map((key) => (
            <button key={key} onClick={() => setSortBy(key)} style={{
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 9, letterSpacing: 1,
              color: sortBy === key ? "#00d4aa" : "#334155",
              background: sortBy === key ? "rgba(0,212,170,0.08)" : "transparent",
              border: `1px solid ${sortBy === key ? "rgba(0,212,170,0.3)" : "rgba(51,65,85,0.4)"}`,
              borderRadius: 3, padding: "2px 8px", cursor: "pointer",
            }}>
              {key === "missing_pct" ? "MISSING" : "A→Z"}
            </button>
          ))}
        </div>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(30,41,59,0.8)" }}>
              {["Column", "Type", "Missing %", "Unique", "Min", "Max"].map((h) => (
                <th key={h} style={{
                  padding: "8px 12px", textAlign: "left",
                  fontFamily: "JetBrains Mono, monospace",
                  fontSize: 9, color: "#334155", letterSpacing: 1, fontWeight: 600,
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 20).map((col, i) => (
              <tr key={col.name} style={{
                borderBottom: i < sorted.length - 1 ? "1px solid rgba(15,23,42,0.8)" : "none",
                transition: "background 0.15s",
              }}
                onMouseEnter={(e) => e.currentTarget.style.background = "rgba(51,65,85,0.08)"}
                onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
              >
                <td style={{
                  padding: "8px 12px",
                  fontFamily: "JetBrains Mono, monospace",
                  fontSize: 11, color: "#e2e8f0", maxWidth: 160,
                  overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>
                  {col.name}
                </td>
                <td style={{ padding: "8px 12px" }}>
                  <span style={{
                    fontFamily: "JetBrains Mono, monospace",
                    fontSize: 9, letterSpacing: 1,
                    color: typeColor[col.inferred_type] ?? "#475569",
                    padding: "1px 6px", borderRadius: 3,
                    background: `${typeColor[col.inferred_type] ?? "#475569"}14`,
                  }}>
                    {col.inferred_type.toUpperCase()}
                  </span>
                </td>
                <td style={{ padding: "8px 12px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{
                      width: 40, height: 3,
                      background: "rgba(15,23,42,0.8)", borderRadius: 2,
                    }}>
                      <div style={{
                        height: "100%",
                        width: `${Math.min(col.missing_pct, 100)}%`,
                        background: col.missing_pct > 30 ? "#f43f5e" : col.missing_pct > 10 ? "#f59e0b" : "#22c55e",
                        borderRadius: 2,
                      }} />
                    </div>
                    <span style={{
                      fontFamily: "JetBrains Mono, monospace",
                      fontSize: 10,
                      color: col.missing_pct > 30 ? "#f43f5e" : col.missing_pct > 10 ? "#f59e0b" : "#475569",
                    }}>
                      {col.missing_pct.toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td style={{
                  padding: "8px 12px",
                  fontFamily: "JetBrains Mono, monospace",
                  fontSize: 10, color: "#475569",
                }}>
                  {col.unique.toLocaleString()}
                </td>
                <td style={{
                  padding: "8px 12px",
                  fontFamily: "JetBrains Mono, monospace",
                  fontSize: 10, color: "#334155",
                }}>
                  {col.min !== undefined ? col.min.toFixed(2) : "—"}
                </td>
                <td style={{
                  padding: "8px 12px",
                  fontFamily: "JetBrains Mono, monospace",
                  fontSize: 10, color: "#334155",
                }}>
                  {col.max !== undefined ? col.max.toFixed(2) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function WarningsFeed({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return null;
  return (
    <div style={{
      background: "rgba(245,158,11,0.04)",
      border: "1px solid rgba(245,158,11,0.15)",
      borderRadius: 8, padding: 14,
    }}>
      <div style={{
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 9, fontWeight: 700, letterSpacing: 2,
        color: "#f59e0b", marginBottom: 10,
      }}>
        WARNINGS ({warnings.length})
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {warnings.map((w, i) => (
          <div key={i} style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 10, color: "#92400e",
            display: "flex", gap: 8,
          }}>
            <span style={{ color: "#f59e0b", flexShrink: 0 }}>▸</span>
            {w}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

export default function Dashboard({
  jobId, token, goal, filename, onNewAnalysis, workspace = "DIAGNOSTICS", apiUrl,
}: DashboardProps) {
  const [resultData, setResultData] = useState<JobResultData | null>(null);
  const [activeTab, setActiveTab]   = useState<string>("overview");
  const [isLoadingResult, setIsLoadingResult] = useState(false);
  const resultFetched = useRef(false);

  const API_BASE = apiUrl ?? DEFAULT_API_BASE;

  // Reset local state when jobId changes
  useEffect(() => {
    setResultData(null);
    resultFetched.current = false;
    setIsLoadingResult(false);
  }, [jobId]);

  // Reset active tab when workspace or jobId changes
  useEffect(() => {
    if (workspace === "DIAGNOSTICS") setActiveTab("overview");
    else if (workspace === "REPAIR_SHOP") setActiveTab("checklist");
    else if (workspace === "VISION_LAB") setActiveTab("gallery");
    else if (workspace === "TEXT_INTELLIGENCE") setActiveTab("analysis");
  }, [workspace, jobId]);

  const wsOptions: any = {
    jobId,
    token,
    apiUrl,
    onDone: async (r: WSResultPayload) => {
      if (resultFetched.current) return;
      resultFetched.current = true;
      setIsLoadingResult(true);
      try {
        const res = await fetch(`${API_BASE}${r.result_url}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          const data = await res.json();
          setResultData(data);
        }
      } catch { /* handled by error state */ }
      finally { setIsLoadingResult(false); }
    },
  };

  const {
    connectionState, progress, currentStep, warnings,
    result, error, isConnected, retryCount, elapsedMs,
    messages,
  } = useNydraWS(wsOptions);

  const jobProgress = progress?.progress ?? 0;
  const isDone      = connectionState === "closed" && result !== null;
  const isFailed    = (connectionState === "closed" && error !== null) || connectionState === "error";
  const isCancelled = connectionState === "closed" && result === null && error === null;
  const isRunning   = connectionState === "connected" || connectionState === "reconnecting" || connectionState === "connecting";

  const wsWarnings  = messages
    .filter((m) => m.event === "warning")
    .map((m) => (m.payload as any).message as string);

  const handleDownloadReport = async (fmt: string) => {
    if (!jobId) return;
    const res = await fetch(`${API_BASE}/api/v1/jobs/${jobId}/report`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, format: fmt }),
    });
    if (!res.ok) {
      console.error("Report generation failed:", res.status);
      return;
    }
    const data = await res.json();
    const fileRes = await fetch(`${API_BASE}${data.download_url}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!fileRes.ok) {
      console.error("Report download failed:", fileRes.status);
      return;
    }
    const blob = await fileRes.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `nydra-report-${jobId.slice(0, 8)}.${fmt}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  if (!jobId) return null;

  // Define tabs per workspace
  const workspaceTabs: Record<string, { id: string; label: string }[]> = {
    DIAGNOSTICS: [
      { id: "overview",      label: "OVERVIEW" },
      { id: "columns",       label: "COLUMNS" },
      { id: "outliers",      label: "OUTLIERS" },
      { id: "distributions", label: "DISTRIBUTIONS" },
    ],
    REPAIR_SHOP: [
      { id: "checklist", label: "FIX CHECKLIST" },
      { id: "issues",    label: `ISSUES${resultData?.issues ? ` (${resultData.issues.length})` : ""}` },
      { id: "imputation", label: "SMART IMPUTE" },
    ],
    VISION_LAB: [
      { id: "gallery",   label: "GALLERY" },
      { id: "quality",   label: "QUALITY" },
      { id: "detections", label: "DETECTIONS" },
    ],
    TEXT_INTELLIGENCE: [
      { id: "analysis",  label: "ANALYSIS" },
      { id: "entities",  label: "ENTITIES" },
      { id: "sentiment", label: "SENTIMENT" },
    ],
  };

  const tabs = workspaceTabs[workspace] || workspaceTabs.DIAGNOSTICS;

  return (
    <div style={{
      fontFamily: "JetBrains Mono, monospace",
      display: "flex", flexDirection: "column", gap: 16,
      color: "#e2e8f0",
    }}>
      {/* ── Header ────────────────────────────────────────────────── */}
      <div style={{
        display: "flex", alignItems: "flex-start",
        justifyContent: "space-between", gap: 12,
      }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <span style={{
              fontSize: 9, fontWeight: 700, letterSpacing: 2,
              color: "#00d4aa", padding: "2px 8px",
              border: "1px solid rgba(0,212,170,0.3)", borderRadius: 3,
            }}>
              {workspace.replace(/_/g, " ")} // {goal.replace(/_/g, " ").toUpperCase()}
            </span>
            {jobId && <ConnectionBadge state={connectionState} retryCount={retryCount} />}
          </div>
          <div style={{
            fontFamily: "Syne, sans-serif",
            fontSize: 16, fontWeight: 700, color: "#f1f5f9",
          }}>
            {filename}
          </div>
          <div style={{ fontSize: 10, color: "#334155", marginTop: 2 }}>
            job · {jobId.slice(0, 8)}...
          </div>
        </div>

        {/* Actions */}
        {isDone && (
          <div style={{ display: "flex", gap: 6 }}>
            {["json", "md", "pdf"].map((fmt) => (
              <button key={fmt} onClick={() => handleDownloadReport(fmt)} style={{
                padding: "6px 12px", borderRadius: 5, cursor: "pointer",
                background: "rgba(15,23,42,0.8)",
                border: "1px solid rgba(51,65,85,0.5)",
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 9, fontWeight: 700, letterSpacing: 1,
                color: "#64748b", transition: "all 0.2s",
              }}
                onMouseEnter={(e) => { e.currentTarget.style.color = "#00d4aa"; e.currentTarget.style.borderColor = "rgba(0,212,170,0.3)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = "#64748b"; e.currentTarget.style.borderColor = "rgba(51,65,85,0.5)"; }}
              >
                ↓ {fmt.toUpperCase()}
              </button>
            ))}
            {onNewAnalysis && (
              <button onClick={onNewAnalysis} style={{
                padding: "6px 14px", borderRadius: 5, cursor: "pointer",
                background: "rgba(0,212,170,0.08)",
                border: "1px solid rgba(0,212,170,0.3)",
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 9, fontWeight: 700, letterSpacing: 1,
                color: "#00d4aa",
              }}>
                + NEW
              </button>
            )}
          </div>
        )}
      </div>

      {/* ── Running state ──────────────────────────────────────────── */}
      {isRunning && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <ProgressSection
            progress={jobProgress}
            currentStep={currentStep?.step_name ?? progress?.current_step ?? ""}
            elapsedMs={elapsedMs}
            etaMs={progress?.eta_ms ?? null}
            isConnected={isConnected}
            retryCount={retryCount}
          />
          {currentStep && (
            <div style={{
              padding: 16, borderRadius: 10,
              background: "rgba(8,14,27,0.4)",
              border: "1px solid rgba(51,65,85,0.2)",
            }}>
              <StepTracker
                currentStep={currentStep.step_name}
                isDone={isDone}
                steps={Array.from({ length: currentStep.total_steps }).map((_, i) => ({
                  name: i === currentStep.step_index ? currentStep.step_name : `Step ${i + 1}`,
                  status: i < currentStep.step_index ? "done" : i === currentStep.step_index ? "active" : "pending",
                }))}
              />
            </div>
          )}
        </div>
      )}

      {/* ── Cancelled state ────────────────────────────────────────── */}
      {isCancelled && (
        <div style={{
          padding: 16, borderRadius: 8,
          background: "rgba(100,116,139,0.06)",
          border: "1px solid rgba(100,116,139,0.25)",
        }}>
          <div style={{
            fontSize: 11, fontWeight: 700, color: "#64748b",
            letterSpacing: 1, marginBottom: 6,
          }}>
            ○ JOB CANCELLED
          </div>
          <div style={{ fontSize: 11, color: "#475569" }}>
            The analysis was stopped by the user or the system.
          </div>
        </div>
      )}

      {/* ── Failed state ───────────────────────────────────────────── */}
      {isFailed && (
        <div style={{
          padding: 16, borderRadius: 8,
          background: "rgba(244,63,94,0.06)",
          border: "1px solid rgba(244,63,94,0.25)",
        }}>
          <div style={{
            fontSize: 11, fontWeight: 700, color: "#f43f5e",
            letterSpacing: 1, marginBottom: 6,
          }}>
            ⬟ JOB FAILED
          </div>
          <div style={{ fontSize: 11, color: "#9f1239" }}>
            {error?.message ?? "An unexpected error occurred"}
          </div>
        </div>
      )}

      {/* ── Warnings ───────────────────────────────────────────────── */}
      {wsWarnings.length > 0 && <WarningsFeed warnings={wsWarnings} />}

      {/* ── Results ────────────────────────────────────────────────── */}
      {isDone && (
        <>
          {/* Score summary */}
          {result && (
            <div style={{
              padding: "14px 16px", borderRadius: 8,
              background: "rgba(34,197,94,0.05)",
              border: "1px solid rgba(34,197,94,0.2)",
              display: "flex", alignItems: "center", gap: 16,
            }}>
              {result.score !== null && result.score !== undefined && (
                <div style={{ textAlign: "center", flexShrink: 0 }}>
                  <div style={{
                    fontFamily: "Syne, sans-serif",
                    fontSize: 28, fontWeight: 800,
                    color: result.score >= 80 ? "#22c55e" : result.score >= 60 ? "#f59e0b" : "#f43f5e",
                  }}>
                    {result.score}
                  </div>
                  <div style={{ fontSize: 9, color: "#475569", letterSpacing: 1 }}>SCORE</div>
                </div>
              )}
              <div>
                <div style={{
                  fontFamily: "Syne, sans-serif",
                  fontSize: 13, fontWeight: 700, color: "#22c55e", marginBottom: 4,
                }}>
                  Analysis complete
                </div>
                <div style={{ fontSize: 10, color: "#475569" }}>{result.summary}</div>
                <div style={{ fontSize: 10, color: "#334155", marginTop: 4 }}>
                  {result.total_issues} issue{result.total_issues !== 1 ? "s" : ""} found
                  · {Math.round(result.duration_ms / 1000)}s
                </div>
              </div>
            </div>
          )}

          {/* Skeleton while fetching result data */}
          {isLoadingResult && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <Skeleton h={120} />
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 8 }}>
                {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} h={60} />)}
              </div>
              <Skeleton h={200} />
            </div>
          )}

          {/* Full result data */}
          {resultData && !isLoadingResult && (
            <>
              {/* Overview stats (always visible in results) */}
              {resultData.overview && <OverviewStats overview={resultData.overview} />}

              {/* Quality gauge (always visible in results) */}
              {resultData.quality && <QualitySection quality={resultData.quality} />}

              {/* Workspace Tabs */}
              <div>
                <div style={{
                  display: "flex", gap: 2,
                  borderBottom: "1px solid rgba(30,41,59,0.8)",
                  marginBottom: 16,
                }}>
                  {tabs.map((tab) => (
                    <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
                      padding: "8px 14px", background: "none",
                      border: "none", cursor: "pointer",
                      borderBottom: activeTab === tab.id
                        ? "2px solid #00d4aa"
                        : "2px solid transparent",
                      fontFamily: "JetBrains Mono, monospace",
                      fontSize: 9, fontWeight: 700, letterSpacing: 1.5,
                      color: activeTab === tab.id ? "#00d4aa" : "#334155",
                      transition: "all 0.2s",
                      marginBottom: -1,
                    }}>
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* --- DIAGNOSTICS VIEWS --- */}
                {activeTab === "overview" && resultData.overview && <OverviewView overview={resultData.overview} />}
                {activeTab === "columns" && resultData.overview && (
                  <ColumnTable columns={resultData.overview.column_stats} />
                )}
                {activeTab === "outliers" && <OutliersView data={(resultData as any).detect_outliers} />}
                {activeTab === "distributions" && <DistributionsView jobId={jobId} token={token} apiBase={API_BASE} />}

                {/* --- REPAIR SHOP VIEWS --- */}
                {activeTab === "checklist" && resultData.quality && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {resultData.quality.fix_checklist.map((item, i) => (
                      <div key={i} style={{
                        display: "flex", gap: 10, alignItems: "flex-start",
                        padding: "10px 14px", borderRadius: 6,
                        background: "rgba(8,14,27,0.6)",
                        border: "1px solid rgba(51,65,85,0.3)",
                      }}>
                        <span style={{
                          fontFamily: "JetBrains Mono, monospace",
                          fontSize: 10, color: "#334155", flexShrink: 0,
                          marginTop: 1,
                        }}>
                          {String(i + 1).padStart(2, "0")}
                        </span>
                        <span style={{
                          fontFamily: "JetBrains Mono, monospace",
                          fontSize: 11, color: "#94a3b8",
                        }}>
                          {item}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                {activeTab === "issues" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {resultData.issues?.length ? (
                      resultData.issues
                        .sort((a, b) => {
                          const order = ["critical","high","medium","low","info"];
                          return order.indexOf(a.severity) - order.indexOf(b.severity);
                        })
                        .map((issue) => <IssueCard key={issue.issue_id} issue={issue} />)
                    ) : (
                      <div style={{
                        textAlign: "center", padding: 32,
                        fontFamily: "JetBrains Mono, monospace",
                        fontSize: 12, color: "#22c55e",
                      }}>
                        ✓ No issues found
                      </div>
                    )}
                  </div>
                )}
                {activeTab === "imputation" && <RepairResultView info={(resultData as any).cleaned_file} jobId={jobId} token={token} apiBase={API_BASE} />}

                {/* --- VISION LAB VIEWS --- */}
                {activeTab === "gallery" && (
                   <div style={{ textAlign: "center", padding: 40, border: "1px dashed rgba(51,65,85,0.5)", borderRadius: 10 }}>
                     <span style={{ fontSize: 10, color: "#475569" }}>VISION GALLERY ACTIVE. {resultData.overview?.file_type?.includes("csv") ? "(Not available for tabular data)" : "Waiting for image stream..."}</span>
                   </div>
                )}

                {/* --- TEXT INTEL VIEWS --- */}
                {activeTab === "analysis" && (
                   <div style={{ textAlign: "center", padding: 40, border: "1px dashed rgba(51,65,85,0.5)", borderRadius: 10 }}>
                     <span style={{ fontSize: 10, color: "#475569" }}>NLP INTELLIGENCE ACTIVE. {resultData.overview?.file_type?.includes("csv") ? "Select a text column to run PII and Sentiment audit." : "Analyzing document structure..."}</span>
                   </div>
                )}
              </div>
            </>
          )}
        </>
      )}

      {/* ── Idle state ─────────────────────────────────────────────── */}
      {connectionState === "idle" && (
        <div style={{
          textAlign: "center", padding: 40,
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 11, color: "#1e293b",
        }}>
          Waiting for job to start...
        </div>
      )}
    </div>
  );
}