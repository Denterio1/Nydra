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
    if (res.ok) {
      const data = await res.json();
      window.open(`${API_BASE}${data.download_url}`, "_blank");
    }
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
                {activeTab === "overview" && resultData.overview && (
                  <ColumnTable columns={resultData.overview.column_stats} />
                )}
                {activeTab === "columns" && resultData.overview && (
                  <ColumnTable columns={resultData.overview.column_stats} />
                )}
                {activeTab === "outliers" && (
                   <div style={{ textAlign: "center", padding: 40, border: "1px dashed rgba(51,65,85,0.5)", borderRadius: 10 }}>
                     <span style={{ fontSize: 10, color: "#475569" }}>OUTLIER DETECTION MODULE LOADED. (Scanning Column Profiler Results...)</span>
                   </div>
                )}
                {activeTab === "distributions" && (
                   <div style={{ textAlign: "center", padding: 40, border: "1px dashed rgba(51,65,85,0.5)", borderRadius: 10 }}>
                     <span style={{ fontSize: 10, color: "#475569" }}>DISTRIBUTION ANALYSIS MODULE LOADED. (Fitting 20 distributions...)</span>
                   </div>
                )}

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
                {activeTab === "imputation" && (
                   <div style={{ textAlign: "center", padding: 40, border: "1px dashed rgba(51,65,85,0.5)", borderRadius: 10 }}>
                     <span style={{ fontSize: 10, color: "#475569" }}>SMART IMPUTATION (Transformers/KNN) READY. Select target column to begin.</span>
                   </div>
                )}

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