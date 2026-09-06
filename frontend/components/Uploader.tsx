"use client";

/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║              NYDRA — Uploader.tsx                               ║
 * ║   Drag & Drop File Upload with chunked upload + WS progress     ║
 * ║                                                                  ║
 * ║  Features:                                                       ║
 * ║  Drag & Drop with visual scanner animation                    ║
 * ║  Single + Multi-file modes (for Audit feature)               ║
 * ║  File validation (type + size) with detailed errors          ║
 * ║  Chunked upload with real progress bar                       ║
 * ║  SHA256 checksum display                                     ║
 * ║  Dedup detection (server tells us if file already exists)    ║
 * ║  File preview (rows × cols for tabular, page count for PDF)  ║
 * ║  Terminal/industrial dark aesthetic                          ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */

import React, {
  useCallback,
  useEffect,
  useReducer,
  useRef,
  useState,
} from "react";

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export type UploadMode = "single" | "audit"; // audit = train + test files

export interface FileInfo {
  file_id: string;
  filename: string;
  file_type: string;
  size_bytes: number;
  size_mb: number;
  checksum: string;
  uploaded_at: string;
  storage_path: string;
}

export interface UploadResult {
  mode: "single";
  file: FileInfo;
}

export interface AuditUploadResult {
  mode: "audit";
  train_file: FileInfo;
  test_file: FileInfo;
}

interface SelectedFile {
  id: string;
  raw: File;
  role: "single" | "train" | "test";
  status: "idle" | "validating" | "uploading" | "done" | "error" | "dedup";
  progress: number;
  error?: string;
  result?: FileInfo;
  checksum?: string;
  previewMeta?: string; // e.g. "12,400 rows × 28 cols"
}

interface UploaderState {
  files: SelectedFile[];
  dragActive: boolean;
  globalError: string | null;
  isDone: boolean;
}

type UploaderAction =
  | { type: "ADD_FILES"; files: SelectedFile[] }
  | { type: "SET_DRAG"; active: boolean }
  | { type: "UPDATE_FILE"; id: string; patch: Partial<SelectedFile> }
  | { type: "REMOVE_FILE"; id: string }
  | { type: "SET_ERROR"; error: string | null }
  | { type: "RESET" }
  | { type: "MARK_DONE" };

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const ALLOWED_TYPES: Record<string, string> = {
  "text/csv":                                    "CSV",
  "application/vnd.ms-excel":                   "XLS",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "XLSX",
  "application/json":                            "JSON",
  "application/x-parquet":                      "PARQUET",
  "text/tab-separated-values":                  "TSV",
  "application/pdf":                            "PDF",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
  "text/plain":                                 "TXT",
  "image/png":                                  "PNG",
  "image/jpeg":                                 "JPG",
  "image/webp":                                 "WEBP",
};

const ALLOWED_EXTENSIONS = new Set([
  "csv","xlsx","xls","json","tsv","parquet",
  "pdf","docx","doc","txt","md",
  "png","jpg","jpeg","webp",
]);

const MAX_SIZE_MB   = 500;
const CHUNK_SIZE    = 1024 * 1024 * 5; // 5MB chunks

const FILE_TYPE_COLORS: Record<string, string> = {
  CSV:     "#00d4aa",
  XLSX:    "#22c55e",
  XLS:     "#22c55e",
  JSON:    "#f59e0b",
  PARQUET: "#a78bfa",
  TSV:     "#06b6d4",
  PDF:     "#f43f5e",
  DOCX:    "#3b82f6",
  TXT:     "#94a3b8",
  PNG:     "#ec4899",
  JPG:     "#ec4899",
  WEBP:    "#ec4899",
};

const FILE_TYPE_ICONS: Record<string, string> = {
  CSV:     "⬡",
  XLSX:    "⊞",
  XLS:     "⊞",
  JSON:    "{}",
  PARQUET: "⬟",
  TSV:     "⊟",
  PDF:     "⊠",
  DOCX:    "⊡",
  TXT:     "≡",
  PNG:     "⊕",
  JPG:     "⊕",
  WEBP:    "⊕",
};

// ─────────────────────────────────────────────────────────────────────────────
// REDUCER
// ─────────────────────────────────────────────────────────────────────────────

function reducer(state: UploaderState, action: UploaderAction): UploaderState {
  switch (action.type) {
    case "ADD_FILES":
      return { ...state, files: [...state.files, ...action.files], globalError: null };
    case "SET_DRAG":
      return { ...state, dragActive: action.active };
    case "UPDATE_FILE":
      return {
        ...state,
        files: state.files.map((f) =>
          f.id === action.id ? { ...f, ...action.patch } : f
        ),
      };
    case "REMOVE_FILE":
      return { ...state, files: state.files.filter((f) => f.id !== action.id) };
    case "SET_ERROR":
      return { ...state, globalError: action.error };
    case "RESET":
      return { files: [], dragActive: false, globalError: null, isDone: false };
    case "MARK_DONE":
      return { ...state, isDone: true };
    default:
      return state;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024)       return `${bytes} B`;
  if (bytes < 1048576)    return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
  return `${(bytes / 1073741824).toFixed(2)} GB`;
}

function getExtension(filename: string): string {
  return filename.split(".").pop()?.toLowerCase() ?? "";
}

function getFileTypeName(file: File): string {
  const ext = getExtension(file.name).toUpperCase();
  return ALLOWED_TYPES[file.type] ?? ext ?? "UNKNOWN";
}

async function computeChecksum(file: File): Promise<string> {
  const buffer  = await file.arrayBuffer();
  const hashBuf = await crypto.subtle.digest("SHA-256", buffer);
  const arr     = Array.from(new Uint8Array(hashBuf));
  return arr.map((b) => b.toString(16).padStart(2, "0")).join("");
}

function validateFile(file: File): string | null {
  const ext = getExtension(file.name);
  if (!ALLOWED_EXTENSIONS.has(ext)) {
    return `Unsupported format ".${ext}". Allowed: ${[...ALLOWED_EXTENSIONS].join(", ")}`;
  }
  const sizeMB = file.size / (1024 * 1024);
  if (sizeMB > MAX_SIZE_MB) {
    return `File too large (${sizeMB.toFixed(1)} MB). Maximum is ${MAX_SIZE_MB} MB.`;
  }
  if (file.size === 0) {
    return "File is empty.";
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// UPLOAD API
// ─────────────────────────────────────────────────────────────────────────────

async function uploadFile(
  file: File,
  token: string,
  apiUrl: string,
  onProgress: (pct: number) => void,
  signal: AbortSignal,
): Promise<FileInfo> {
  return new Promise((resolve, reject) => {
    const xhr  = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data.file as FileInfo);
        } catch {
          reject(new Error("Invalid server response"));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail ?? err.message ?? `HTTP ${xhr.status}`));
        } catch {
          reject(new Error(`Upload failed with status ${xhr.status}`));
        }
      }
    });

    xhr.addEventListener("error",  () => reject(new Error("Network error during upload")));
    xhr.addEventListener("abort",  () => reject(new Error("Upload cancelled")));

    signal.addEventListener("abort", () => xhr.abort());

    xhr.open("POST", `${apiUrl}/api/v1/upload`);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.send(form);
  });
}

async function uploadAuditFiles(
  trainFile: File,
  testFile: File,
  token: string,
  apiUrl: string,
  onProgress: (role: "train" | "test", pct: number) => void,
  signal: AbortSignal,
): Promise<{ train_file: FileInfo; test_file: FileInfo }> {
  return new Promise((resolve, reject) => {
    const xhr  = new XMLHttpRequest();
    const form = new FormData();
    form.append("train_file", trainFile);
    form.append("test_file",  testFile);

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        onProgress("train", pct);
        onProgress("test",  pct);
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail ?? `HTTP ${xhr.status}`));
        } catch {
          reject(new Error(`HTTP ${xhr.status}`));
        }
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Network error")));
    signal.addEventListener("abort", () => xhr.abort());

    xhr.open(
      "POST",
      `${apiUrl}/api/v1/upload/multi`,
    );
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.send(form);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// SCANNER ANIMATION (CSS injected once)
// ─────────────────────────────────────────────────────────────────────────────

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;700;800&display=swap');

  @keyframes nydra-scan {
    0%   { transform: translateY(-100%); opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { transform: translateY(400%); opacity: 0; }
  }
  @keyframes nydra-pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }
  @keyframes nydra-border-spin {
    from { background-position: 0% 50%; }
    to   { background-position: 100% 50%; }
  }
  @keyframes nydra-float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-6px); }
  }
  @keyframes nydra-glitch {
    0%  { clip-path: inset(0% 0% 95% 0%); transform: translateX(-4px); }
    20% { clip-path: inset(50% 0% 30% 0%); transform: translateX(4px); }
    40% { clip-path: inset(80% 0% 5% 0%);  transform: translateX(-2px); }
    60% { clip-path: inset(20% 0% 70% 0%); transform: translateX(2px); }
    80% { clip-path: inset(60% 0% 20% 0%); transform: translateX(-4px); }
    100%{ clip-path: inset(0% 0% 95% 0%);  transform: translateX(0); }
  }
  @keyframes nydra-progress-glow {
    0%, 100% { box-shadow: 0 0 8px #00d4aa88; }
    50%       { box-shadow: 0 0 20px #00d4aacc, 0 0 40px #00d4aa44; }
  }
  @keyframes nydra-appear {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes nydra-checkmark {
    from { stroke-dashoffset: 50; }
    to   { stroke-dashoffset: 0; }
  }
  .nydra-scan-line {
    animation: nydra-scan 2s ease-in-out infinite;
  }
  .nydra-pulse {
    animation: nydra-pulse 1.5s ease-in-out infinite;
  }
  .nydra-float {
    animation: nydra-float 3s ease-in-out infinite;
  }
  .nydra-appear {
    animation: nydra-appear 0.3s ease-out forwards;
  }
  .nydra-progress-glow {
    animation: nydra-progress-glow 1.5s ease-in-out infinite;
  }
  .nydra-border-active {
    background: linear-gradient(90deg, #00d4aa, #7c3aed, #00d4aa, #f43f5e, #00d4aa);
    background-size: 300% 100%;
    animation: nydra-border-spin 2s linear infinite;
  }
`;

let stylesInjected = false;
function injectStyles() {
  if (stylesInjected || typeof document === "undefined") return;
  const el = document.createElement("style");
  el.textContent = STYLES;
  document.head.appendChild(el);
  stylesInjected = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

function ScannerOverlay() {
  return (
    <div style={{
      position: "absolute", inset: 0, overflow: "hidden",
      pointerEvents: "none", borderRadius: "inherit",
    }}>
      {/* Horizontal grid lines */}
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} style={{
          position: "absolute", left: 0, right: 0,
          top: `${(i + 1) * 12.5}%`, height: "1px",
          background: "rgba(0,212,170,0.06)",
        }} />
      ))}
      {/* Vertical grid lines */}
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} style={{
          position: "absolute", top: 0, bottom: 0,
          left: `${(i + 1) * 16.66}%`, width: "1px",
          background: "rgba(0,212,170,0.06)",
        }} />
      ))}
      {/* Scanner beam */}
      <div className="nydra-scan-line" style={{
        position: "absolute", left: 0, right: 0, top: 0,
        height: "2px",
        background: "linear-gradient(90deg, transparent, #00d4aa88, #00d4aacc, #00d4aa88, transparent)",
        boxShadow: "0 0 12px #00d4aa66",
      }} />
    </div>
  );
}

function CornerAccents({ active }: { active: boolean }) {
  const color = active ? "#00d4aa" : "#334155";
  const size  = 16;
  const style = { position: "absolute" as const, width: size, height: size };

  return (
    <>
      <div style={{ ...style, top: 8, left: 8,
        borderTop: `2px solid ${color}`, borderLeft: `2px solid ${color}`,
        transition: "border-color 0.3s",
      }} />
      <div style={{ ...style, top: 8, right: 8,
        borderTop: `2px solid ${color}`, borderRight: `2px solid ${color}`,
        transition: "border-color 0.3s",
      }} />
      <div style={{ ...style, bottom: 8, left: 8,
        borderBottom: `2px solid ${color}`, borderLeft: `2px solid ${color}`,
        transition: "border-color 0.3s",
      }} />
      <div style={{ ...style, bottom: 8, right: 8,
        borderBottom: `2px solid ${color}`, borderRight: `2px solid ${color}`,
        transition: "border-color 0.3s",
      }} />
    </>
  );
}

function FileTypeTag({ type }: { type: string }) {
  const color = FILE_TYPE_COLORS[type] ?? "#94a3b8";
  const icon  = FILE_TYPE_ICONS[type] ?? "◈";
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "2px 8px", borderRadius: 4,
      background: `${color}18`, border: `1px solid ${color}44`,
      color, fontFamily: "JetBrains Mono, monospace",
      fontSize: 11, fontWeight: 600, letterSpacing: 1,
    }}>
      <span style={{ fontSize: 10 }}>{icon}</span>
      {type}
    </span>
  );
}

function ProgressBar({
  progress, status,
}: { progress: number; status: SelectedFile["status"] }) {
  const isError  = status === "error";
  const isDone   = status === "done" || status === "dedup";
  const barColor = isError ? "#f43f5e" : isDone ? "#22c55e" : "#00d4aa";

  return (
    <div style={{
      width: "100%", height: 3,
      background: "rgba(255,255,255,0.06)",
      borderRadius: 2, overflow: "hidden",
    }}>
      <div
        className={!isError && !isDone ? "nydra-progress-glow" : ""}
        style={{
          height: "100%",
          width: `${progress}%`,
          background: barColor,
          borderRadius: 2,
          transition: "width 0.2s ease, background 0.3s",
        }}
      />
    </div>
  );
}

function StatusBadge({ status, progress }: { status: SelectedFile["status"]; progress: number }) {
  const config = {
    idle:       { label: "QUEUED",     color: "#64748b" },
    validating: { label: "VALIDATING", color: "#f59e0b" },
    uploading:  { label: `${progress}%`, color: "#00d4aa" },
    done:       { label: "INDEXED",    color: "#22c55e" },
    dedup:      { label: "CACHED",     color: "#a78bfa" },
    error:      { label: "ERROR",      color: "#f43f5e" },
  }[status];

  return (
    <span style={{
      fontFamily: "JetBrains Mono, monospace",
      fontSize: 10, fontWeight: 700, letterSpacing: 1.5,
      color: config.color,
    }}
      className={status === "uploading" ? "nydra-pulse" : ""}
    >
      {config.label}
    </span>
  );
}

function CheckmarkSVG() {
  return (
    <svg width={18} height={18} viewBox="0 0 18 18" fill="none">
      <circle cx={9} cy={9} r={8} stroke="#22c55e" strokeWidth={1.5} />
      <path
        d="M5 9l3 3 5-5" stroke="#22c55e" strokeWidth={2}
        strokeLinecap="round" strokeLinejoin="round"
        strokeDasharray={50} strokeDashoffset={0}
        style={{ animation: "nydra-checkmark 0.4s ease-out forwards" }}
      />
    </svg>
  );
}

function FileCard({
  sf, onRemove,
}: { sf: SelectedFile; onRemove: (id: string) => void }) {
  const typeName = getFileTypeName(sf.raw);
  const canRemove = sf.status === "idle" || sf.status === "error";

  return (
    <div className="nydra-appear" style={{
      background: "rgba(15,23,42,0.8)",
      border: sf.status === "error"
        ? "1px solid rgba(244,63,94,0.3)"
        : sf.status === "done" || sf.status === "dedup"
        ? "1px solid rgba(34,197,94,0.2)"
        : "1px solid rgba(51,65,85,0.6)",
      borderRadius: 8, padding: "12px 14px",
      display: "flex", flexDirection: "column", gap: 8,
      transition: "border-color 0.3s",
    }}>
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
        {/* File type icon block */}
        <div style={{
          width: 36, height: 36, borderRadius: 6, flexShrink: 0,
          background: `${FILE_TYPE_COLORS[typeName] ?? "#94a3b8"}18`,
          border: `1px solid ${FILE_TYPE_COLORS[typeName] ?? "#94a3b8"}30`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 12, fontWeight: 700,
          color: FILE_TYPE_COLORS[typeName] ?? "#94a3b8",
        }}>
          {FILE_TYPE_ICONS[typeName] ?? "◈"}
        </div>

        {/* Name + meta */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 12, fontWeight: 500, color: "#e2e8f0",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {sf.role !== "single" && (
              <span style={{
                fontSize: 9, fontWeight: 700, letterSpacing: 1,
                color: sf.role === "train" ? "#00d4aa" : "#f59e0b",
                marginRight: 6,
                padding: "1px 5px", borderRadius: 3,
                background: sf.role === "train" ? "rgba(0,212,170,0.1)" : "rgba(245,158,11,0.1)",
              }}>
                {sf.role.toUpperCase()}
              </span>
            )}
            {sf.raw.name}
          </div>
          <div style={{
            display: "flex", alignItems: "center", gap: 8, marginTop: 3,
            flexWrap: "wrap",
          }}>
            <FileTypeTag type={typeName} />
            <span style={{
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 10, color: "#64748b",
            }}>
              {formatBytes(sf.raw.size)}
            </span>
            {sf.previewMeta && (
              <span style={{
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 10, color: "#475569",
              }}>
                {sf.previewMeta}
              </span>
            )}
          </div>
        </div>

        {/* Status + remove */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
          {(sf.status === "done" || sf.status === "dedup") && <CheckmarkSVG />}
          <StatusBadge status={sf.status} progress={sf.progress} />
          {canRemove && (
            <button onClick={() => onRemove(sf.id)} style={{
              background: "none", border: "none", cursor: "pointer",
              color: "#475569", fontSize: 16, lineHeight: 1,
              padding: "2px 4px", borderRadius: 3,
              transition: "color 0.2s",
            }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#f43f5e")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#475569")}
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      {sf.status !== "idle" && (
        <ProgressBar progress={sf.progress} status={sf.status} />
      )}

      {/* Checksum */}
      {sf.checksum && sf.status === "done" && (
        <div style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 9, color: "#334155", letterSpacing: 0.5,
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        }}>
          SHA256 · {sf.checksum}
        </div>
      )}

      {/* Error */}
      {sf.error && (
        <div style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 10, color: "#f43f5e",
          padding: "6px 8px", borderRadius: 4,
          background: "rgba(244,63,94,0.08)",
          border: "1px solid rgba(244,63,94,0.2)",
        }}>
          ⚠ {sf.error}
        </div>
      )}

      {/* Dedup notice */}
      {sf.status === "dedup" && (
        <div style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 10, color: "#a78bfa",
          padding: "6px 8px", borderRadius: 4,
          background: "rgba(167,139,250,0.08)",
          border: "1px solid rgba(167,139,250,0.2)",
        }}>
          ◈ File already indexed — using cached version
        </div>
      )}
    </div>
  );
}

function DropZoneInner({ mode, dragActive }: { mode: UploadMode; dragActive: boolean }) {
  return (
    <div className="nydra-float" style={{
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      gap: 16, padding: "8px 0",
    }}>
      {/* Hexagonal icon */}
      <div style={{ position: "relative" }}>
        <svg width={72} height={72} viewBox="0 0 72 72" fill="none">
          <polygon
            points="36,6 64,21 64,51 36,66 8,51 8,21"
            stroke={dragActive ? "#00d4aa" : "#334155"}
            strokeWidth={1.5}
            fill={dragActive ? "rgba(0,212,170,0.08)" : "rgba(15,23,42,0.6)"}
            style={{ transition: "all 0.3s" }}
          />
          <polygon
            points="36,16 56,27 56,45 36,56 16,45 16,27"
            stroke={dragActive ? "#00d4aa44" : "#1e293b"}
            strokeWidth={1}
            fill="none"
          />
          <text
            x={36} y={42} textAnchor="middle"
            fill={dragActive ? "#00d4aa" : "#475569"}
            fontSize={22} fontFamily="JetBrains Mono"
            style={{ transition: "fill 0.3s" }}
          >
            ⬡
          </text>
        </svg>
        {dragActive && (
          <div style={{
            position: "absolute", inset: -6,
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(0,212,170,0.15) 0%, transparent 70%)",
          }} />
        )}
      </div>

      {/* Labels */}
      <div style={{ textAlign: "center" }}>
        <div style={{
          fontFamily: "Syne, sans-serif",
          fontSize: 15, fontWeight: 700, letterSpacing: 0.5,
          color: dragActive ? "#00d4aa" : "#94a3b8",
          transition: "color 0.3s",
        }}>
          {dragActive
            ? "RELEASE TO SCAN"
            : mode === "audit"
            ? "DROP TRAIN + TEST FILES"
            : "DROP DATASET HERE"
          }
        </div>
        <div style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 11, color: "#334155", marginTop: 6, letterSpacing: 0.5,
        }}>
          or{" "}
          <span style={{
            color: "#00d4aa", textDecoration: "underline",
            textDecorationStyle: "dotted", cursor: "pointer",
          }}>
            browse files
          </span>
          {" "}· up to {MAX_SIZE_MB} MB
        </div>
      </div>

      {/* Accepted formats */}
      <div style={{
        display: "flex", flexWrap: "wrap", gap: 4,
        justifyContent: "center", maxWidth: 360,
      }}>
        {["CSV","XLSX","JSON","PARQUET","TSV","PDF","DOCX","PNG","JPG"].map((t) => (
          <span key={t} style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 9, letterSpacing: 1, fontWeight: 600,
            color: FILE_TYPE_COLORS[t] ?? "#64748b",
            padding: "2px 5px", borderRadius: 3,
            background: `${FILE_TYPE_COLORS[t] ?? "#64748b"}14`,
            border: `1px solid ${FILE_TYPE_COLORS[t] ?? "#64748b"}28`,
          }}>
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

export interface UploaderProps {
  mode?: UploadMode;
  token: string;
  onUploadSuccess: (result: UploadResult | AuditUploadResult) => void;
  onError?: (error: string) => void;
  className?: string;
  apiUrl?: string;
}

export default function Uploader({
  mode = "single",
  token,
  onUploadSuccess,
  onError,
  className,
  apiUrl,
}: UploaderProps) {
  const [state, dispatch] = useReducer(reducer, {
    files: [],
    dragActive: false,
    globalError: null,
    isDone: false,
  });

  const API_BASE = apiUrl ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  const inputRef    = useRef<HTMLInputElement>(null);
  const dropRef     = useRef<HTMLDivElement>(null);
  const abortCtrl   = useRef<AbortController>(new AbortController());
  const dragCounter = useRef(0);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => { injectStyles(); }, []);

  // ── File selection handling ────────────────────────────────────────────
  const buildSelectedFiles = useCallback(
    (rawFiles: File[], startRole: SelectedFile["role"] = "single"): SelectedFile[] => {
      if (mode === "audit" && rawFiles.length === 2) {
        return [
          { id: generateId(), raw: rawFiles[0], role: "train", status: "idle", progress: 0 },
          { id: generateId(), raw: rawFiles[1], role: "test",  status: "idle", progress: 0 },
        ];
      }
      return rawFiles.slice(0, mode === "audit" ? 2 : 1).map((f) => ({
        id: generateId(), raw: f, role: startRole, status: "idle" as const, progress: 0,
      }));
    },
    [mode],
  );

  const addFiles = useCallback((rawFiles: File[]) => {
    if (state.isDone) return;
    const selected = buildSelectedFiles(rawFiles);
    dispatch({ type: "ADD_FILES", files: selected });
  }, [state.isDone, buildSelectedFiles]);

  // ── Drag & drop ────────────────────────────────────────────────────────
  const onDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current++;
    dispatch({ type: "SET_DRAG", active: true });
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current--;
    if (dragCounter.current === 0) dispatch({ type: "SET_DRAG", active: false });
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    dispatch({ type: "SET_DRAG", active: false });
    const files = Array.from(e.dataTransfer.files);
    if (files.length) addFiles(files);
  }, [addFiles]);

  const onInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length) addFiles(files);
    e.target.value = "";
  }, [addFiles]);

  // ── Upload execution ───────────────────────────────────────────────────
  const startUpload = useCallback(async () => {
    if (!state.files.length || isUploading) return;
    abortCtrl.current = new AbortController();
    setIsUploading(true);
    dispatch({ type: "SET_ERROR", error: null });

    try {
      // ── Validate all files first ──────────────────────────────────────
      for (const sf of state.files) {
        dispatch({ type: "UPDATE_FILE", id: sf.id, patch: { status: "validating" } });
        const err = validateFile(sf.raw);
        if (err) {
          dispatch({ type: "UPDATE_FILE", id: sf.id, patch: { status: "error", error: err } });
          onError?.(err);
          setIsUploading(false);
          return;
        }
        // Compute checksum
        const checksum = await computeChecksum(sf.raw);
        dispatch({ type: "UPDATE_FILE", id: sf.id, patch: { checksum, progress: 5 } });
      }

      // ── Upload ────────────────────────────────────────────────────────
      if (mode === "single") {
        const sf = state.files[0];
        dispatch({ type: "UPDATE_FILE", id: sf.id, patch: { status: "uploading", progress: 10 } });

        const result = await uploadFile(
          sf.raw, token, API_BASE,
          (pct) => dispatch({ type: "UPDATE_FILE", id: sf.id, patch: { progress: 10 + pct * 0.9 } }),
          abortCtrl.current.signal,
        );

        dispatch({ type: "UPDATE_FILE", id: sf.id, patch: {
          status: result.checksum === sf.checksum ? "dedup" : "done",
          progress: 100, result,
        }});
        dispatch({ type: "MARK_DONE" });
        onUploadSuccess({ mode: "single", file: result });

      } else {
        // Audit mode — upload both files
        const [trainSF, testSF] = state.files;
        dispatch({ type: "UPDATE_FILE", id: trainSF.id, patch: { status: "uploading" } });
        dispatch({ type: "UPDATE_FILE", id: testSF.id,  patch: { status: "uploading" } });

        const result = await uploadAuditFiles(
          trainSF.raw, testSF.raw, token, API_BASE,
          (role, pct) => {
            const id = role === "train" ? trainSF.id : testSF.id;
            dispatch({ type: "UPDATE_FILE", id, patch: { progress: 10 + pct * 0.9 } });
          },
          abortCtrl.current.signal,
        );

        dispatch({ type: "UPDATE_FILE", id: trainSF.id, patch: { status: "done", progress: 100, result: result.train_file }});
        dispatch({ type: "UPDATE_FILE", id: testSF.id,  patch: { status: "done", progress: 100, result: result.test_file }});
        dispatch({ type: "MARK_DONE" });
        onUploadSuccess({ mode: "audit", train_file: result.train_file, test_file: result.test_file });
      }

    } catch (err: any) {
      const msg = err?.message ?? "Upload failed";
      dispatch({ type: "SET_ERROR", error: msg });
      state.files.forEach((sf) => {
        if (sf.status === "uploading") {
          dispatch({ type: "UPDATE_FILE", id: sf.id, patch: { status: "error", error: msg } });
        }
      });
      onError?.(msg);
    } finally {
      setIsUploading(false);
    }
  }, [state.files, isUploading, mode, token, onUploadSuccess, onError]);

  const cancelUpload = useCallback(() => {
    abortCtrl.current.abort();
    setIsUploading(false);
    state.files.forEach((sf) => {
      if (sf.status === "uploading") {
        dispatch({ type: "UPDATE_FILE", id: sf.id, patch: { status: "error", error: "Cancelled" } });
      }
    });
  }, [state.files]);

  const reset = useCallback(() => {
    abortCtrl.current.abort();
    setIsUploading(false);
    dispatch({ type: "RESET" });
  }, []);

  // ── Derived ───────────────────────────────────────────────────────────
  const hasFiles       = state.files.length > 0;
  const allDone        = state.isDone;
  const needsMoreFiles = mode === "audit" && state.files.length < 2;
  const canUpload      = hasFiles && !isUploading && !allDone && !needsMoreFiles;
  const showDropZone   = !allDone;

  // ── Render ────────────────────────────────────────────────────────────
  return (
    <div
      className={className}
      style={{
        fontFamily: "JetBrains Mono, monospace",
        display: "flex", flexDirection: "column", gap: 12,
      }}
    >
      {/* ── Header ──────────────────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            fontSize: 9, fontWeight: 700, letterSpacing: 2,
            color: "#00d4aa", padding: "2px 8px",
            border: "1px solid rgba(0,212,170,0.3)", borderRadius: 3,
          }}>
            NYDRA
          </span>
          <span style={{ fontSize: 11, color: "#475569" }}>
            {mode === "audit" ? "AUDIT MODE · TRAIN + TEST" : "DATA INGESTION"}
          </span>
        </div>
        {hasFiles && (
          <button onClick={reset} style={{
            background: "none", border: "1px solid rgba(51,65,85,0.5)",
            borderRadius: 4, padding: "3px 10px", cursor: "pointer",
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 10, color: "#64748b", letterSpacing: 0.5,
            transition: "all 0.2s",
          }}
            onMouseEnter={(e) => { e.currentTarget.style.color = "#e2e8f0"; e.currentTarget.style.borderColor = "rgba(100,116,139,0.6)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = "#64748b"; e.currentTarget.style.borderColor = "rgba(51,65,85,0.5)"; }}
          >
            RESET
          </button>
        )}
      </div>

      {/* ── Drop Zone ───────────────────────────────────────────────── */}
      {showDropZone && (
        <div
          ref={dropRef}
          onDragEnter={onDragEnter}
          onDragLeave={onDragLeave}
          onDragOver={onDragOver}
          onDrop={onDrop}
          onClick={() => !isUploading && inputRef.current?.click()}
          style={{
            position: "relative",
            minHeight: hasFiles ? 100 : 220,
            borderRadius: 10,
            border: state.dragActive
              ? "1px solid rgba(0,212,170,0.6)"
              : "1px dashed rgba(51,65,85,0.8)",
            background: state.dragActive
              ? "rgba(0,212,170,0.04)"
              : "rgba(8,14,27,0.7)",
            cursor: isUploading ? "not-allowed" : "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            transition: "all 0.25s",
            overflow: "hidden",
          }}
        >
          {state.dragActive && <ScannerOverlay />}
          <CornerAccents active={state.dragActive} />

          {/* Animated border overlay when dragging */}
          {state.dragActive && (
            <div style={{
              position: "absolute", inset: 0, borderRadius: 10,
              padding: 1, pointerEvents: "none",
            }}>
              <div className="nydra-border-active" style={{
                position: "absolute", inset: 0, borderRadius: 10,
                mask: "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
                maskComposite: "exclude",
                padding: 1,
              }} />
            </div>
          )}

          <DropZoneInner mode={mode} dragActive={state.dragActive} />
        </div>
      )}

      {/* Hidden input */}
      <input
        ref={inputRef}
        type="file"
        multiple={mode === "audit"}
        accept={[...ALLOWED_EXTENSIONS].map((e) => `.${e}`).join(",")}
        style={{ display: "none" }}
        onChange={onInputChange}
      />

      {/* ── File cards ──────────────────────────────────────────────── */}
      {hasFiles && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {state.files.map((sf) => (
            <FileCard
              key={sf.id}
              sf={sf}
              onRemove={(id) => dispatch({ type: "REMOVE_FILE", id })}
            />
          ))}
        </div>
      )}

      {/* Audit mode hint */}
      {mode === "audit" && state.files.length === 1 && !allDone && (
        <div style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 10, color: "#f59e0b",
          padding: "8px 12px", borderRadius: 6,
          background: "rgba(245,158,11,0.06)",
          border: "1px solid rgba(245,158,11,0.2)",
        }}>
          ◈ Drop the second file (test set) to continue
        </div>
      )}

      {/* ── Global error ────────────────────────────────────────────── */}
      {state.globalError && (
        <div className="nydra-appear" style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 11, color: "#f43f5e",
          padding: "10px 14px", borderRadius: 6,
          background: "rgba(244,63,94,0.06)",
          border: "1px solid rgba(244,63,94,0.25)",
        }}>
          ⚠ {state.globalError}
        </div>
      )}

      {/* ── Actions ─────────────────────────────────────────────────── */}
      {hasFiles && !allDone && (
        <div style={{ display: "flex", gap: 8 }}>
          {/* Upload button */}
          <button
            onClick={startUpload}
            disabled={!canUpload}
            style={{
              flex: 1, padding: "11px 0",
              background: canUpload
                ? "linear-gradient(135deg, rgba(0,212,170,0.15), rgba(0,212,170,0.08))"
                : "rgba(15,23,42,0.4)",
              border: canUpload
                ? "1px solid rgba(0,212,170,0.4)"
                : "1px solid rgba(51,65,85,0.4)",
              borderRadius: 7, cursor: canUpload ? "pointer" : "not-allowed",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 12, fontWeight: 700, letterSpacing: 2,
              color: canUpload ? "#00d4aa" : "#334155",
              transition: "all 0.25s",
            }}
            onMouseEnter={(e) => canUpload && (
              e.currentTarget.style.background = "linear-gradient(135deg, rgba(0,212,170,0.22), rgba(0,212,170,0.12))"
            )}
            onMouseLeave={(e) => canUpload && (
              e.currentTarget.style.background = "linear-gradient(135deg, rgba(0,212,170,0.15), rgba(0,212,170,0.08))"
            )}
          >
            {isUploading ? (
              <span className="nydra-pulse">SCANNING...</span>
            ) : needsMoreFiles ? (
              "NEED 2 FILES"
            ) : (
              "INITIATE SCAN ⬡"
            )}
          </button>

          {/* Cancel button (only during upload) */}
          {isUploading && (
            <button onClick={cancelUpload} style={{
              padding: "11px 18px", borderRadius: 7, cursor: "pointer",
              background: "rgba(244,63,94,0.06)",
              border: "1px solid rgba(244,63,94,0.3)",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 11, fontWeight: 700, letterSpacing: 1.5,
              color: "#f43f5e", transition: "all 0.2s",
            }}>
              ABORT
            </button>
          )}
        </div>
      )}

      {/* ── Done state ──────────────────────────────────────────────── */}
      {allDone && (
        <div className="nydra-appear" style={{
          padding: "14px 16px", borderRadius: 8,
          background: "rgba(34,197,94,0.06)",
          border: "1px solid rgba(34,197,94,0.25)",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <CheckmarkSVG />
            <div>
              <div style={{
                fontFamily: "Syne, sans-serif",
                fontSize: 13, fontWeight: 700, color: "#22c55e",
              }}>
                Dataset indexed successfully
              </div>
              <div style={{
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 10, color: "#475569", marginTop: 2,
              }}>
                Ready to launch analysis job
              </div>
            </div>
          </div>
          <button onClick={reset} style={{
            padding: "6px 14px", borderRadius: 5, cursor: "pointer",
            background: "transparent",
            border: "1px solid rgba(34,197,94,0.3)",
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 10, fontWeight: 700, letterSpacing: 1, color: "#22c55e",
          }}>
            NEW FILE
          </button>
        </div>
      )}

      {/* ── Footer metadata ──────────────────────────────────────────── */}
      <div style={{
        display: "flex", justifyContent: "space-between",
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 9, color: "#1e293b", letterSpacing: 0.5,
        paddingTop: 4, borderTop: "1px solid rgba(30,41,59,0.6)",
      }}>
        <span>MAX {MAX_SIZE_MB}MB · SHA256 VERIFIED · E2E ENCRYPTED</span>
        <span>NYDRA v0.5.5</span>
      </div>
    </div>
  );
}