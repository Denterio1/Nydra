"use client";

/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║           NYDRA — ChatBox.tsx                                   ║
 * ║   AI Chat Interface — Context-aware, streaming, terminal UI     ║
 * ║                                                                  ║
 * ║  Streaming responses via WebSocket                           ║
 * ║  Job context injection (grounded in analysis results)        ║
 * ║  Message history with timestamps                             ║
 * ║  Suggested prompts based on goal                             ║
 * ║  Copy message to clipboard                                   ║
 * ║  Auto-scroll with manual override                            ║
 * ║  Markdown-lite rendering (bold, code, lists)                 ║
 * ║  Token counter                                               ║
 * ║  Terminal aesthetic with typing cursor                       ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */

import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

type MessageRole = "user" | "assistant" | "system";

interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  tokensUsed?: number;
  usedContext?: boolean;
  error?: boolean;
}

interface SuggestedPrompt {
  label: string;
  prompt: string;
  icon: string;
}

export interface ChatBoxProps {
  token: string;
  jobId?: string | null;
  goal?: string;
  filename?: string;
  isMinimized?: boolean;
  onToggleMinimize?: () => void;
  className?: string;
  apiUrl?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const MAX_TOKENS    = 4000;
const MAX_HISTORY   = 50;

const SUGGESTED_PROMPTS: Record<string, SuggestedPrompt[]> = {
  inspect: [
    { icon: "⬡", label: "Missing values",    prompt: "Which columns have the most missing values and what should I do?" },
    { icon: "◈", label: "Data types",        prompt: "Are there any columns with incorrect data types?" },
    { icon: "⬟", label: "Duplicates",        prompt: "How severe is the duplicate problem in this dataset?" },
  ],
  clean: [
    { icon: "⬡", label: "Best imputation",   prompt: "What is the best imputation strategy for this dataset?" },
    { icon: "◈", label: "Outlier handling",  prompt: "Should I remove or cap the outliers found?" },
    { icon: "⬟", label: "After cleaning",    prompt: "What should I do after cleaning is complete?" },
  ],
  run_automl: [
    { icon: "⬡", label: "Best model",        prompt: "Why did the best model outperform the others?" },
    { icon: "◈", label: "Overfitting",       prompt: "Is there a risk of overfitting with the top model?" },
    { icon: "⬟", label: "Feature importance",prompt: "Which features are most important and can I remove any?" },
  ],
  full_pipeline: [
    { icon: "⬡", label: "Summary",           prompt: "Give me a plain English summary of what was found." },
    { icon: "◈", label: "Top priority",      prompt: "What is the single most important thing I should fix first?" },
    { icon: "⬟", label: "ML readiness",      prompt: "Is this dataset ready to train an ML model?" },
  ],
  default: [
    { icon: "⬡", label: "Summary",           prompt: "Summarize the analysis results for me." },
    { icon: "◈", label: "Next steps",        prompt: "What are the recommended next steps?" },
    { icon: "⬟", label: "Quality score",     prompt: "Explain the quality score and how to improve it." },
    { icon: "○", label: "Export",            prompt: "How do I export the cleaned dataset?" },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
}

function estimateTokens(text: string): number {
  return Math.ceil(text.split(/\s+/).length * 1.3);
}

async function copyToClipboard(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const el = document.createElement("textarea");
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand("copy");
    document.body.removeChild(el);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARKDOWN-LITE RENDERER
// Renders bold, inline code, code blocks, and bullet lists
// ─────────────────────────────────────────────────────────────────────────────

function renderMarkdownLite(text: string): React.ReactNode[] {
  const lines   = text.split("\n");
  const nodes: React.ReactNode[] = [];
  let codeBlock = false;
  let codeLines: string[] = [];
  let key = 0;

  const inlineRender = (line: string): React.ReactNode => {
    // Inline code `...`
    const parts = line.split(/(`[^`]+`)/g);
    return parts.map((part, i) => {
      if (part.startsWith("`") && part.endsWith("`")) {
        return (
          <code key={i} style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: "0.88em",
            background: "rgba(0,212,170,0.1)",
            border: "1px solid rgba(0,212,170,0.2)",
            borderRadius: 3, padding: "1px 5px",
            color: "#00d4aa",
          }}>
            {part.slice(1, -1)}
          </code>
        );
      }
      // Bold **...**
      const boldParts = part.split(/(\*\*[^*]+\*\*)/g);
      return boldParts.map((bp, j) =>
        bp.startsWith("**") && bp.endsWith("**")
          ? <strong key={j} style={{ color: "#e2e8f0", fontWeight: 700 }}>{bp.slice(2, -2)}</strong>
          : <span key={j}>{bp}</span>
      );
    });
  };

  for (const line of lines) {
    // Code fence
    if (line.startsWith("```")) {
      if (codeBlock) {
        nodes.push(
          <pre key={key++} style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 11, color: "#00d4aa",
            background: "rgba(0,212,170,0.05)",
            border: "1px solid rgba(0,212,170,0.15)",
            borderRadius: 6, padding: "10px 12px",
            overflowX: "auto", margin: "6px 0",
            whiteSpace: "pre-wrap", wordBreak: "break-all",
          }}>
            {codeLines.join("\n")}
          </pre>
        );
        codeLines = [];
        codeBlock = false;
      } else {
        codeBlock = true;
      }
      continue;
    }

    if (codeBlock) { codeLines.push(line); continue; }

    // Bullet list
    if (line.match(/^[-*•]\s/)) {
      nodes.push(
        <div key={key++} style={{ display: "flex", gap: 8, margin: "2px 0" }}>
          <span style={{ color: "#00d4aa", flexShrink: 0, marginTop: 1 }}>▸</span>
          <span>{inlineRender(line.slice(2))}</span>
        </div>
      );
      continue;
    }

    // Numbered list
    if (line.match(/^\d+\.\s/)) {
      const match = line.match(/^(\d+)\.\s(.*)$/);
      if (match) {
        nodes.push(
          <div key={key++} style={{ display: "flex", gap: 8, margin: "2px 0" }}>
            <span style={{ color: "#475569", flexShrink: 0, minWidth: 16 }}>{match[1]}.</span>
            <span>{inlineRender(match[2])}</span>
          </div>
        );
        continue;
      }
    }

    // Heading (## or #)
    if (line.startsWith("## ")) {
      nodes.push(
        <div key={key++} style={{
          fontFamily: "Syne, sans-serif",
          fontSize: 13, fontWeight: 700, color: "#00d4aa",
          margin: "10px 0 4px",
        }}>
          {line.slice(3)}
        </div>
      );
      continue;
    }

    if (line.startsWith("# ")) {
      nodes.push(
        <div key={key++} style={{
          fontFamily: "Syne, sans-serif",
          fontSize: 15, fontWeight: 800, color: "#e2e8f0",
          margin: "10px 0 4px",
        }}>
          {line.slice(2)}
        </div>
      );
      continue;
    }

    // Blank line
    if (!line.trim()) {
      nodes.push(<div key={key++} style={{ height: 6 }} />);
      continue;
    }

    // Regular text
    nodes.push(
      <div key={key++} style={{ margin: "1px 0" }}>
        {inlineRender(line)}
      </div>
    );
  }

  return nodes;
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

function TypingCursor() {
  return (
    <span style={{
      display: "inline-block",
      width: 7, height: 14,
      background: "#00d4aa",
      borderRadius: 1,
      marginLeft: 2,
      verticalAlign: "text-bottom",
      animation: "nydra-pulse 0.8s ease-in-out infinite",
    }} />
  );
}

function MessageBubble({
  message,
  onCopy,
}: {
  message: ChatMessage;
  onCopy: (text: string) => void;
}) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await onCopy(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div style={{
      display: "flex",
      flexDirection: isUser ? "row-reverse" : "row",
      gap: 10, alignItems: "flex-start",
    }}
      className="nydra-appear"
    >
      {/* Avatar */}
      <div style={{
        width: 28, height: 28, borderRadius: 6, flexShrink: 0,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: isUser
          ? "rgba(124,58,237,0.15)"
          : "rgba(0,212,170,0.12)",
        border: isUser
          ? "1px solid rgba(124,58,237,0.3)"
          : "1px solid rgba(0,212,170,0.2)",
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 10, fontWeight: 700,
        color: isUser ? "#7c3aed" : "#00d4aa",
      }}>
        {isUser ? "U" : "N"}
      </div>

      {/* Bubble */}
      <div style={{
        maxWidth: "78%",
        display: "flex", flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        gap: 4,
      }}>
        {/* Meta row */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          flexDirection: isUser ? "row-reverse" : "row",
        }}>
          <span style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 9, fontWeight: 700, letterSpacing: 1,
            color: isUser ? "#7c3aed" : "#00d4aa",
          }}>
            {isUser ? "YOU" : "NYDRA"}
          </span>
          <span style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 9, color: "#1e293b",
          }}>
            {formatTime(message.timestamp)}
          </span>
          {message.usedContext && (
            <span style={{
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 8, color: "#00d4aa", letterSpacing: 1,
              padding: "1px 5px", borderRadius: 3,
              background: "rgba(0,212,170,0.08)",
              border: "1px solid rgba(0,212,170,0.15)",
            }}>
              CTX
            </span>
          )}
        </div>

        {/* Content */}
        <div style={{
          padding: "10px 14px", borderRadius: 8,
          background: isUser
            ? "rgba(124,58,237,0.1)"
            : message.error
            ? "rgba(244,63,94,0.08)"
            : "rgba(8,14,27,0.8)",
          border: isUser
            ? "1px solid rgba(124,58,237,0.2)"
            : message.error
            ? "1px solid rgba(244,63,94,0.2)"
            : "1px solid rgba(51,65,85,0.4)",
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 12, color: message.error ? "#f43f5e" : "#94a3b8",
          lineHeight: 1.7,
          position: "relative",
        }}
          onMouseEnter={(e) => {
            const btn = e.currentTarget.querySelector(".copy-btn") as HTMLElement;
            if (btn) btn.style.opacity = "1";
          }}
          onMouseLeave={(e) => {
            const btn = e.currentTarget.querySelector(".copy-btn") as HTMLElement;
            if (btn) btn.style.opacity = "0";
          }}
        >
          {isUser
            ? message.content
            : renderMarkdownLite(message.content)
          }
          {message.isStreaming && <TypingCursor />}

          {/* Copy button */}
          {!message.isStreaming && (
            <button
              className="copy-btn"
              onClick={handleCopy}
              style={{
                position: "absolute", top: 6, right: 6,
                background: "rgba(15,23,42,0.8)",
                border: "1px solid rgba(51,65,85,0.5)",
                borderRadius: 4, padding: "2px 7px", cursor: "pointer",
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 8, color: copied ? "#22c55e" : "#475569",
                opacity: 0, transition: "opacity 0.2s, color 0.2s",
                letterSpacing: 0.5,
              }}
            >
              {copied ? "COPIED" : "COPY"}
            </button>
          )}
        </div>

        {/* Token count */}
        {message.tokensUsed && (
          <span style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 8, color: "#1e293b",
          }}>
            {message.tokensUsed} tokens
          </span>
        )}
      </div>
    </div>
  );
}

function SuggestedPromptChip({
  prompt, onSelect,
}: { prompt: SuggestedPrompt; onSelect: (p: string) => void }) {
  return (
    <button onClick={() => onSelect(prompt.prompt)} style={{
      display: "flex", alignItems: "center", gap: 6,
      padding: "6px 12px", borderRadius: 20, cursor: "pointer",
      background: "rgba(8,14,27,0.8)",
      border: "1px solid rgba(51,65,85,0.4)",
      fontFamily: "JetBrains Mono, monospace",
      fontSize: 10, color: "#475569",
      transition: "all 0.2s", whiteSpace: "nowrap",
    }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = "rgba(0,212,170,0.3)";
        e.currentTarget.style.color = "#00d4aa";
        e.currentTarget.style.background = "rgba(0,212,170,0.05)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = "rgba(51,65,85,0.4)";
        e.currentTarget.style.color = "#475569";
        e.currentTarget.style.background = "rgba(8,14,27,0.8)";
      }}
    >
      <span>{prompt.icon}</span>
      {prompt.label}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

export default function ChatBox({
  token, jobId, goal, filename, isMinimized = false, onToggleMinimize, className, apiUrl,
}: ChatBoxProps) {
  const [messages,    setMessages]    = useState<ChatMessage[]>([]);
  const [input,       setInput]       = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [wsStatus,    setWsStatus]    = useState<"disconnected" | "connecting" | "connected">("disconnected");
  const [tokenCount,  setTokenCount]  = useState(0);
  const [autoScroll,  setAutoScroll]  = useState(true);
  const [showSuggestions, setShowSuggestions] = useState(true);

  const API_BASE = apiUrl ?? DEFAULT_API_BASE;
  const WS_BASE  = API_BASE.replace(/^http/, "ws");

  const wsRef       = useRef<WebSocket | null>(null);
  const scrollRef   = useRef<HTMLDivElement>(null);
  const inputRef    = useRef<HTMLTextAreaElement>(null);
  const streamIdRef = useRef<string | null>(null);
  const mountedRef  = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // ── Suggested prompts based on goal ───────────────────────────────────
  const suggestions = useMemo(
    () => SUGGESTED_PROMPTS[goal ?? "default"] ?? SUGGESTED_PROMPTS.default,
    [goal],
  );

  // ── Token counter ──────────────────────────────────────────────────────
  useEffect(() => {
    setTokenCount(estimateTokens(input));
  }, [input]);

  // ── Auto-scroll ────────────────────────────────────────────────────────
  useLayoutEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, autoScroll]);

  const onScroll = useCallback(() => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    setAutoScroll(scrollHeight - scrollTop - clientHeight < 60);
  }, []);

  // ── WS Chat connection ─────────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const url = `${WS_BASE}/api/v1/ws/chat?token=${encodeURIComponent(token)}`;
    setWsStatus("connecting");

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen  = () => { if (mountedRef.current) setWsStatus("connected"); };
    ws.onclose = () => { if (mountedRef.current) setWsStatus("disconnected"); };
    ws.onerror = () => { if (mountedRef.current) setWsStatus("disconnected"); };

    ws.onmessage = (e) => {
      if (!mountedRef.current) return;
      try {
        const data = JSON.parse(e.data);
        const sid  = streamIdRef.current;
        if (!sid) return;

        if (data.token !== undefined) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === sid
                ? { ...m, content: m.content + data.token, isStreaming: true }
                : m
            )
          );
        }

        if (data.done) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === sid
                ? { ...m, isStreaming: false }
                : m
            )
          );
          streamIdRef.current = null;
          setIsStreaming(false);
        }

        if (data.error) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === sid
                ? { ...m, content: data.error, isStreaming: false, error: true }
                : m
            )
          );
          streamIdRef.current = null;
          setIsStreaming(false);
        }
      } catch { /* ignore malformed */ }
    };
  }, [token]);

  // ── REST fallback (when WS not available) ─────────────────────────────
  const sendViaRest = useCallback(async (
    history: ChatMessage[],
    assistantMsgId: string,
  ) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/chat`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: history.slice(-20).map((m) => ({
            role: m.role,
            content: m.content,
            timestamp: m.timestamp.toISOString(),
            message_id: m.id,
          })),
          job_id: jobId ?? null,
          stream: false,
          max_tokens: 1000,
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const reply = data.message?.content ?? "No response received.";

      if (mountedRef.current) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? {
                  ...m,
                  content: reply,
                  isStreaming: false,
                  tokensUsed: data.tokens_used,
                  usedContext: data.used_context,
                }
              : m
          )
        );
      }
    } catch (err: any) {
      if (mountedRef.current) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId
              ? { ...m, content: `Error: ${err.message}`, isStreaming: false, error: true }
              : m
          )
        );
      }
    } finally {
      if (mountedRef.current) {
        streamIdRef.current = null;
        setIsStreaming(false);
      }
    }
  }, [token, jobId]);

  // ── Send message ───────────────────────────────────────────────────────
  const sendMessage = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || isStreaming) return;

    setShowSuggestions(false);

    const userMsg: ChatMessage = {
      id: generateId(), role: "user",
      content: trimmed, timestamp: new Date(),
    };
    const assistantId = generateId();
    const assistantMsg: ChatMessage = {
      id: assistantId, role: "assistant",
      content: "", timestamp: new Date(), isStreaming: true,
    };

    const updatedHistory = [...messages, userMsg];
    setMessages([...updatedHistory, assistantMsg]);
    setInput("");
    setIsStreaming(true);
    streamIdRef.current = assistantId;
    setAutoScroll(true);

    // Trim history if too long
    if (updatedHistory.length > MAX_HISTORY) {
      setMessages((prev) => [
        ...prev.slice(prev.length - MAX_HISTORY),
        assistantMsg,
      ]);
    }

    // Use WS if connected, otherwise REST
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        message: trimmed,
        job_id: jobId ?? null,
      }));
    } else {
      await sendViaRest(updatedHistory, assistantId);
    }
  }, [messages, isStreaming, jobId, sendViaRest]);

  // ── Keyboard handler ───────────────────────────────────────────────────
  const onKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }, [input, sendMessage]);

  // ── Connect WS on mount ────────────────────────────────────────────────
  useEffect(() => {
    connectWS();
    return () => {
      wsRef.current?.close(1000, "Component unmounted");
    };
  }, [connectWS]);

  // ── Clear chat ─────────────────────────────────────────────────────────
  const clearChat = useCallback(() => {
    setMessages([]);
    setShowSuggestions(true);
    streamIdRef.current = null;
    setIsStreaming(false);
  }, []);

  const wsStatusColor = { disconnected: "#475569", connecting: "#f59e0b", connected: "#22c55e" }[wsStatus];
  const isOverLimit   = tokenCount > MAX_TOKENS * 0.8;
  const canSend       = input.trim().length > 0 && !isStreaming && tokenCount <= MAX_TOKENS;

  if (isMinimized) {
    return (
      <div
        onClick={onToggleMinimize}
        style={{
          padding: "10px 16px", cursor: "pointer",
          background: "rgba(8,14,27,0.9)",
          border: "1px solid rgba(51,65,85,0.5)",
          borderRadius: 8,
          display: "flex", alignItems: "center", gap: 10,
        }}
      >
        <div style={{ width: 6, height: 6, borderRadius: "50%", background: wsStatusColor }} />
        <span style={{
          fontFamily: "JetBrains Mono, monospace",
          fontSize: 10, fontWeight: 700, letterSpacing: 1.5, color: "#475569",
        }}>
          NYDRA AI
        </span>
        {messages.length > 0 && (
          <span style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 9, color: "#334155",
          }}>
            {messages.length} messages
          </span>
        )}
        <span style={{ marginLeft: "auto", color: "#334155", fontSize: 12 }}>▲</span>
      </div>
    );
  }

  return (
    <div
      className={className}
      style={{
        display: "flex", flexDirection: "column",
        height: "100%", minHeight: 480,
        background: "rgba(5,9,18,0.95)",
        border: "1px solid rgba(51,65,85,0.5)",
        borderRadius: 10, overflow: "hidden",
        fontFamily: "JetBrains Mono, monospace",
      }}
    >
      {/* ── Header ────────────────────────────────────────────────── */}
      <div style={{
        padding: "12px 16px",
        borderBottom: "1px solid rgba(15,23,42,0.8)",
        display: "flex", alignItems: "center", gap: 10,
        background: "rgba(8,14,27,0.8)",
      }}>
        {/* Status dot */}
        <div style={{
          width: 7, height: 7, borderRadius: "50%",
          background: wsStatusColor,
          boxShadow: wsStatus === "connected" ? `0 0 6px ${wsStatusColor}` : "none",
          animation: wsStatus === "connecting" ? "nydra-pulse 1s ease-in-out infinite" : "none",
        }} />

        <div style={{ flex: 1 }}>
          <div style={{
            fontFamily: "Syne, sans-serif",
            fontSize: 13, fontWeight: 700, color: "#e2e8f0",
          }}>
            Nydra AI
          </div>
          <div style={{ fontSize: 9, color: "#334155", marginTop: 1 }}>
            {jobId
              ? `Grounded in analysis · ${jobId.slice(0, 8)}...`
              : "General data intelligence assistant"
            }
            {filename && ` · ${filename}`}
          </div>
        </div>

        <div style={{ display: "flex", gap: 6 }}>
          {messages.length > 0 && (
            <button onClick={clearChat} style={{
              background: "none", border: "1px solid rgba(51,65,85,0.4)",
              borderRadius: 4, padding: "3px 8px", cursor: "pointer",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 8, color: "#334155", letterSpacing: 0.5,
              transition: "all 0.2s",
            }}
              onMouseEnter={(e) => { e.currentTarget.style.color = "#f43f5e"; e.currentTarget.style.borderColor = "rgba(244,63,94,0.3)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = "#334155"; e.currentTarget.style.borderColor = "rgba(51,65,85,0.4)"; }}
            >
              CLEAR
            </button>
          )}
          {onToggleMinimize && (
            <button onClick={onToggleMinimize} style={{
              background: "none", border: "1px solid rgba(51,65,85,0.4)",
              borderRadius: 4, padding: "3px 8px", cursor: "pointer",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 10, color: "#334155",
            }}>
              ▼
            </button>
          )}
        </div>
      </div>

      {/* ── Messages ────────────────────────────────────────────────── */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        style={{
          flex: 1, overflowY: "auto", padding: 16,
          display: "flex", flexDirection: "column", gap: 16,
          scrollbarWidth: "thin",
          scrollbarColor: "rgba(51,65,85,0.4) transparent",
        }}
      >
        {/* Empty state */}
        {messages.length === 0 && (
          <div style={{ textAlign: "center", padding: "32px 16px" }}>
            <div style={{
              width: 48, height: 48, borderRadius: 10, margin: "0 auto 12px",
              background: "rgba(0,212,170,0.08)",
              border: "1px solid rgba(0,212,170,0.15)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 20, color: "#00d4aa",
            }}>
              ⬡
            </div>
            <div style={{
              fontFamily: "Syne, sans-serif",
              fontSize: 14, fontWeight: 700, color: "#334155", marginBottom: 6,
            }}>
              Ask Nydra anything
            </div>
            <div style={{ fontSize: 10, color: "#1e293b" }}>
              {jobId
                ? "I have full context of your analysis results"
                : "Ask about your data, ML strategies, or data quality"
              }
            </div>
          </div>
        )}

        {/* Message list */}
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            onCopy={copyToClipboard}
          />
        ))}

        {/* Scroll-to-bottom button */}
        {!autoScroll && (
          <button
            onClick={() => { setAutoScroll(true); scrollRef.current?.scrollTo({ top: 99999, behavior: "smooth" }); }}
            style={{
              position: "sticky", bottom: 0, alignSelf: "center",
              padding: "6px 14px", borderRadius: 20,
              background: "rgba(0,212,170,0.12)",
              border: "1px solid rgba(0,212,170,0.25)",
              cursor: "pointer", fontFamily: "JetBrains Mono, monospace",
              fontSize: 9, color: "#00d4aa", letterSpacing: 1,
            }}
          >
            ↓ SCROLL TO BOTTOM
          </button>
        )}
      </div>

      {/* ── Suggested prompts ───────────────────────────────────────── */}
      {showSuggestions && messages.length === 0 && (
        <div style={{
          padding: "0 16px 12px",
          display: "flex", flexWrap: "wrap", gap: 6,
        }}>
          {suggestions.map((s) => (
            <SuggestedPromptChip key={s.label} prompt={s} onSelect={sendMessage} />
          ))}
        </div>
      )}

      {/* ── Input area ──────────────────────────────────────────────── */}
      <div style={{
        padding: "12px 14px",
        borderTop: "1px solid rgba(15,23,42,0.8)",
        background: "rgba(8,14,27,0.6)",
      }}>
        <div style={{
          display: "flex", gap: 8, alignItems: "flex-end",
          background: "rgba(5,9,18,0.8)",
          border: `1px solid ${isStreaming ? "rgba(0,212,170,0.2)" : "rgba(51,65,85,0.5)"}`,
          borderRadius: 8, padding: "8px 10px",
          transition: "border-color 0.3s",
        }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={isStreaming}
            placeholder={isStreaming ? "Nydra is thinking..." : "Ask anything about your data..."}
            rows={1}
            style={{
              flex: 1, background: "none", border: "none", outline: "none",
              resize: "none", fontFamily: "JetBrains Mono, monospace",
              fontSize: 12, color: "#94a3b8",
              lineHeight: 1.6, maxHeight: 120, overflowY: "auto",
            }}
            onInput={(e) => {
              const el = e.currentTarget;
              el.style.height = "auto";
              el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
            }}
          />
          {/* Token counter */}
          <div style={{
            fontFamily: "JetBrains Mono, monospace",
            fontSize: 8, color: isOverLimit ? "#f43f5e" : "#1e293b",
            alignSelf: "flex-end", flexShrink: 0, transition: "color 0.2s",
          }}>
            {tokenCount}/{MAX_TOKENS}
          </div>
          {/* Send button */}
          <button
            onClick={() => sendMessage(input)}
            disabled={!canSend}
            style={{
              width: 32, height: 32, borderRadius: 6, flexShrink: 0,
              background: canSend
                ? "rgba(0,212,170,0.15)"
                : "rgba(15,23,42,0.4)",
              border: `1px solid ${canSend ? "rgba(0,212,170,0.4)" : "rgba(30,41,59,0.6)"}`,
              cursor: canSend ? "pointer" : "not-allowed",
              display: "flex", alignItems: "center", justifyContent: "center",
              color: canSend ? "#00d4aa" : "#1e293b",
              fontSize: 14, transition: "all 0.2s",
            }}
            onMouseEnter={(e) => canSend && (e.currentTarget.style.background = "rgba(0,212,170,0.25)")}
            onMouseLeave={(e) => canSend && (e.currentTarget.style.background = "rgba(0,212,170,0.15)")}
          >
            {isStreaming ? (
              <span className="nydra-pulse">⬡</span>
            ) : "↑"}
          </button>
        </div>

        {/* Footer hint */}
        <div style={{
          display: "flex", justifyContent: "space-between",
          marginTop: 6, fontSize: 8, color: "#1e293b",
        }}>
          <span>ENTER to send · SHIFT+ENTER for new line</span>
          <span style={{ color: wsStatusColor }}>
            {wsStatus === "connected" ? "● WS LIVE" : wsStatus === "connecting" ? "● CONNECTING" : "● REST MODE"}
          </span>
        </div>
      </div>
    </div>
  );
}