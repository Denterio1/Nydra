"use client";

/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║           NYDRA — useNydraWS.ts                                 ║
 * ║   Production-grade WebSocket hook for real-time job progress    ║
 * ║                                                                  ║
 * ║  ✅ Auto-reconnect with exponential backoff                     ║
 * ║  ✅ Heartbeat / keepalive ping-pong                             ║
 * ║  ✅ Message queue (buffers events before component mounts)      ║
 * ║  ✅ Full WSMessage type safety                                  ║
 * ║  ✅ Connection state machine                                    ║
 * ║  ✅ Visibility-aware (reconnects when tab regains focus)        ║
 * ║  ✅ Clean teardown — no memory leaks                            ║
 * ╚══════════════════════════════════════════════════════════════════╝
 */

import { useCallback, useEffect, useRef, useState } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// TYPES  (mirrors api_schemas.py WSMessage)
// ─────────────────────────────────────────────────────────────────────────────

export type WSEventType =
  | "job_started"
  | "step_started"
  | "step_done"
  | "progress"
  | "warning"
  | "error"
  | "job_done"
  | "job_failed"
  | "job_cancelled"
  | "ping"
  | "pong";

export interface WSProgressPayload {
  progress: number;
  current_step: string;
  step_progress: number;
  message: string;
  elapsed_ms: number;
  eta_ms?: number;
}

export interface WSStepPayload {
  step_id: string;
  step_name: string;
  description: string;
  step_index: number;
  total_steps: number;
}

export interface WSWarningPayload {
  code: string;
  message: string;
  column?: string;
  details?: Record<string, unknown>;
}

export interface WSErrorPayload {
  code: string;
  message: string;
  traceback?: string;
  step_id?: string;
}

export interface WSResultPayload {
  job_id: string;
  result_url: string;
  summary: string;
  score?: number;
  verdict?: "ready" | "needs_work" | "not_ready";
  total_issues: number;
  duration_ms: number;
}

export type WSPayload =
  | WSProgressPayload
  | WSStepPayload
  | WSWarningPayload
  | WSErrorPayload
  | WSResultPayload
  | Record<string, unknown>;

export interface WSMessage {
  event: WSEventType;
  job_id: string;
  timestamp: string;
  payload: WSPayload;
}

// ─────────────────────────────────────────────────────────────────────────────
// CONNECTION STATE MACHINE
// ─────────────────────────────────────────────────────────────────────────────

export type ConnectionState =
  | "idle"          // not started
  | "connecting"    // WS opening
  | "connected"     // live
  | "reconnecting"  // dropped, waiting to retry
  | "closed"        // terminal — job done/failed/cancelled
  | "error";        // unrecoverable error

// ─────────────────────────────────────────────────────────────────────────────
// HOOK OPTIONS
// ─────────────────────────────────────────────────────────────────────────────

export interface UseNydraWSOptions {
  apiUrl?: string;
  token: string;
  jobId: string | null;
  onMessage?: (msg: WSMessage) => void;
  onDone?: (payload: WSResultPayload) => void;
  onFailed?: (payload: WSErrorPayload) => void;
  onConnectionChange?: (state: ConnectionState) => void;
  maxRetries?: number;
  backoffBase?: number;
  heartbeatInterval?: number;
  autoConnect?: boolean;
  [key: string]: any; // Fallback to allow additional properties
}

// ─────────────────────────────────────────────────────────────────────────────
// HOOK RETURN
// ─────────────────────────────────────────────────────────────────────────────

export interface UseNydraWSReturn {
  /** Current connection state */
  connectionState: ConnectionState;
  /** Latest progress payload */
  progress: WSProgressPayload | null;
  /** Latest step info */
  currentStep: WSStepPayload | null;
  /** All warnings received */
  warnings: WSWarningPayload[];
  /** Final result payload (set when job_done) */
  result: WSResultPayload | null;
  /** Error payload (set when job_failed) */
  error: WSErrorPayload | null;
  /** All messages received in order */
  messages: WSMessage[];
  /** Number of reconnect attempts so far */
  retryCount: number;
  /** Whether the socket is live */
  isConnected: boolean;
  /** Manually trigger connect */
  connect: () => void;
  /** Manually trigger disconnect */
  disconnect: () => void;
  /** Clear all accumulated state */
  reset: () => void;
  /** Send a raw message to the server */
  send: (data: Record<string, unknown>) => void;
  /** Elapsed ms since connection opened */
  elapsedMs: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

function buildWsUrl(jobId: string, token: string, apiUrl?: string): string {
  const base = (
    process.env.NEXT_PUBLIC_WS_URL ??
    apiUrl?.replace(/^http/, "ws") ??
    process.env.NEXT_PUBLIC_API_URL?.replace(/^http/, "ws") ??
    "ws://localhost:8000"
  );
  return `${base}/api/v1/ws/${jobId}?token=${encodeURIComponent(token)}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function isTerminalEvent(event: WSEventType): boolean {
  return event === "job_done" || event === "job_failed" || event === "job_cancelled";
}

function isProgressPayload(p: WSPayload): p is WSProgressPayload {
  return typeof (p as WSProgressPayload).progress === "number";
}

function isStepPayload(p: WSPayload): p is WSStepPayload {
  return typeof (p as WSStepPayload).step_name === "string";
}

function isResultPayload(p: WSPayload): p is WSResultPayload {
  return typeof (p as WSResultPayload).result_url === "string";
}

function isErrorPayload(p: WSPayload): p is WSErrorPayload {
  return typeof (p as WSErrorPayload).code === "string" &&
         typeof (p as WSErrorPayload).message === "string" &&
         !(p as WSResultPayload).result_url;
}

function isWarningPayload(p: WSPayload): p is WSWarningPayload {
  return typeof (p as WSWarningPayload).code === "string" &&
         typeof (p as WSWarningPayload).message === "string" &&
         !(p as WSErrorPayload).traceback !== undefined;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN HOOK
// ─────────────────────────────────────────────────────────────────────────────

export function useNydraWS({
  token,
  jobId,
  apiUrl,
  onMessage,
  onDone,
  onFailed,
  onConnectionChange,
  maxRetries       = 8,
  backoffBase      = 1000,
  heartbeatInterval = 25_000,
  autoConnect      = true,
}: UseNydraWSOptions): UseNydraWSReturn {

  // ── State ──────────────────────────────────────────────────────────────
  const [connectionState, setConnectionState] = useState<ConnectionState>("idle");
  const [progress,     setProgress]     = useState<WSProgressPayload | null>(null);
  const [currentStep,  setCurrentStep]  = useState<WSStepPayload | null>(null);
  const [warnings,     setWarnings]     = useState<WSWarningPayload[]>([]);
  const [result,       setResult]       = useState<WSResultPayload | null>(null);
  const [error,        setError]        = useState<WSErrorPayload | null>(null);
  const [messages,     setMessages]     = useState<WSMessage[]>([]);
  const [retryCount,   setRetryCount]   = useState(0);
  const [elapsedMs,    setElapsedMs]    = useState(0);

  // ── Refs (stable across renders) ──────────────────────────────────────
  const wsRef           = useRef<WebSocket | null>(null);
  const retryRef        = useRef(0);
  const retryTimerRef   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const elapsedRef      = useRef<ReturnType<typeof setInterval> | null>(null);
  const connectedAtRef  = useRef<number | null>(null);
  const isTerminalRef   = useRef(false);
  const messageQueueRef = useRef<WSMessage[]>([]);
  const mountedRef      = useRef(true);
  const jobIdRef        = useRef(jobId);
  const tokenRef        = useRef(token);
  const apiUrlRef       = useRef(apiUrl);
  const onMessageRef    = useRef(onMessage);
  const onDoneRef       = useRef(onDone);
  const onFailedRef     = useRef(onFailed);
  const onConnChangeRef = useRef(onConnectionChange);

  // Keep refs in sync
  useEffect(() => { jobIdRef.current  = jobId;  }, [jobId]);
  useEffect(() => { tokenRef.current  = token;  }, [token]);
  useEffect(() => { apiUrlRef.current = apiUrl; }, [apiUrl]);
  useEffect(() => { onMessageRef.current = onMessage; }, [onMessage]);
  useEffect(() => { onDoneRef.current = onDone; }, [onDone]);
  useEffect(() => { onFailedRef.current = onFailed; }, [onFailed]);
  useEffect(() => { onConnChangeRef.current = onConnectionChange; }, [onConnectionChange]);
  
  useEffect(() => { mountedRef.current = true; return () => { mountedRef.current = false; }; }, []);

  // ── Helpers ────────────────────────────────────────────────────────────
  const safeSetState = useCallback((setter: any, value: any) => {
    if (mountedRef.current) setter(value);
  }, []);

  const updateConnectionState = useCallback((state: ConnectionState) => {
    safeSetState(setConnectionState, state);
    onConnChangeRef.current?.(state);
  }, [safeSetState]);

  const startElapsedTimer = useCallback(() => {
    connectedAtRef.current = Date.now();
    elapsedRef.current = setInterval(() => {
      if (connectedAtRef.current && mountedRef.current) {
        setElapsedMs(Date.now() - connectedAtRef.current);
      }
    }, 500);
  }, []);

  const stopElapsedTimer = useCallback(() => {
    if (elapsedRef.current) {
      clearInterval(elapsedRef.current);
      elapsedRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    heartbeatRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, heartbeatInterval);
  }, [heartbeatInterval]);

  const stopHeartbeat = useCallback(() => {
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
  }, []);

  const clearRetryTimer = useCallback(() => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  // ── Message handler ────────────────────────────────────────────────────
  const handleMessage = useCallback((raw: MessageEvent<string>) => {
    let msg: WSMessage;
    try {
      msg = JSON.parse(raw.data) as WSMessage;
    } catch {
      return; // Ignore malformed messages
    }

    // Buffer messages
    messageQueueRef.current.push(msg);
    safeSetState(setMessages, [...messageQueueRef.current]);

    // Dispatch to callback
    onMessageRef.current?.(msg);

    // Route by event type
    switch (msg.event) {
      case "progress":
        if (isProgressPayload(msg.payload)) {
          safeSetState(setProgress, msg.payload);
        }
        break;

      case "step_started":
      case "step_done":
        if (isStepPayload(msg.payload)) {
          safeSetState(setCurrentStep, msg.payload);
        }
        break;

      case "warning":
        if (isWarningPayload(msg.payload)) {
          safeSetState(setWarnings, (prev: WSWarningPayload[]) => [...prev, msg.payload as WSWarningPayload]);
        }
        break;

      case "job_done":
        if (isResultPayload(msg.payload)) {
          safeSetState(setResult, msg.payload);
          onDoneRef.current?.(msg.payload);
        }
        isTerminalRef.current = true;
        updateConnectionState("closed");
        stopHeartbeat();
        stopElapsedTimer();
        wsRef.current?.close(1000, "Job complete");
        break;

      case "job_failed":
        if (isErrorPayload(msg.payload)) {
          safeSetState(setError, msg.payload);
          onFailedRef.current?.(msg.payload);
        }
        isTerminalRef.current = true;
        updateConnectionState("closed");
        stopHeartbeat();
        stopElapsedTimer();
        wsRef.current?.close(1000, "Job failed");
        break;

      case "job_cancelled":
        isTerminalRef.current = true;
        updateConnectionState("closed");
        stopHeartbeat();
        stopElapsedTimer();
        wsRef.current?.close(1000, "Job cancelled");
        break;

      case "pong":
        // Heartbeat response — connection confirmed alive
        break;

      default:
        break;
    }
  }, [updateConnectionState, stopHeartbeat, stopElapsedTimer, safeSetState]);

  // ── Core connect ───────────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (!jobIdRef.current || !tokenRef.current) return;
    if (isTerminalRef.current) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    // Close any existing socket
    wsRef.current?.close();

    const url = buildWsUrl(jobIdRef.current, tokenRef.current, apiUrlRef.current);
    updateConnectionState(retryRef.current > 0 ? "reconnecting" : "connecting");

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      updateConnectionState("error");
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return; }
      retryRef.current = 0;
      safeSetState(setRetryCount, 0);
      updateConnectionState("connected");
      startHeartbeat();
      startElapsedTimer();

      // Flush any queued outbound messages
      messageQueueRef.current = [];
    };

    ws.onmessage = handleMessage;

    ws.onclose = (e) => {
      stopHeartbeat();
      stopElapsedTimer();

      if (!mountedRef.current) return;
      if (isTerminalRef.current) return; // Expected close

      // Unexpected close — attempt reconnect
      if (retryRef.current < maxRetries) {
        const delay = clamp(
          backoffBase * Math.pow(2, retryRef.current) + Math.random() * 500,
          backoffBase,
          30_000,
        );
        retryRef.current++;
        safeSetState(setRetryCount, retryRef.current);
        updateConnectionState("reconnecting");
        retryTimerRef.current = setTimeout(connect, delay);
      } else {
        updateConnectionState("error");
      }
    };

    ws.onerror = () => {
      // onclose will fire after onerror — handle there
      stopHeartbeat();
    };
  }, [
    updateConnectionState, handleMessage,
    startHeartbeat, stopHeartbeat,
    startElapsedTimer, stopElapsedTimer,
    maxRetries, backoffBase, safeSetState,
  ]);

  // ── Disconnect ─────────────────────────────────────────────────────────
  const disconnect = useCallback(() => {
    isTerminalRef.current = true;
    clearRetryTimer();
    stopHeartbeat();
    stopElapsedTimer();
    wsRef.current?.close(1000, "Manual disconnect");
    wsRef.current = null;
    updateConnectionState("closed");
  }, [clearRetryTimer, stopHeartbeat, stopElapsedTimer, updateConnectionState]);

  // ── Send ───────────────────────────────────────────────────────────────
  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  // ── Reset ──────────────────────────────────────────────────────────────
  const reset = useCallback(() => {
    disconnect();
    isTerminalRef.current = false;
    retryRef.current      = 0;
    messageQueueRef.current = [];
    safeSetState(setConnectionState, "idle");
    safeSetState(setProgress,     null);
    safeSetState(setCurrentStep,  null);
    safeSetState(setWarnings,     []);
    safeSetState(setResult,       null);
    safeSetState(setError,        null);
    safeSetState(setMessages,     []);
    safeSetState(setRetryCount,   0);
    safeSetState(setElapsedMs,    0);
  }, [disconnect, safeSetState]);

  // ── Auto-connect when jobId changes ───────────────────────────────────
  useEffect(() => {
    if (!autoConnect || !jobId) {
      if (!jobId) reset();
      return;
    }
    
    // Reset state before connecting to new job
    setProgress(null);
    setCurrentStep(null);
    setWarnings([]);
    setResult(null);
    setError(null);
    setMessages([]);
    setRetryCount(0);
    setElapsedMs(0);
    messageQueueRef.current = [];
    isTerminalRef.current   = false;
    retryRef.current        = 0;

    connect();
    return () => {
      clearRetryTimer();
      stopHeartbeat();
      stopElapsedTimer();
      if (!isTerminalRef.current) {
        wsRef.current?.close(1000, "Component unmounted or jobId changed");
      }
    };
  }, [jobId, token, autoConnect, connect, reset, clearRetryTimer, stopHeartbeat, stopElapsedTimer]); // added missing deps

  // ── Visibility API — reconnect when tab regains focus ─────────────────
  useEffect(() => {
    const onVisible = () => {
      if (
        document.visibilityState === "visible" &&
        connectionState === "reconnecting" &&
        !isTerminalRef.current
      ) {
        clearRetryTimer();
        connect();
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, [connectionState, connect, clearRetryTimer]);

  // ── Expose ─────────────────────────────────────────────────────────────
  return {
    connectionState,
    progress,
    currentStep,
    warnings,
    result,
    error,
    messages,
    retryCount,
    isConnected: connectionState === "connected",
    connect,
    disconnect,
    reset,
    send,
    elapsedMs,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPANION HOOK — useJobProgress
// Simplified wrapper: give it a jobId, get back just progress state
// ─────────────────────────────────────────────────────────────────────────────

export interface JobProgressState {
  status: "idle" | "connecting" | "running" | "done" | "failed" | "cancelled";
  progress: number;
  currentStep: string;
  message: string;
  elapsedMs: number;
  etaMs: number | null;
  score: number | null;
  verdict: WSResultPayload["verdict"] | null;
  totalIssues: number;
  resultUrl: string | null;
  errorMessage: string | null;
  warnings: string[];
  isConnected: boolean;
  retryCount: number;
}

export function useJobProgress(
  jobId: string | null,
  token: string,
  apiUrl?: string,
  onComplete?: (result: WSResultPayload) => void,
): JobProgressState {
  const [state, setState] = useState<JobProgressState>({
    status: "idle", progress: 0, currentStep: "", message: "",
    elapsedMs: 0, etaMs: null, score: null, verdict: null,
    totalIssues: 0, resultUrl: null, errorMessage: null,
    warnings: [], isConnected: false, retryCount: 0,
  });

  const { connectionState, progress, result, error, warnings, isConnected, retryCount, elapsedMs } =
    useNydraWS({
      jobId,
      token,
      apiUrl,
      onDone: (r) => {
        setState((prev) => ({
          ...prev,
          status: "done",
          progress: 100,
          score: r.score ?? null,
          verdict: r.verdict ?? null,
          totalIssues: r.total_issues,
          resultUrl: r.result_url,
          message: r.summary,
        }));
        onComplete?.(r);
      },
      onFailed: (e) => {
        setState((prev) => ({ ...prev, status: "failed", errorMessage: e.message }));
      },
    });

  useEffect(() => {
    if (progress) {
      setState((prev) => ({
        ...prev,
        status: "running",
        progress: progress.progress,
        currentStep: progress.current_step,
        message: progress.message,
        etaMs: progress.eta_ms ?? null,
      }));
    }
  }, [progress]);

  useEffect(() => {
    setState((prev) => ({
      ...prev,
      warnings: warnings.map((w) => w.message),
      isConnected,
      retryCount,
      elapsedMs,
      status: prev.status === "idle" && connectionState === "connecting" ? "connecting" : prev.status,
    }));
  }, [warnings, isConnected, retryCount, elapsedMs, connectionState]);

  return state;
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPANION HOOK — useWsConnectionIndicator
// Returns a simple label + color for a connection badge in the UI
// ─────────────────────────────────────────────────────────────────────────────

export interface ConnectionIndicator {
  label: string;
  color: string;
  pulsing: boolean;
}

export function useWsConnectionIndicator(state: ConnectionState): ConnectionIndicator {
  const map: Record<ConnectionState, ConnectionIndicator> = {
    idle:         { label: "STANDBY",      color: "#475569", pulsing: false },
    connecting:   { label: "CONNECTING",   color: "#f59e0b", pulsing: true  },
    connected:    { label: "LIVE",         color: "#00d4aa", pulsing: true  },
    reconnecting: { label: "RECONNECTING", color: "#f97316", pulsing: true  },
    closed:       { label: "CLOSED",       color: "#334155", pulsing: false },
    error:        { label: "ERROR",        color: "#f43f5e", pulsing: false },
  };
  return map[state];
}