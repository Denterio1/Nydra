"use client";

import { useCallback, useEffect, useState } from "react";

type Props = { token: string | null; apiBase: string };

type ApiKey = {
  id: string;
  name: string;
  key_prefix: string;
  created_at?: string | null;
  expires_at?: string | null;
  last_used?: string | null;
  is_active: boolean;
};

const fmt = (d?: string | null) => (d ? new Date(d).toLocaleDateString() : "never");

export default function ApiKeysTab({ token, apiBase }: Props) {
  const base = apiBase.replace(/\/+$/, "");
  const headers = { Authorization: `Bearer ${token ?? ""}`, "Content-Type": "application/json" };

  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [days, setDays] = useState("0");
  const [busy, setBusy] = useState(false);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch(`${base}/api/v1/auth/api-keys`, { headers });
      if (!res.ok) throw new Error(`Could not load keys (${res.status})`);
      setKeys(await res.json());
    } catch (e: any) {
      setError(e?.message ?? "Could not load keys");
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base, token]);

  useEffect(() => {
    load();
  }, [load]);

  const create = async () => {
    if (!name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const body: { name: string; expires_in?: number } = { name: name.trim() };
      if (Number(days) > 0) body.expires_in = Number(days);
      const res = await fetch(`${base}/api/v1/auth/api-keys`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`Could not create key (${res.status})`);
      const data = await res.json();
      setNewKey(data.full_key);
      setCopied(false);
      setName("");
      await load();
    } catch (e: any) {
      setError(e?.message ?? "Could not create key");
    } finally {
      setBusy(false);
    }
  };

  const revoke = async (k: ApiKey) => {
    if (!window.confirm(`Revoke "${k.name}"? Anything using this key will stop working.`)) return;
    setError(null);
    try {
      const res = await fetch(`${base}/api/v1/auth/api-keys/${k.id}`, { method: "DELETE", headers });
      if (!res.ok && res.status !== 204) throw new Error(`Could not revoke key (${res.status})`);
      await load();
    } catch (e: any) {
      setError(e?.message ?? "Could not revoke key");
    }
  };

  const copy = async () => {
    if (!newKey) return;
    try {
      await navigator.clipboard.writeText(newKey);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="max-h-[320px] overflow-y-auto pr-2 space-y-5">
      <div>
        <h3 className="text-sm font-bold text-white">API keys</h3>
        <p className="text-[11px] text-white/40 mt-1">
          Keys let scripts and other services call the Nydra API on your behalf. Treat them like passwords.
        </p>
      </div>

      {error && <p className="text-[11px] text-red-400">{error}</p>}

      {newKey && (
        <div className="rounded-xl border border-[#c8f06e]/30 bg-[#c8f06e]/5 p-4 space-y-2">
          <p className="text-[11px] text-[#c8f06e] font-bold">
            Copy your new key now. You will not be able to see it again.
          </p>
          <div className="flex gap-2">
            <code className="flex-1 break-all rounded-lg bg-black/40 px-3 py-2 text-[11px] text-white/80">
              {newKey}
            </code>
            <button
              onClick={copy}
              className="px-4 rounded-lg bg-[#c8f06e] text-[#0f172a] text-[10px] font-bold uppercase tracking-widest"
            >
              {copied ? "Copied" : "Copy"}
            </button>
            <button
              onClick={() => setNewKey(null)}
              className="px-3 rounded-lg border border-white/10 text-white/50 text-[10px] uppercase tracking-widest"
            >
              Done
            </button>
          </div>
        </div>
      )}

      <div className="flex gap-2 items-center">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          maxLength={100}
          placeholder="Key name (e.g. my-script)"
          className="flex-1 rounded-lg bg-black/30 border border-white/10 px-3 py-2 text-xs text-white outline-none focus:border-[#c8f06e]/40"
        />
        <select
          value={days}
          onChange={(e) => setDays(e.target.value)}
          className="rounded-lg bg-black/30 border border-white/10 px-3 py-2 text-xs text-white outline-none"
        >
          <option value="0">No expiry</option>
          <option value="30">30 days</option>
          <option value="90">90 days</option>
          <option value="365">1 year</option>
        </select>
        <button
          onClick={create}
          disabled={busy || !name.trim()}
          className="px-4 py-2 rounded-lg bg-[#c8f06e] text-[#0f172a] text-[10px] font-bold uppercase tracking-widest disabled:opacity-40"
        >
          {busy ? "Creating..." : "Create key"}
        </button>
      </div>

      {loading ? (
        <p className="text-[11px] text-white/30">Loading...</p>
      ) : keys.length === 0 ? (
        <p className="text-[11px] text-white/30">No active keys yet.</p>
      ) : (
        <div className="space-y-2">
          {keys.map((k) => (
            <div
              key={k.id}
              className="flex items-center justify-between rounded-xl border border-white/5 bg-black/20 px-4 py-3"
            >
              <div className="min-w-0">
                <p className="text-xs font-bold text-white truncate">{k.name}</p>
                <p className="text-[10px] text-white/40 font-mono">
                  {k.key_prefix}… · created {fmt(k.created_at)} · last used {fmt(k.last_used)} · expires{" "}
                  {k.expires_at ? fmt(k.expires_at) : "never"}
                </p>
              </div>
              <button
                onClick={() => revoke(k)}
                className="ml-4 px-3 py-1.5 rounded-lg border border-red-400/30 text-red-300 text-[10px] uppercase tracking-widest hover:bg-red-400/10"
              >
                Revoke
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}