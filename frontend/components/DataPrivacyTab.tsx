"use client";

import { useCallback, useEffect, useState } from "react";

type Props = { token: string | null; apiBase: string };

type FileRow = {
  file_id: string;
  filename: string;
  file_type: string;
  size_bytes: number;
  uploaded_at: string | null;
};
type JobRow = {
  job_id: string;
  goal: string;
  status: string;
  filename: string;
  created_at: string;
};

const mb = (b: number) => (b / (1024 * 1024)).toFixed(b < 10 * 1024 * 1024 ? 2 : 1);
const day = (d?: string | null) => (d ? new Date(d).toLocaleDateString() : "");

export default function DataPrivacyTab({ token, apiBase }: Props) {
  const base = apiBase.replace(/\/+$/, "");
  const headers = { Authorization: `Bearer ${token ?? ""}` };

  const [files, setFiles] = useState<FileRow[]>([]);
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [usedBytes, setUsedBytes] = useState(0);
  const [quotaMb, setQuotaMb] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [fr, jr] = await Promise.all([
        fetch(`${base}/api/v1/files`, { headers }),
        fetch(`${base}/api/v1/jobs?per_page=100`, { headers }),
      ]);
      if (!fr.ok) throw new Error(`Could not load files (${fr.status})`);
      if (!jr.ok) throw new Error(`Could not load jobs (${jr.status})`);
      const f = await fr.json();
      const j = await jr.json();
      setFiles(f.files ?? []);
      setUsedBytes(f.used_bytes ?? 0);
      setQuotaMb(f.quota_mb ?? 0);
      setJobs(j.jobs ?? []);
    } catch (e: any) {
      setError(e?.message ?? "Could not load your data");
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base, token]);

  useEffect(() => {
    load();
  }, [load]);

  const remove = async (url: string, what: string) => {
    if (!window.confirm(`Delete ${what}? This cannot be undone.`)) return;
    setError(null);
    try {
      const res = await fetch(url, { method: "DELETE", headers });
      if (res.status === 409) throw new Error("This job is still running. Try again when it finishes.");
      if (!res.ok && res.status !== 204) throw new Error(`Delete failed (${res.status})`);
      await load();
    } catch (e: any) {
      setError(e?.message ?? "Delete failed");
    }
  };

  const pct = quotaMb > 0 ? Math.min(100, (usedBytes / (quotaMb * 1024 * 1024)) * 100) : 0;

  return (
    <div className="max-h-[320px] overflow-y-auto pr-2 space-y-5">
      <div>
        <h3 className="text-sm font-bold text-white">Data &amp; privacy</h3>
        <p className="text-[11px] text-white/40 mt-1">
          Deleting a file removes the upload. Deleting a job removes its result and report files. Reports, cleaned
          files and large-table samples are also removed automatically after 7 days; temporary working folders
          after 24 hours.
        </p>
      </div>

      {error && <p className="text-[11px] text-red-400">{error}</p>}

      {loading ? (
        <p className="text-[11px] text-white/30">Loading...</p>
      ) : (
        <>
          {quotaMb > 0 && (
            <div>
              <div className="flex justify-between text-[10px] text-white/40 mb-1">
                <span>Storage</span>
                <span>
                  {mb(usedBytes)} MB of {quotaMb} MB
                </span>
              </div>
              <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                <div className="h-full bg-[#c8f06e]" style={{ width: `${pct}%` }} />
              </div>
            </div>
          )}

          <div>
            <p className="text-[10px] font-bold text-white/30 uppercase tracking-[0.2em] mb-2">
              My files ({files.length})
            </p>
            {files.length === 0 ? (
              <p className="text-[11px] text-white/30">No uploaded files.</p>
            ) : (
              <div className="space-y-2">
                {files.map((f) => (
                  <div
                    key={f.file_id}
                    className="flex items-center justify-between rounded-xl border border-white/5 bg-black/20 px-4 py-2.5"
                  >
                    <div className="min-w-0">
                      <p className="text-xs font-bold text-white truncate">{f.filename}</p>
                      <p className="text-[10px] text-white/40 font-mono">
                        {f.file_type} · {mb(f.size_bytes)} MB · {day(f.uploaded_at)}
                      </p>
                    </div>
                    <button
                      onClick={() => remove(`${base}/api/v1/files/${f.file_id}`, `"${f.filename}"`)}
                      className="ml-4 px-3 py-1.5 rounded-lg border border-red-400/30 text-red-300 text-[10px] uppercase tracking-widest hover:bg-red-400/10"
                    >
                      Delete
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div>
            <p className="text-[10px] font-bold text-white/30 uppercase tracking-[0.2em] mb-2">
              My jobs ({jobs.length})
            </p>
            {jobs.length === 0 ? (
              <p className="text-[11px] text-white/30">No jobs yet.</p>
            ) : (
              <div className="space-y-2">
                {jobs.map((j) => (
                  <div
                    key={j.job_id}
                    className="flex items-center justify-between rounded-xl border border-white/5 bg-black/20 px-4 py-2.5"
                  >
                    <div className="min-w-0">
                      <p className="text-xs font-bold text-white truncate">{j.filename}</p>
                      <p className="text-[10px] text-white/40 font-mono">
                        {j.goal} · {j.status} · {day(j.created_at)}
                      </p>
                    </div>
                    <button
                      onClick={() =>
                        remove(`${base}/api/v1/jobs/${j.job_id}/data`, `the job for "${j.filename}"`)
                      }
                      className="ml-4 px-3 py-1.5 rounded-lg border border-red-400/30 text-red-300 text-[10px] uppercase tracking-widest hover:bg-red-400/10"
                    >
                      Delete data
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}