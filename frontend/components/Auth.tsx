"use client";

import React, { useState } from "react";
import axios from "axios";
import { Lock, User, Mail, Loader2, ShieldCheck, AlertCircle } from "lucide-react";

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
}

interface AuthProps {
  onLogin: (tokens: AuthTokens) => void;
  apiUrl: string;
}

export default function Auth({ onLogin, apiUrl }: AuthProps) {
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
  });

  const applyTokens = (data: { access_token: string; refresh_token: string }) => {
    onLogin({
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (isLogin) {
        const res = await axios.post(`${apiUrl}/api/v1/auth/login`, {
          username: formData.email || formData.username,
          password: formData.password,
        });
        applyTokens(res.data);
      } else {
        await axios.post(`${apiUrl}/api/v1/auth/register`, {
          username: formData.username,
          email: formData.email,
          password: formData.password,
        });
        // Auto-login after register so the user lands in the app immediately
        const res = await axios.post(`${apiUrl}/api/v1/auth/login`, {
          username: formData.email,
          password: formData.password,
        });
        setSuccess(true);
        applyTokens(res.data);
      }
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const message =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d: any) => d.msg || JSON.stringify(d)).join(", ")
            : "Authentication failed. Please try again.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const startGoogleLogin = () => {
    const width = 500;
    const height = 600;
    const left = window.screenX + (window.outerWidth - width) / 2;
    const top = window.screenY + (window.outerHeight - height) / 2;
    const origin = encodeURIComponent(window.location.origin);

    window.open(
      `${apiUrl}/api/v1/auth/google/login?frontend_origin=${origin}`,
      "google_login",
      `width=${width},height=${height},left=${left},top=${top}`
    );

    const handleMessage = async (event: MessageEvent) => {
      if (event.origin !== apiUrl.replace(/\/$/, "")) return;
      if (event.data?.type !== "nydra_oauth" || !event.data?.code) return;

      window.removeEventListener("message", handleMessage);
      setLoading(true);
      setError(null);
      try {
        const res = await axios.post(`${apiUrl}/api/v1/auth/google/exchange`, {
          code: event.data.code,
        });
        applyTokens(res.data);
      } catch (err: any) {
        setError(err.response?.data?.detail || "Google sign-in failed.");
      } finally {
        setLoading(false);
      }
    };
    window.addEventListener("message", handleMessage);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-md px-4">
      <div className="w-full max-w-md overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl">
        <div className="bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 p-8 text-center border-b border-slate-800/50">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            <ShieldCheck size={28} />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white font-syne uppercase">
            Nydra Intelligence
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            {isLogin ? "Sign in with email or Google" : "Create your account"}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="p-8 space-y-5">
          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-red-500/10 p-3 text-sm text-red-400 border border-red-500/20">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {success && (
            <div className="flex items-center gap-2 rounded-lg bg-emerald-500/10 p-3 text-sm text-emerald-400 border border-emerald-500/20">
              <ShieldCheck size={16} />
              <span>Account created — signing you in...</span>
            </div>
          )}

          <div className="space-y-4">
            <button
              type="button"
              onClick={startGoogleLogin}
              className="flex w-full items-center justify-center gap-3 rounded-xl border border-slate-700 bg-slate-800/50 py-2.5 text-sm font-medium text-white transition-all hover:bg-slate-800 hover:border-slate-600 active:scale-[0.98]"
            >
              <svg className="h-5 w-5" viewBox="0 0 24 24">
                <path
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-3.22 3.28-7.96 3.28-13.09z"
                  fill="#4285F4"
                />
                <path
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  fill="#34A853"
                />
                <path
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.16H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.84l3.66-2.75z"
                  fill="#FBBC05"
                />
                <path
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.16l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  fill="#EA4335"
                />
              </svg>
              <span>Continue with Google</span>
            </button>

            <div className="relative flex items-center py-2">
              <div className="flex-grow border-t border-slate-800"></div>
              <span className="mx-4 flex-shrink text-[10px] font-mono text-slate-600 uppercase tracking-widest">
                or use email
              </span>
              <div className="flex-grow border-t border-slate-800"></div>
            </div>

            {isLogin ? (
              <div className="relative group">
                <Mail className="absolute left-3 top-3 h-5 w-5 text-slate-500 transition-colors group-focus-within:text-emerald-400" />
                <input
                  type="text"
                  placeholder="Email or username"
                  required
                  autoComplete="username"
                  className="w-full rounded-xl border border-slate-700 bg-slate-800 py-2.5 pl-11 pr-4 text-white placeholder-slate-500 outline-none ring-offset-slate-900 focus:border-emerald-500/50 focus:ring-2 focus:ring-emerald-500/20 transition-all"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value, username: e.target.value })}
                />
              </div>
            ) : (
              <>
                <div className="relative group">
                  <User className="absolute left-3 top-3 h-5 w-5 text-slate-500 transition-colors group-focus-within:text-emerald-400" />
                  <input
                    type="text"
                    placeholder="Username"
                    required
                    autoComplete="username"
                    className="w-full rounded-xl border border-slate-700 bg-slate-800 py-2.5 pl-11 pr-4 text-white placeholder-slate-500 outline-none ring-offset-slate-900 focus:border-emerald-500/50 focus:ring-2 focus:ring-emerald-500/20 transition-all"
                    value={formData.username}
                    onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                  />
                </div>
                <div className="relative group">
                  <Mail className="absolute left-3 top-3 h-5 w-5 text-slate-500 transition-colors group-focus-within:text-emerald-400" />
                  <input
                    type="email"
                    placeholder="Email Address"
                    required
                    autoComplete="email"
                    className="w-full rounded-xl border border-slate-700 bg-slate-800 py-2.5 pl-11 pr-4 text-white placeholder-slate-500 outline-none focus:border-emerald-500/50 focus:ring-2 focus:ring-emerald-500/20 transition-all"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  />
                </div>
              </>
            )}

            <div className="relative group">
              <Lock className="absolute left-3 top-3 h-5 w-5 text-slate-500 transition-colors group-focus-within:text-emerald-400" />
              <input
                type="password"
                placeholder="Password"
                required
                autoComplete={isLogin ? "current-password" : "new-password"}
                className="w-full rounded-xl border border-slate-700 bg-slate-800 py-2.5 pl-11 pr-4 text-white placeholder-slate-500 outline-none focus:border-emerald-500/50 focus:ring-2 focus:ring-emerald-500/20 transition-all"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="group relative w-full overflow-hidden rounded-xl bg-gradient-to-r from-emerald-600 to-emerald-500 py-3 font-semibold text-white shadow-lg transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-70"
          >
            <div className="flex items-center justify-center gap-2">
              {loading ? <Loader2 className="animate-spin" size={20} /> : isLogin ? "Sign In" : "Create Account"}
            </div>
            <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/10 to-transparent transition-transform duration-1000 group-hover:translate-x-full" />
          </button>

          <div className="text-center">
            <button
              type="button"
              className="text-sm text-slate-500 hover:text-emerald-400 transition-colors"
              onClick={() => {
                setIsLogin(!isLogin);
                setError(null);
                setSuccess(false);
              }}
            >
              {isLogin ? "Don't have an account? Create one" : "Already have an account? Sign in"}
            </button>
          </div>
        </form>

        <div className="bg-slate-950 p-4 text-center">
          <p className="text-[10px] tracking-widest text-slate-600 uppercase font-mono">
            E2E Encrypted · PBKDF2 Hashed · Vault Protected
          </p>
        </div>
      </div>
    </div>
  );
}
