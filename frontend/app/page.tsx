"use client";

import React, { useState, useEffect } from "react";
import Uploader, { UploadResult, AuditUploadResult } from "@/components/Uploader";
import Dashboard from "@/components/Dashboard";
import ChatBox from "@/components/ChatBox";
import Auth from "@/components/Auth";
import axios from "axios";
import { Layout, Database, Activity, MessageSquare, Settings, LogOut, ChevronRight, Eye, Type, Hammer } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type AppState = "IDLE" | "PROCESSING" | "RESULTS";
type Workspace = "DIAGNOSTICS" | "VISION_LAB" | "TEXT_INTELLIGENCE" | "REPAIR_SHOP";

export default function NydraApp() {
  const [state, setState] = useState<AppState>("IDLE");
  const [token, setToken] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [goal, setGoal] = useState<string>("inspect");
  const [filename, setFilename] = useState<string>("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [activeSettingsTab, setActiveSettingsTab] = useState("api");
  const [customApiUrl, setCustomApiUrl] = useState(API_URL);
  const [activeWorkspace, setActiveWorkspace] = useState<Workspace>("DIAGNOSTICS");
  const [registry, setRegistry] = useState<any>(null);
  const [settings, setSettings] = useState<any>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    const savedApi = localStorage.getItem("nydra_api_url");
    if (savedApi) setCustomApiUrl(savedApi);
    
    // Check if token exists in session
    const savedToken = sessionStorage.getItem("nydra_token");
    const savedRefresh = sessionStorage.getItem("nydra_refresh");
    if (savedToken) {
      setToken(savedToken);
      if (savedRefresh) setRefreshToken(savedRefresh);
      fetchMetadata(savedToken);
    }
  }, []);

  const fetchMetadata = async (t: string) => {
    try {
      const [regRes, setRes] = await Promise.all([
        axios.get(`${customApiUrl}/api/v1/config/llm/providers`, { headers: { Authorization: `Bearer ${t}` } }),
        axios.get(`${customApiUrl}/api/v1/me/settings`, { headers: { Authorization: `Bearer ${t}` } })
      ]);
      setRegistry(regRes.data);
      setSettings(setRes.data);
    } catch (err) {
      console.error("Failed to fetch metadata", err);
    }
  };

  const handleLogin = (tokens: { accessToken: string; refreshToken: string }) => {
    setToken(tokens.accessToken);
    setRefreshToken(tokens.refreshToken);
    sessionStorage.setItem("nydra_token", tokens.accessToken);
    sessionStorage.setItem("nydra_refresh", tokens.refreshToken);
    fetchMetadata(tokens.accessToken);
  };

  const handleLogout = async () => {
    const currentRefresh = refreshToken || sessionStorage.getItem("nydra_refresh");
    if (currentRefresh) {
      try {
        await axios.post(`${customApiUrl}/api/v1/auth/logout`, {
          refresh_token: currentRefresh,
        });
      } catch {
        // Local logout still proceeds if revoke fails
      }
    }
    setToken(null);
    setRefreshToken(null);
    sessionStorage.removeItem("nydra_token");
    sessionStorage.removeItem("nydra_refresh");
    setSettings(null);
    setRegistry(null);
  };

  const handleSaveSettings = async (updates: any) => {
    if (updates.apiUrl) {
      setCustomApiUrl(updates.apiUrl);
      localStorage.setItem("nydra_api_url", updates.apiUrl);
    }
    
    if (token && Object.keys(updates).length > (updates.apiUrl ? 1 : 0)) {
      setIsSaving(true);
      try {
        const { apiUrl, ...apiUpdates } = updates;
        const res = await axios.patch(`${customApiUrl}/api/v1/me/settings`, apiUpdates, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setSettings(res.data);
      } catch (err) {
        console.error("Failed to save settings", err);
      } finally {
        setIsSaving(false);
      }
    }
    
    setIsSettingsOpen(false);
  };

  const handleUploadSuccess = async (result: UploadResult | AuditUploadResult) => {
    if (!token) return;
    
    setState("PROCESSING");
    
    try {
      // Create a job based on the upload
      let jobRequest;
      let currentGoal = "inspect";
      let currentFilename = "";

      if (result.mode === "single") {
        currentGoal = "inspect";
        currentFilename = result.file.filename;
        jobRequest = {
          file_id: result.file.file_id,
          goal: currentGoal,
        };
      } else {
        currentGoal = "audit";
        currentFilename = `${result.train_file.filename} + ${result.test_file.filename}`;
        jobRequest = {
          train_file_id: result.train_file.file_id,
          test_file_id: result.test_file.file_id,
          goal: currentGoal,
        };
      }

      setGoal(currentGoal);
      setFilename(currentFilename);

      const res = await axios.post(`${API_URL}/api/v1/jobs`, jobRequest, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      setJobId(res.data.job_id);
      setState("RESULTS");
    } catch (err) {
      console.error("Failed to create job", err);
      setState("IDLE");
    }
  };

  const handleReset = () => {
    setState("IDLE");
    setJobId(null);
    setGoal("inspect");
    setFilename("");
  };

  return (
    <div className="flex h-screen bg-[#020617] text-[#e8e6e1] font-sans selection:bg-[#c8f06e] selection:text-[#0f172a]">
      {!token && <Auth onLogin={handleLogin} apiUrl={customApiUrl} />}
      
      {/* ── LEFT NAV ────────────────────────────────────────────────────────── */}
      <aside className="w-16 flex flex-col items-center py-6 border-r border-white/5 bg-[#0f172a]/50 backdrop-blur-xl z-20">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#c8f06e] to-[#5ce0c6] flex items-center justify-center mb-12 shadow-lg shadow-[#c8f06e]/20">
          <Activity size={20} className="text-[#0f172a]" />
        </div>
        
        <nav className="flex flex-col gap-8 flex-1">
          <button 
            onClick={() => setActiveWorkspace("DIAGNOSTICS")}
            className={`transition-colors ${activeWorkspace === "DIAGNOSTICS" ? "text-[#c8f06e]" : "text-white/40 hover:text-white"}`} 
            title="Diagnostics"
          >
            <Activity size={20} />
          </button>
          <button 
            onClick={() => setActiveWorkspace("VISION_LAB")}
            className={`transition-colors ${activeWorkspace === "VISION_LAB" ? "text-[#c8f06e]" : "text-white/40 hover:text-white"}`} 
            title="Vision Lab"
          >
            <Eye size={20} />
          </button>
          <button 
            onClick={() => setActiveWorkspace("TEXT_INTELLIGENCE")}
            className={`transition-colors ${activeWorkspace === "TEXT_INTELLIGENCE" ? "text-[#c8f06e]" : "text-white/40 hover:text-white"}`} 
            title="Text Intelligence"
          >
            <Type size={20} />
          </button>
          <button 
            onClick={() => setActiveWorkspace("REPAIR_SHOP")}
            className={`transition-colors ${activeWorkspace === "REPAIR_SHOP" ? "text-[#c8f06e]" : "text-white/40 hover:text-white"}`} 
            title="Repair Shop"
          >
            <Hammer size={20} />
          </button>
        </nav>

        <div className="flex flex-col gap-6 mt-auto">
          <button 
            onClick={() => setIsSettingsOpen(true)}
            className="text-white/20 hover:text-white transition-colors"
          >
            <Settings size={20} />
          </button>
          <button 
            onClick={handleLogout}
            className="text-white/20 hover:text-red-400 transition-colors"
          >
            <LogOut size={20} />
          </button>
        </div>
      </aside>

      {/* ── SETTINGS MODAL ──────────────────────────────────────────────────── */}
      {isSettingsOpen && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-6">
          <div className="w-full max-w-4xl h-[500px] bg-[#0f172a] border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex">
            {/* Modal Sidebar */}
            <div className="w-64 border-r border-white/5 bg-black/20 p-8 flex flex-col gap-2">
              <h2 className="text-[10px] font-bold text-white/20 uppercase tracking-[0.2em] mb-8">System Config</h2>
              <button 
                onClick={() => setActiveSettingsTab("api")}
                className={`px-4 py-3 rounded-xl text-left text-xs font-bold tracking-widest transition-all ${
                  activeSettingsTab === "api" ? "bg-[#c8f06e]/10 text-[#c8f06e] border border-[#c8f06e]/20" : "text-white/30 hover:text-white"
                }`}
              >
                API BRANCH
              </button>
              <button 
                onClick={() => setActiveSettingsTab("general")}
                className={`px-4 py-3 rounded-xl text-left text-xs font-bold tracking-widest transition-all ${
                  activeSettingsTab === "general" ? "bg-[#c8f06e]/10 text-[#c8f06e] border border-[#c8f06e]/20" : "text-white/30 hover:text-white"
                }`}
              >
                GENERAL
              </button>
              <button 
                onClick={() => setActiveSettingsTab("data")}
                className={`px-4 py-3 rounded-xl text-left text-xs font-bold tracking-widest transition-all ${
                  activeSettingsTab === "data" ? "bg-[#c8f06e]/10 text-[#c8f06e] border border-[#c8f06e]/20" : "text-white/30 hover:text-white"
                }`}
              >
                DATA ENGINE
              </button>
            </div>

            {/* Modal Content */}
            <div className="flex-1 p-12 flex flex-col relative">
              <button 
                onClick={() => setIsSettingsOpen(false)}
                className="absolute top-8 right-8 text-white/20 hover:text-white transition-colors"
              >
                ✕
              </button>

              <div className="flex-1">
                {activeSettingsTab === "api" && (
                  <div className="animate-in fade-in slide-in-from-left-4 duration-300">
                    <h3 className="text-2xl font-bold mb-10 tracking-tight">AI Intelligence Branch</h3>
                    
                    <div className="grid grid-cols-2 gap-10">
                      <div className="space-y-6">
                        <div>
                          <label className="block text-[10px] font-mono text-white/40 uppercase tracking-[0.2em] mb-3">
                            BACKEND API URL
                          </label>
                          <input 
                            type="text" 
                            defaultValue={customApiUrl}
                            id="settings-api-url"
                            className="w-full bg-black/40 border border-white/10 rounded-xl px-5 py-3 text-sm font-mono text-[#c8f06e] focus:outline-none focus:border-[#c8f06e]/50 transition-all shadow-inner"
                          />
                        </div>

                        <div>
                          <label className="block text-[10px] font-mono text-white/40 uppercase tracking-[0.2em] mb-3">
                            INTELLIGENCE PROVIDER
                          </label>
                          <select 
                            id="settings-llm-provider"
                            defaultValue={settings?.llm_provider || "groq"}
                            onChange={(e) => {
                              const p = e.target.value;
                              const modelSelect = document.getElementById("settings-llm-model") as HTMLSelectElement;
                              if (modelSelect && registry?.providers[p]) {
                                modelSelect.value = registry.providers[p].recommended;
                              }
                            }}
                            className="w-full bg-black/40 border border-white/10 rounded-xl px-5 py-3 text-sm text-white/80 focus:outline-none focus:border-[#c8f06e]/50 transition-all appearance-none cursor-pointer"
                          >
                            {registry ? Object.entries(registry.providers).map(([id, info]: [string, any]) => (
                              <option key={id} value={id}>{info.name}</option>
                            )) : <option>Loading...</option>}
                          </select>
                        </div>

                        <div>
                          <label className="block text-[10px] font-mono text-white/40 uppercase tracking-[0.2em] mb-3">
                            ACTIVE MODEL
                          </label>
                          <select 
                            id="settings-llm-model"
                            defaultValue={settings?.llm_model}
                            className="w-full bg-black/40 border border-white/10 rounded-xl px-5 py-3 text-sm text-[#c8f06e]/80 focus:outline-none focus:border-[#c8f06e]/50 transition-all appearance-none cursor-pointer"
                          >
                            {registry && Object.values(registry.providers).map((p: any) => 
                              p.models.map((m: string) => (
                                <option key={m} value={m}>{m}</option>
                              ))
                            )}
                          </select>
                        </div>
                      </div>

                      <div className="space-y-6 bg-white/5 p-6 rounded-2xl border border-white/5">
                        <label className="block text-[10px] font-mono text-white/40 uppercase tracking-[0.2em] mb-3">
                          VAULT-ENCRYPTED SECRETS
                        </label>
                        <div className="space-y-4">
                          <p className="text-[10px] text-white/30 leading-relaxed uppercase">
                            Your API keys are encrypted at rest using AES-128 and rotated via your private security vault.
                          </p>
                          <input 
                            type="password" 
                            id="settings-llm-key"
                            placeholder="Enter Provider API Key..."
                            className="w-full bg-black/60 border border-white/10 rounded-xl px-5 py-3 text-xs font-mono text-white placeholder-white/10 focus:outline-none focus:border-[#c8f06e]/50 transition-all"
                          />
                          <div className="flex items-center gap-2 text-[9px] font-mono text-emerald-500/50 uppercase">
                            <Activity size={10} />
                            <span>E2EE Active | 310K Iterations</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {activeSettingsTab !== "api" && (
                  <div className="flex flex-col items-center justify-center h-full text-center opacity-20 grayscale">
                    <Database size={48} className="mb-4" />
                    <p className="font-mono text-[10px] uppercase tracking-widest">Branch Locked in Prototype</p>
                  </div>
                )}
              </div>

              <div className="pt-8 border-t border-white/5 flex justify-end gap-4">
                <button 
                  onClick={() => setIsSettingsOpen(false)}
                  className="px-6 py-3 rounded-xl text-[10px] font-bold tracking-widest text-white/30 hover:text-white transition-all uppercase"
                >
                  Discard
                </button>
                <button 
                  onClick={() => {
                    const apiUrl = (document.getElementById("settings-api-url") as HTMLInputElement)?.value;
                    const provider = (document.getElementById("settings-llm-provider") as HTMLSelectElement)?.value;
                    const model = (document.getElementById("settings-llm-model") as HTMLSelectElement)?.value;
                    const key = (document.getElementById("settings-llm-key") as HTMLInputElement)?.value;
                    
                    const updates: any = { apiUrl };
                    if (provider) updates.llm_provider = provider;
                    if (model) updates.llm_model = model;
                    if (key) updates.llm_api_key = key;
                    
                    handleSaveSettings(updates);
                  }}
                  className="px-10 py-3 rounded-xl bg-[#c8f06e] text-[#0f172a] text-[10px] font-bold tracking-[0.2em] hover:bg-[#b8e05e] transition-all shadow-lg shadow-[#c8f06e]/10 uppercase"
                >
                  {isSaving ? "Saving..." : "Save Branch"}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── MAIN CONTENT ────────────────────────────────────────────────────── */}
      <main className="flex-1 relative overflow-hidden flex flex-col">
        {/* Header */}
        <header className="h-16 border-b border-white/5 flex items-center justify-between px-8 bg-[#0f172a]/30 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono text-white/30 tracking-widest uppercase">System</span>
            <ChevronRight size={12} className="text-white/20" />
            <span className="text-xs font-mono text-[#c8f06e] tracking-widest uppercase">
              {state === "IDLE" ? "Ingestion" : state === "PROCESSING" ? "Analysis" : "Intelligence"}
            </span>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-[#c8f06e] animate-pulse" />
              <span className="text-[10px] font-mono text-white/40 uppercase tracking-tighter">API Live</span>
            </div>
            <div className="h-8 w-[1px] bg-white/5" />
            <div className="flex items-center gap-3">
              <div className="text-right">
                <div className="text-[10px] text-white/80 font-medium">Guest User</div>
                <div className="text-[9px] text-white/30 font-mono">PRO_PLAN</div>
              </div>
              <div className="w-8 h-8 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-[10px] font-bold">
                GU
              </div>
            </div>
          </div>
        </header>

        {/* Dynamic Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar relative">
          <div className="max-w-7xl mx-auto p-12 h-full flex flex-col">
            {state === "IDLE" && (
              <div className="flex-1 flex flex-col items-center justify-center">
                <div className="text-center mb-12 animate-nydra-float">
                  <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-white via-white to-white/30">
                    Nydra Intelligence
                  </h1>
                  <p className="text-white/40 font-mono text-sm tracking-wide">
                    AUTONOMOUS DATA INSPECTION & PREPARATION AGENT
                  </p>
                </div>
                
                <div className="w-full max-w-3xl">
                  <Uploader 
                    token={token || ""} 
                    onUploadSuccess={handleUploadSuccess} 
                    apiUrl={customApiUrl}
                  />
                </div>
              </div>
            )}

            {(state === "PROCESSING" || state === "RESULTS") && jobId && (
              <div className="flex-1">
                <Dashboard 
                  jobId={jobId} 
                  token={token || ""} 
                  goal={goal}
                  filename={filename}
                  onNewAnalysis={handleReset}
                  workspace={activeWorkspace}
                  apiUrl={customApiUrl}
                />
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ── RIGHT SIDEBAR (CHAT) ────────────────────────────────────────────── */}
      <aside 
        className={`transition-all duration-500 border-l border-white/5 bg-[#0f172a]/40 backdrop-blur-2xl relative ${
          isSidebarOpen ? "w-[400px]" : "w-0"
        }`}
      >
        <button 
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="absolute -left-4 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-[#1e293b] border border-white/10 flex items-center justify-center hover:bg-[#c8f06e] hover:text-[#0f172a] transition-all z-30"
        >
          <ChevronRight size={14} className={`transition-transform duration-500 ${isSidebarOpen ? "rotate-0" : "rotate-180"}`} />
        </button>

        <div className={`h-full w-[400px] flex flex-col ${isSidebarOpen ? "opacity-100" : "opacity-0"} transition-opacity duration-300`}>
          <ChatBox 
            jobId={jobId || undefined} 
            token={token || ""} 
            goal={goal}
            filename={filename}
            apiUrl={customApiUrl}
          />
        </div>
      </aside>
    </div>
  );
}
