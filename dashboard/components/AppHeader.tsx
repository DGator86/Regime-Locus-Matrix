"use client";

import React, { useState } from "react";
import { Bell, Radio, Settings, Wifi } from "lucide-react";
import { cn } from "@/lib/utils";

const MODES = ["Basic", "PRO", "Research"] as const;

export default function AppHeader() {
  const [symbol, setSymbol] = useState("SPY");
  const [timeframe, setTimeframe] = useState("1D");
  const [profile, setProfile] = useState("Default Pro");
  const [mode, setMode] = useState<(typeof MODES)[number]>("PRO");

  return (
    <header className="shrink-0 border-b border-white/[0.06] bg-[#080a0d]/80 backdrop-blur-xl px-4 py-3 z-20">
      <div className="flex flex-wrap items-center justify-between gap-3 max-w-[1920px] mx-auto">
        <div className="flex flex-wrap items-center gap-2 md:gap-3">
          <div className="hidden sm:flex items-center gap-2 pr-3 border-r border-white/10">
            <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400/90">
              Control
            </span>
            <Radio className="w-3.5 h-3.5 text-cyan-400/70" />
          </div>

          <label className="flex items-center gap-1.5 text-xs text-slate-500">
            <span className="hidden sm:inline">Symbol</span>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className={cn(
                "h-9 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 pr-8 text-sm text-foreground",
                "outline-none focus:ring-1 focus:ring-cyan-500/50 appearance-none cursor-pointer font-mono",
              )}
            >
              {["SPY", "QQQ", "IWM"].map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-1.5 text-xs text-slate-500">
            <span className="hidden sm:inline">Timeframe</span>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="h-9 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 text-sm outline-none focus:ring-1 focus:ring-cyan-500/50"
            >
              {["1D", "4H", "1H"].map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </label>

          <label className="hidden md:flex items-center gap-1.5 text-xs text-slate-500">
            Profile
            <select
              value={profile}
              onChange={(e) => setProfile(e.target.value)}
              className="h-9 min-w-[8.5rem] rounded-lg border border-white/10 bg-white/[0.04] px-2.5 text-sm outline-none focus:ring-1 focus:ring-cyan-500/50"
            >
              <option>Default Pro</option>
              <option>Conservative</option>
              <option>Research</option>
            </select>
          </label>

          <div
            className="flex rounded-lg border border-white/10 bg-black/40 p-0.5 ml-1"
            role="tablist"
            aria-label="Workspace mode"
          >
            {MODES.map((m) => (
              <button
                key={m}
                type="button"
                role="tab"
                aria-selected={mode === m}
                onClick={() => setMode(m)}
                className={cn(
                  "px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide rounded-md transition-all",
                  mode === m
                    ? "bg-gradient-to-b from-violet-600/90 to-violet-900/90 text-white shadow-[0_0_20px_rgba(139,92,246,0.35)]"
                    : "text-slate-500 hover:text-slate-300",
                )}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2 sm:gap-4">
          <span className="hidden lg:inline-flex items-center gap-2 text-[11px] text-slate-500 font-mono">
            <span className="text-emerald-400/90">IBKR</span>
            feed
          </span>
          <Wifi className="w-4 h-4 text-slate-500 hidden sm:block" />
          <button
            type="button"
            className="relative rounded-lg p-2 text-slate-400 hover:bg-white/[0.06] hover:text-foreground transition-colors"
            aria-label="Notifications"
          >
            <Bell className="w-4 h-4" />
            <span className="absolute top-1 right-1 flex h-4 min-w-[16px] items-center justify-center rounded-full bg-red-500 px-1 text-[10px] font-bold text-white">
              3
            </span>
          </button>
          <button
            type="button"
            className="rounded-lg p-2 text-slate-400 hover:bg-white/[0.06] hover:text-foreground transition-colors"
            aria-label="Settings"
          >
            <Settings className="w-4 h-4" />
          </button>
          <button
            type="button"
            className="flex h-9 w-9 items-center justify-center rounded-full border border-cyan-500/40 bg-gradient-to-br from-cyan-500/20 to-violet-600/30 text-xs font-bold text-cyan-200 shadow-[0_0_18px_rgba(34,211,238,0.25)]"
            aria-label="Account"
          >
            DV
          </button>
        </div>
      </div>
    </header>
  );
}
