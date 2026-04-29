"use client";

import React from "react";
import { RefreshCcw } from "lucide-react";
import { cn } from "@/lib/utils";

export function PageHero({
  eyebrow = "RLM · Dashboard",
  title,
  subtitle,
  className,
  action,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  className?: string;
  action?: React.ReactNode;
}) {
  return (
    <header
      className={cn(
        "flex flex-col lg:flex-row lg:items-end justify-between gap-4 pb-5 mb-2 border-b border-white/[0.06]",
        className,
      )}
    >
      <div className="min-w-0">
        <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-cyan-400/85 neon-text-soft mb-1">
          {eyebrow}
        </p>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">{title}</h1>
        {subtitle && (
          <p className="text-slate-500 mt-1.5 text-sm max-w-3xl leading-relaxed">{subtitle}</p>
        )}
      </div>
      {action ? (
        <div className="shrink-0 flex flex-wrap items-center gap-3">{action}</div>
      ) : null}
    </header>
  );
}

export function HudRefreshButton({
  loading,
  onClick,
  label = "Refresh",
}: {
  loading?: boolean;
  onClick: () => void;
  label?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={loading}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-xl border border-cyan-500/35",
        "bg-gradient-to-b from-cyan-500/12 to-transparent px-4 py-2.5 text-sm font-semibold text-cyan-100",
        "shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_0_28px_rgba(34,211,238,0.06)]",
        "hover:from-cyan-500/20 transition-colors disabled:opacity-45 font-[family-name:var(--font-mono)]",
      )}
    >
      <RefreshCcw className={cn("w-4 h-4 shrink-0", loading && "animate-spin")} />
      {loading ? "Refreshing…" : label}
    </button>
  );
}

/** Tooltip chrome aligned with Overview charts */
export const HUD_CHART_TOOLTIP = {
  backgroundColor: "#0d1118",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: "12px",
  fontSize: 12,
} as const;
