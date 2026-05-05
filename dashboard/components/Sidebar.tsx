"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Cpu,
  LayoutDashboard,
  Map,
  ShieldCheck,
  ChevronRight,
  Layers,
  Radio,
  FlaskConical,
  Settings,
  Stethoscope,
  Waves,
  Wallet,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const primaryNav = [
  { icon: LayoutDashboard, label: "Overview", href: "/" },
  { icon: Wallet, label: "Trading", href: "/trading" },
  { icon: Map, label: "State Map", href: "/state-map" },
  { icon: Waves, label: "Locus Matrix", href: "/matrix" },
  { icon: BarChart3, label: "Analysis", href: "/analysis" },
  { icon: ShieldCheck, label: "Risk Center", href: "/risk" },
];

const secondaryNav: {
  icon: React.ElementType;
  label: string;
  href?: string;
}[] = [
  { icon: Layers, label: "Factors", href: "/analysis" },
  { icon: Radio, label: "Forecast", href: "/state-map" },
  { icon: Cpu, label: "ROEE", href: "/analysis" },
  { icon: Activity, label: "Chain", href: "/matrix" },
  { icon: FlaskConical, label: "Backtest Lab", href: "/risk" },
  { icon: Stethoscope, label: "Diagnostics" },
  { icon: Settings, label: "Settings" },
];

type DataAgePayload = {
  ibkrLastUpdated?: string | null;
  massiveLastUpdated?: string | null;
  lakeLastUpdated?: string | null;
  doctorLastUpdated?: string | null;
};

function formatAge(iso: string | null | undefined): string {
  if (!iso) return "—";
  const ts = new Date(iso).getTime();
  if (!Number.isFinite(ts)) return "—";

  const diffMs = Date.now() - ts;
  if (diffMs < 0) return "now";

  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m`;

  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h`;

  const days = Math.floor(hours / 24);
  return `${days}d`;
}

export default function Sidebar() {
  const pathname = usePathname();
  const [dataAge, setDataAge] = useState<DataAgePayload | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch("/api/metrics");
        const json = await res.json();
        setDataAge(json?.dataAge ?? null);
      } catch {
        setDataAge(null);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 15000);
    return () => clearInterval(interval);
  }, []);

  const dataRows = useMemo(
    () => [
      { label: "IBKR", value: formatAge(dataAge?.ibkrLastUpdated) },
      { label: "Massive", value: formatAge(dataAge?.massiveLastUpdated) },
      { label: "Lake", value: formatAge(dataAge?.lakeLastUpdated) },
      { label: "Doctor", value: formatAge(dataAge?.doctorLastUpdated) },
    ],
    [dataAge],
  );

  return (
    <aside className="w-[260px] shrink-0 glass border-r border-white/[0.06] flex flex-col h-full z-30">
      <div className="p-5 pb-4 border-b border-white/[0.05]">
        <div className="flex items-start gap-3">
          <div className="relative mt-0.5">
            <div className="absolute inset-0 rounded-xl bg-cyan-400/25 blur-lg scale-110" />
            <div className="relative w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400/80 to-violet-600/90 flex items-center justify-center shadow-[0_0_24px_rgba(34,211,238,0.35)] border border-white/10">
              <span className="text-[11px] font-black tracking-tighter text-[#041014] font-[family-name:var(--font-mono)]">
                RLM
              </span>
            </div>
          </div>
          <div className="min-w-0">
            <p className="font-bold text-[15px] leading-tight tracking-tight text-foreground">
              RLM
            </p>
            <p className="text-[11px] text-slate-500 leading-snug mt-0.5">
              Regime Locus Matrix
            </p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-4 overflow-y-auto">
        <div>
          <p className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-600">
            Navigate
          </p>
          <div className="space-y-0.5">
            {primaryNav.map((item) => {
              const active = pathname === item.href;
              return (
                <Link key={item.href} href={item.href}>
                  <div
                    className={cn(
                      "flex items-center justify-between px-3 py-2.5 rounded-xl transition-all duration-200 group",
                      active
                        ? "bg-violet-600/25 text-violet-200 border border-violet-500/25 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]"
                        : "text-slate-400 hover:text-foreground hover:bg-white/[0.04]",
                    )}
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <item.icon
                        className={cn(
                          "w-[18px] h-[18px] shrink-0",
                          active ? "text-violet-300" : "",
                        )}
                      />
                      <span className="font-medium text-[13px] truncate">{item.label}</span>
                    </div>
                    {active && <ChevronRight className="w-4 h-4 shrink-0 opacity-70" />}
                  </div>
                </Link>
              );
            })}
          </div>
        </div>

        <div>
          <p className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-600">
            Labs
          </p>
          <div className="space-y-0.5">
            {secondaryNav.map((item) => {
              const active = item.href ? pathname === item.href : false;
              const row = (
                <div
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-xl transition-colors",
                    item.href
                      ? active
                        ? "bg-cyan-500/10 text-cyan-200 border border-cyan-500/20"
                        : "text-slate-500 hover:text-slate-300 hover:bg-white/[0.04]"
                      : "text-slate-600 cursor-not-allowed opacity-70",
                  )}
                >
                  <item.icon
                    className={cn(
                      "w-[18px] h-[18px] shrink-0",
                      item.href && !active && "opacity-70",
                    )}
                  />
                  <span className="font-medium text-[13px] truncate">{item.label}</span>
                </div>
              );
              return item.href ? (
                <Link key={item.label} href={item.href}>
                  {row}
                </Link>
              ) : (
                <div key={item.label} title="Coming soon">
                  {row}
                </div>
              );
            })}
          </div>
        </div>
      </nav>

      <div className="p-4 border-t border-white/[0.05] space-y-4 mt-auto bg-black/25">
        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.06] px-3 py-2.5 flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-40" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
          </span>
          <span className="text-[12px] font-semibold text-emerald-400/95">
            All Systems Operational
          </span>
        </div>

        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600 mb-2 px-1">
            Data age
          </p>
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] overflow-hidden">
            <table className="w-full text-[11px] font-[family-name:var(--font-mono)]">
              <tbody>
                {dataRows.map((row) => (
                  <tr
                    key={row.label}
                    className="border-b border-white/[0.04] last:border-0"
                  >
                    <td className="px-3 py-2 text-slate-500">{row.label}</td>
                    <td className="px-3 py-2 text-right text-slate-300">{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="flex justify-between items-center text-[10px] text-slate-600 px-1 font-[family-name:var(--font-mono)]">
          <span>v2.4.0</span>
          <span>
            {new Date().toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
            })}
          </span>
        </div>
      </div>
    </aside>
  );
}
