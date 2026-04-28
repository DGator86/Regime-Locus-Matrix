"use client";

import React, { useEffect, useState, useMemo, useCallback } from "react";
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import { Activity, BarChart3, Grid3X3, TrendingUp, RefreshCcw } from "lucide-react";
import { cn } from "@/lib/utils";

interface ActivePlan {
  symbol: string;
  status: string;
  strategy: string;
  action: string;
  confidence: number;
  regimeConfidence: number;
  S_D: number;
  S_V: number;
  S_L: number;
  S_G: number;
  M_D: number;
  M_V: number;
  M_L: number;
  M_G: number;
  rankScore: number | null;
}

interface ForecastPoint {
  timestamp: string;
  close: number;
  S_D: number;
  S_V: number;
  S_L: number;
  S_G: number;
  hmm_state: number | null;
  hmm_confidence: number | null;
  hmm_state_label: string | null;
}

const FACTOR_COLORS = {
  S_D: "#00f5ff",
  S_V: "#a855f7",
  S_L: "#f59e0b",
  S_G: "#22c55e",
} as const;

const CHART_TOOLTIP_STYLE = {
  backgroundColor: "#111114",
  border: "1px solid #1e1e24",
  borderRadius: "12px",
};

function factorCellColor(val: number): string {
  if (val > 0.5) return "bg-green-500/70 text-white";
  if (val > 0.2) return "bg-green-500/30 text-green-300";
  if (val > -0.2) return "bg-white/5 text-muted-foreground";
  if (val > -0.5) return "bg-red-500/30 text-red-300";
  return "bg-red-500/70 text-white";
}

function SectionCard({
  title,
  icon: Icon,
  children,
  delay = 0,
  className = "",
}: {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className={cn("glass rounded-2xl p-6", className)}
    >
      <div className="flex items-center gap-2 mb-5">
        <Icon className="w-4 h-4 text-primary" />
        <h3 className="font-bold">{title}</h3>
      </div>
      {children}
    </motion.div>
  );
}

export default function AnalysisPage() {
  const [plans, setPlans] = useState<ActivePlan[]>([]);
  const [timeseries, setTimeseries] = useState<ForecastPoint[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();
      setPlans(data.activePlans || []);
      setTimeseries(data.forecastTimeseries || []);
    } catch (err) {
      console.error("Failed to fetch analysis data", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const sortedByRank = useMemo(
    () => [...plans].sort((a, b) => (b.rankScore ?? 0) - (a.rankScore ?? 0)),
    [plans],
  );

  const strategyGroups = useMemo(() => {
    const map = new Map<string, { count: number; totalConf: number }>();
    for (const p of plans) {
      const key = p.strategy || "Unknown";
      const entry = map.get(key) || { count: 0, totalConf: 0 };
      entry.count++;
      entry.totalConf += p.confidence;
      map.set(key, entry);
    }
    return Array.from(map.entries())
      .map(([name, v]) => ({ name, count: v.count, avgConf: v.totalConf / v.count }))
      .sort((a, b) => b.count - a.count);
  }, [plans]);

  const confidenceData = useMemo(
    () =>
      sortedByRank.map((p) => ({
        symbol: p.symbol,
        confidence: p.regimeConfidence,
        color: p.action === "enter" ? "#22c55e" : "#64748b",
      })),
    [sortedByRank],
  );

  const chartTimeseries = useMemo(
    () =>
      timeseries.map((t) => ({
        ...t,
        label: t.timestamp.length > 10 ? t.timestamp.slice(5, 10) : t.timestamp,
      })),
    [timeseries],
  );

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Analysis Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Factor decomposition, signal distribution, and universe analytics.
          </p>
        </div>
        <button
          onClick={() => {
            setLoading(true);
            fetchData();
          }}
          className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-2 rounded-xl font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
          disabled={loading}
        >
          <RefreshCcw className={cn("w-4 h-4", loading && "animate-spin")} />
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </header>

      {/* ─── Factor Timeline ─── */}
      <SectionCard title="Factor Timeline" icon={Activity} delay={0.05}>
        <div className="h-[300px] w-full">
          {chartTimeseries.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartTimeseries}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e1e24" />
                <XAxis
                  dataKey="label"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  dy={8}
                  interval="preserveStartEnd"
                />
                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  domain={[-1, 1]}
                />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Line type="monotone" dataKey="S_D" stroke={FACTOR_COLORS.S_D} strokeWidth={2} dot={false} name="Direction" />
                <Line type="monotone" dataKey="S_V" stroke={FACTOR_COLORS.S_V} strokeWidth={2} dot={false} name="Volatility" />
                <Line type="monotone" dataKey="S_L" stroke={FACTOR_COLORS.S_L} strokeWidth={2} dot={false} name="Liquidity" />
                <Line type="monotone" dataKey="S_G" stroke={FACTOR_COLORS.S_G} strokeWidth={2} dot={false} name="Gamma" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
              No forecast timeseries data available
            </div>
          )}
        </div>
        <div className="flex gap-6 mt-4">
          {Object.entries(FACTOR_COLORS).map(([key, color]) => (
            <span key={key} className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className="w-3 h-0.5 rounded-full" style={{ backgroundColor: color }} />
              {key}
            </span>
          ))}
        </div>
      </SectionCard>

      {/* ─── Middle Row ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Signal Distribution */}
        <SectionCard title="Signal Distribution" icon={BarChart3} delay={0.1}>
          <div className="space-y-2 max-h-[340px] overflow-y-auto pr-2">
            {sortedByRank.map((p) => {
              const pct = Math.max(0, Math.min(100, p.regimeConfidence * 100));
              return (
                <div key={p.symbol} className="flex items-center gap-3">
                  <span className="w-12 text-xs font-mono font-bold shrink-0">{p.symbol}</span>
                  <div className="flex-1 h-5 bg-secondary rounded-full overflow-hidden relative">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className={cn(
                        "h-full rounded-full",
                        p.action === "enter" ? "bg-green-500/70" : "bg-white/10",
                      )}
                    />
                    <span className="absolute inset-0 flex items-center justify-end pr-2 text-[10px] font-mono text-muted-foreground">
                      {pct.toFixed(0)}%
                    </span>
                  </div>
                  <span
                    className={cn(
                      "text-[10px] font-bold uppercase px-1.5 py-0.5 rounded",
                      p.action === "enter"
                        ? "bg-green-500/20 text-green-400"
                        : "bg-white/5 text-muted-foreground",
                    )}
                  >
                    {p.action || "—"}
                  </span>
                </div>
              );
            })}
            {plans.length === 0 && (
              <p className="text-sm text-muted-foreground py-4 text-center">No active plans</p>
            )}
          </div>
        </SectionCard>

        {/* Strategy Breakdown */}
        <SectionCard title="Strategy Breakdown" icon={TrendingUp} delay={0.15}>
          {strategyGroups.length > 0 ? (
            <div className="space-y-4">
              {strategyGroups.map((sg) => (
                <div key={sg.name} className="p-3 bg-secondary/30 rounded-xl border border-border">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-sm font-bold">{sg.name}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {sg.count} plan{sg.count !== 1 ? "s" : ""}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-mono font-bold neon-text">
                        {(sg.avgConf * 100).toFixed(0)}%
                      </p>
                      <p className="text-[10px] text-muted-foreground">avg conf</p>
                    </div>
                  </div>
                  <div className="mt-2 h-1.5 bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-accent rounded-full"
                      style={{ width: `${sg.avgConf * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground py-4 text-center">No strategies found</p>
          )}
        </SectionCard>
      </div>

      {/* ─── Bottom Row ─── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Universe Heat Map */}
        <SectionCard title="Universe Heat Map" icon={Grid3X3} delay={0.2}>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 pr-3 font-bold text-muted-foreground">Symbol</th>
                  <th className="text-center py-2 px-2 font-bold" style={{ color: FACTOR_COLORS.S_D }}>
                    S_D
                  </th>
                  <th className="text-center py-2 px-2 font-bold" style={{ color: FACTOR_COLORS.S_V }}>
                    S_V
                  </th>
                  <th className="text-center py-2 px-2 font-bold" style={{ color: FACTOR_COLORS.S_L }}>
                    S_L
                  </th>
                  <th className="text-center py-2 px-2 font-bold" style={{ color: FACTOR_COLORS.S_G }}>
                    S_G
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedByRank.map((p) => (
                  <tr key={p.symbol} className="border-b border-border/50 hover:bg-white/[0.02]">
                    <td className="py-2 pr-3 font-mono font-bold">{p.symbol}</td>
                    {(["S_D", "S_V", "S_L", "S_G"] as const).map((f) => (
                      <td key={f} className="py-1.5 px-1">
                        <span
                          className={cn(
                            "block text-center rounded-md px-2 py-1 font-mono font-bold text-[11px]",
                            factorCellColor(p[f]),
                          )}
                        >
                          {p[f] >= 0 ? "+" : ""}
                          {p[f].toFixed(2)}
                        </span>
                      </td>
                    ))}
                  </tr>
                ))}
                {plans.length === 0 && (
                  <tr>
                    <td colSpan={5} className="py-6 text-center text-muted-foreground">
                      No data
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </SectionCard>

        {/* Confidence Distribution */}
        <SectionCard title="Confidence Distribution" icon={BarChart3} delay={0.25}>
          <div className="h-[280px] w-full">
            {confidenceData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confidenceData} layout="vertical" barCategoryGap={4}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#1e1e24" />
                  <XAxis
                    type="number"
                    domain={[0, 1]}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  />
                  <YAxis
                    type="category"
                    dataKey="symbol"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    width={48}
                  />
                  <Tooltip
                    contentStyle={CHART_TOOLTIP_STYLE}
                    formatter={(val: number) => [`${(val * 100).toFixed(1)}%`, "Confidence"]}
                  />
                  <Bar dataKey="confidence" radius={[0, 4, 4, 0]}>
                    {confidenceData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                No confidence data
              </div>
            )}
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
