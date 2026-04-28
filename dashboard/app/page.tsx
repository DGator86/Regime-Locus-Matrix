"use client";

import React, { useEffect, useState, useCallback, useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
  ComposedChart,
} from "recharts";
import RegimeHeatmap from "@/components/RegimeHeatmap";
import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  Crosshair,
  Gauge,
  Layers,
  Radio,
  Sigma,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { displayHmmState } from "@/lib/hmmDisplay";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ActivePlan {
  symbol: string;
  status: string;
  strategy: string;
  action: string;
  rationale: string;
  confidence: number;
  regimeConfidence: number;
  regimeKey: string;
  close: number;
  sigma: number;
  lower1s: number;
  upper1s: number;
  lower2s: number;
  upper2s: number;
  S_D: number;
  S_V: number;
  S_L: number;
  S_G: number;
  sizeFraction: number;
  vaultTriggered: boolean;
  forecastUncertainty: number;
  hmmConfidence: number;
  hmmTradeAllowed: boolean;
  hmmState?: string;
  M_regime?: string;
  M_D?: number;
  M_V?: number;
  M_L?: number;
  M_G?: number;
  M_trend_strength?: number;
  M_R_trans?: number;
  coordinateStrategy?: string;
  legs?: string;
  entryDebit?: number;
  thresholds?: Record<string, number>;
  planId?: string;
  rankScore?: number;
  runAt?: string;
}

interface TradeSummary {
  totalTrades: number;
  openTrades: number;
  closedTrades: number;
  realizedPnl: number;
  unrealizedPnl: number;
  winRate: number;
  wins?: number;
  losses?: number;
}

interface ForecastPoint {
  timestamp: string;
  close: number;
  S_D?: number;
  S_V?: number;
  S_L?: number;
  S_G?: number;
  mean_price?: number;
  lower_1s?: number;
  upper_1s?: number;
  hmm_state?: number | null;
  hmm_confidence?: number | null;
  hmm_state_label?: string | null;
  sigma?: number | null;
  forecast_return?: number;
  forecast_uncertainty?: number;
}

interface MetricsData {
  marketState?: { posture: string; status: string; lastUpdated: string };
  activePlans?: ActivePlan[];
  topRanked?: string[];
  tradeSummary?: TradeSummary;
  equityPositions?: any[];
  equityTradeSummary?: TradeSummary;
  forecastTimeseries?: ForecastPoint[];
  backtestEquity?: { timestamp: string; equity: number }[];
  walkforwardSummary?: any[];
  generatedAt?: string;
  symbolsInUniverse?: string[];
  accountBalance?: string;
  realizedPnl?: string;
  activeRisk?: string;
  winRate?: string;
  signals?: any[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseRegimeKey(key: string | undefined) {
  if (!key) return { direction: "RANGE", volatility: "MOD VOL", liquidity: "LIQUID", flow: "SUPPORTIVE" };
  const parts = key.split("|").map((s) => s.trim().toLowerCase());
  const directionRaw = parts[0] ?? "";
  const volRaw = parts[1] ?? "";
  const liqRaw = parts[2] ?? "";
  const flowRaw = parts[3] ?? "";

  const direction =
    directionRaw.includes("bull") ? "BULLISH"
    : directionRaw.includes("bear") ? "BEARISH"
    : directionRaw.includes("range") ? "RANGE"
    : directionRaw.includes("transition") ? "TRANSITION"
    : directionRaw.toUpperCase() || "RANGE";

  const volatility =
    volRaw.includes("high") ? "HIGH VOL"
    : volRaw.includes("low") ? "LOW VOL"
    : volRaw.includes("transition") ? "TRANSITION VOL"
    : "MOD VOL";

  const liquidity =
    liqRaw.includes("low") ? "LOW LIQ"
    : liqRaw.includes("high") ? "HIGH LIQ"
    : "LIQUID";

  const flow =
    flowRaw.includes("destab") ? "DESTABILIZING"
    : flowRaw.includes("support") ? "SUPPORTIVE"
    : "UNKNOWN";

  return { direction, volatility, liquidity, flow };
}

function dirColor(dir: string) {
  if (dir.includes("BULL")) return "text-green-400";
  if (dir.includes("BEAR")) return "text-red-400";
  return "text-amber-400";
}

function dirBg(dir: string) {
  if (dir.includes("BULL")) return "bg-green-500/15 border-green-500/30";
  if (dir.includes("BEAR")) return "bg-red-500/15 border-red-500/30";
  return "bg-amber-500/15 border-amber-500/30";
}

function actionColor(action: string) {
  const a = (action ?? "").toUpperCase();
  if (a === "ENTER" || a === "BUY") return { text: "text-green-400", bg: "bg-green-500/15 border-green-500/30" };
  if (a === "EXIT" || a === "CLOSE") return { text: "text-red-400", bg: "bg-red-500/15 border-red-500/30" };
  return { text: "text-slate-400", bg: "bg-slate-500/15 border-slate-500/30" };
}

function vpRating(conf: number) {
  if (conf >= 0.7) return { label: "PASS", color: "text-green-400", bg: "bg-green-500/15" };
  if (conf >= 0.4) return { label: "WATCH", color: "text-amber-400", bg: "bg-amber-500/15" };
  return { label: "FAIL", color: "text-red-400", bg: "bg-red-500/15" };
}

function transitionRiskLabel(val: number | undefined): { text: string; color: string } {
  const v = val ?? 0;
  if (v < 0.25) return { text: "Low", color: "text-emerald-400" };
  if (v < 0.5) return { text: "Medium", color: "text-amber-400" };
  if (v < 0.75) return { text: "High", color: "text-orange-400" };
  return { text: "Critical", color: "text-red-400" };
}

function pct(v: number | undefined) {
  return v != null ? (v * 100).toFixed(1) + "%" : "—";
}

function fmtUsd(v: number | undefined) {
  if (v == null) return "—";
  return v.toLocaleString("en-US", { style: "currency", currency: "USD", signDisplay: "always" });
}

function fmtDate(ts: string | undefined) {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleString("en-US", {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  } catch { return ts; }
}

function factorBar(value: number, label: string) {
  const abs = Math.min(Math.abs(value), 1);
  const positive = value >= 0;
  return (
    <div key={label} className="flex items-center gap-3">
      <span className="text-xs text-slate-500 w-8 font-mono">{label}</span>
      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden relative">
        <div
          className={cn(
            "absolute top-0 h-full rounded-full transition-all duration-700",
            positive ? "bg-cyan-400/70 left-1/2" : "bg-purple-400/70 right-1/2"
          )}
          style={{ width: `${abs * 50}%` }}
        />
        <div className="absolute top-0 left-1/2 w-px h-full bg-white/10" />
      </div>
      <span className={cn("text-xs font-mono w-12 text-right", positive ? "text-cyan-400" : "text-purple-400")}>
        {value >= 0 ? "+" : ""}{value.toFixed(2)}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

function GlassCard({ children, className, delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className={cn("glass rounded-2xl relative overflow-hidden", className)}
    >
      {children}
    </motion.div>
  );
}

function Badge({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={cn("text-[10px] font-bold tracking-wider px-2.5 py-1 rounded-md border uppercase", className)}>
      {children}
    </span>
  );
}

function ConfidenceArc({ value, size = 96, label }: { value: number; size?: number; label?: string }) {
  const radius = (size - 12) / 2;
  const circumference = Math.PI * radius;
  const offset = circumference * (1 - Math.min(value, 1));
  const color = value >= 0.7 ? "#22c55e" : value >= 0.4 ? "#f59e0b" : "#ef4444";

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size / 2 + 8} viewBox={`0 0 ${size} ${size / 2 + 8}`}>
        <path
          d={`M 6 ${size / 2 + 2} A ${radius} ${radius} 0 0 1 ${size - 6} ${size / 2 + 2}`}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="6"
          strokeLinecap="round"
        />
        <path
          d={`M 6 ${size / 2 + 2} A ${radius} ${radius} 0 0 1 ${size - 6} ${size / 2 + 2}`}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1s ease-out", filter: `drop-shadow(0 0 6px ${color}60)` }}
        />
        <text x={size / 2} y={size / 2 - 4} textAnchor="middle" fill={color} fontSize="18" fontWeight="700" fontFamily="monospace">
          {(value * 100).toFixed(0)}%
        </text>
      </svg>
      {label && <span className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</span>}
    </div>
  );
}

function StatBlock({ label, value, sub, icon: Icon, color = "text-cyan-400" }: {
  label: string; value: string; sub?: string; icon?: any; color?: string;
}) {
  return (
    <div className="flex items-start gap-3">
      {Icon && (
        <div className="p-2 rounded-lg bg-white/5 mt-0.5">
          <Icon className={cn("w-4 h-4", color)} />
        </div>
      )}
      <div>
        <p className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</p>
        <p className={cn("text-sm font-bold font-mono", color)}>{value}</p>
        {sub && <p className="text-[10px] text-slate-600">{sub}</p>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function CommandCenter() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const fetchMetrics = useCallback(async () => {
    try {
      const res = await fetch("/api/metrics", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setMetrics(data);
      setError(null);
    } catch (err: any) {
      setError(err.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
    const iv = setInterval(() => {
      fetchMetrics();
      setTick((t) => t + 1);
    }, 15000);
    return () => clearInterval(iv);
  }, [fetchMetrics]);

  const plans = metrics?.activePlans ?? [];
  const topPlan = useMemo(() => {
    if (!plans.length) return null;
    const ranked = metrics?.topRanked;
    if (ranked?.length) {
      const found = plans.find((p) => p.symbol === ranked[0] || p.planId === ranked[0]);
      if (found) return found;
    }
    return [...plans].sort((a, b) => (b.rankScore ?? b.confidence ?? 0) - (a.rankScore ?? a.confidence ?? 0))[0];
  }, [plans, metrics?.topRanked]);

  const spyPlan = useMemo(() => plans.find((p) => p.symbol === "SPY") ?? topPlan, [plans, topPlan]);
  const regime = useMemo(() => parseRegimeKey(spyPlan?.regimeKey), [spyPlan]);

  const ts = metrics?.tradeSummary;
  const eqTs = metrics?.equityTradeSummary;
  const forecast = metrics?.forecastTimeseries ?? [];
  const symbols = metrics?.symbolsInUniverse ?? plans.map((p) => p.symbol);

  const chartData = useMemo(
    () =>
      forecast.map((pt, idx) => ({
        ...pt,
        idx,
        dateLabel: fmtDate(pt.timestamp),
      })),
    [forecast],
  );

  const heatPath = useMemo(
    () =>
      forecast.map((pt) => ({
        sD: pt.S_D != null ? pt.S_D : 5,
        sV: pt.S_V != null ? pt.S_V : 5,
      })),
    [forecast],
  );

  const latestFc = forecast.length ? forecast[forecast.length - 1] : null;
  const transitionLbl = transitionRiskLabel(spyPlan?.M_R_trans);

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  const vp = vpRating(topPlan?.regimeConfidence ?? 0);
  const blendConf = Math.max(
    spyPlan?.regimeConfidence ?? 0,
    spyPlan?.hmmConfidence ?? 0,
    latestFc?.hmm_confidence ?? 0,
  );
  const markovProb =
    latestFc?.hmm_confidence != null
      ? (latestFc.hmm_confidence * 0.92).toFixed(2)
      : spyPlan?.regimeConfidence != null
        ? (spyPlan.regimeConfidence * 0.88).toFixed(2)
        : "—";
  const envOk =
    topPlan && !topPlan.vaultTriggered && vp.label === "PASS";

  return (
    <div className="flex flex-col xl:flex-row gap-6 max-w-[1920px] mx-auto items-start">
      <div className="flex-1 min-w-0 space-y-4 w-full">
      {/* ── ROW 1 · MARKET STATE BAR ──────────────────────────────────── */}
      <GlassCard className="p-5 panel-hud neon-border" delay={0}>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">

          {/* Left: regime badges + metrics */}
          <div className="space-y-4">
          <div className="flex flex-wrap gap-2 items-center">
            <Badge className={dirBg(regime.direction)}>
              <span className={dirColor(regime.direction)}>{regime.direction}</span>
            </Badge>
            <Badge className={cn(regime.volatility.includes("HIGH") ? "bg-red-500/15 border-red-500/30 text-red-400" : regime.volatility.includes("LOW") ? "bg-green-500/15 border-green-500/30 text-green-400" : "bg-amber-500/15 border-amber-500/30 text-amber-400")}>
              {regime.volatility}
            </Badge>
            <Badge className={cn(regime.liquidity.includes("LOW") ? "bg-red-500/15 border-red-500/30 text-red-400" : "bg-green-500/15 border-green-500/30 text-green-400")}>
              {regime.liquidity}
            </Badge>
            <Badge className={cn(regime.flow === "DESTABILIZING" ? "bg-red-500/15 border-red-500/30 text-red-400" : "bg-green-500/15 border-green-500/30 text-green-400")}>
              {regime.flow}
            </Badge>
            {metrics?.marketState?.posture && (
              <span className="text-[10px] text-slate-600 ml-2 font-mono">{metrics.marketState.posture}</span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-2 mt-4 pt-4 border-t border-white/[0.06]">
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-[family-name:var(--font-mono)]">Confidence</p>
              <p className="text-lg font-bold font-[family-name:var(--font-mono)] text-emerald-400">{pct(blendConf)}</p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-[family-name:var(--font-mono)]">HMM State</p>
              <p className="text-lg font-bold font-[family-name:var(--font-mono)] text-violet-300">
                {latestFc ? displayHmmState(latestFc) : "—"}
              </p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-[family-name:var(--font-mono)]">Markov Prob</p>
              <p className="text-lg font-bold font-[family-name:var(--font-mono)] text-cyan-300">{markovProb}</p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-[family-name:var(--font-mono)]">Transition Risk</p>
              <p className={cn("text-lg font-bold font-[family-name:var(--font-mono)]", transitionLbl.color)}>
                {transitionLbl.text}
              </p>
            </div>
          </div>
          </div>

          {/* Center: recommended action */}
          <div className="flex flex-col items-center justify-center text-center gap-2 border-y border-white/[0.06] lg:border-x lg:border-y-0 py-4 lg:py-0 lg:px-4">
            {topPlan ? (
              <>
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Recommended Action</p>
                <div className="flex flex-col items-center gap-2">
                  <span className={cn("text-4xl font-black tracking-tight drop-shadow-[0_0_24px_rgba(52,211,153,0.35)]", actionColor(topPlan.action).text)}>
                    {topPlan.action?.toUpperCase() ?? "—"}
                  </span>
                  <div className="flex flex-wrap items-center justify-center gap-2 text-sm">
                    <Crosshair className={cn("w-4 h-4 shrink-0", actionColor(topPlan.action).text)} />
                    <span className="font-semibold text-foreground">{topPlan.strategy}</span>
                  </div>
                  <p className="text-[11px] text-slate-500">
                    Size{" "}
                    <span className="text-cyan-400 font-bold font-[family-name:var(--font-mono)]">{pct(topPlan.sizeFraction)}</span>{" "}
                    of capital
                  </p>
                </div>
                <div className="mt-2 w-full rounded-xl border border-white/[0.06] bg-black/30 px-4 py-3 text-left">
                  <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Rationale</p>
                  <p className="text-xs text-slate-300 leading-relaxed line-clamp-4">{topPlan.rationale || "No rationale provided."}</p>
                </div>
              </>
            ) : (
              <span className="text-sm text-slate-600">No active plan</span>
            )}
          </div>

          {/* Right: risk status */}
          <div className="flex flex-col gap-3 justify-center">
            <p className="text-[10px] text-slate-500 uppercase tracking-[0.18em]">Risk Status</p>
            <div className="grid grid-cols-2 gap-3 text-right">
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider">Uncertainty</p>
                <p className="text-sm font-bold font-[family-name:var(--font-mono)] text-amber-400">
                  {topPlan ? pct(topPlan.forecastUncertainty) : "—"}
                </p>
              </div>
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider">Vault</p>
                <p className={cn("text-sm font-bold font-[family-name:var(--font-mono)]", topPlan?.vaultTriggered ? "text-red-400" : "text-emerald-400")}>
                  {topPlan?.vaultTriggered ? "ON" : "OFF"}
                </p>
              </div>
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider">VP Gating</p>
                <p className={cn("text-sm font-bold font-[family-name:var(--font-mono)]", vp.color)}>{vp.label}</p>
              </div>
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider">Environment</p>
                <p className={cn("text-sm font-bold font-[family-name:var(--font-mono)]", envOk ? "text-emerald-400" : "text-amber-400")}>
                  {envOk ? "Tradeable" : "Caution"}
                </p>
              </div>
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider">Drawdown Risk</p>
                <p className={cn("text-sm font-bold font-[family-name:var(--font-mono)]", transitionLbl.color)}>
                  {transitionLbl.text}
                </p>
              </div>
            </div>
          </div>
        </div>
      </GlassCard>

      {/* ── ROW 2 · LOCUS HEATMAP + PRICE ─────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-stretch">
        <RegimeHeatmap path={heatPath} title="Regime Locus Matrix" />

        <GlassCard className="p-6 panel-hud flex flex-col" delay={0.05}>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-cyan-400" />
              <div>
                <h2 className="text-sm font-bold uppercase tracking-wider leading-tight">Price &amp; State</h2>
                {spyPlan && (
                  <p className="text-[11px] text-slate-500 font-[family-name:var(--font-mono)] mt-0.5">
                    {spyPlan.symbol} · ${spyPlan.close?.toFixed(2)}
                  </p>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_10px_rgba(34,211,238,0.8)]" />
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">Live</span>
            </div>
          </div>

          <div className="h-[260px] w-full shrink-0">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                  <defs>
                    <linearGradient id="closeFill2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.28} />
                      <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  {chartData.map((d, i) => {
                    if (i >= chartData.length - 1) return null;
                    const bullish = (d.S_D ?? 5) >= 5;
                    return (
                      <ReferenceArea
                        key={`seg-${d.idx}`}
                        x1={d.idx}
                        x2={chartData[i + 1]?.idx ?? d.idx}
                        yAxisId="left"
                        strokeOpacity={0}
                        fill={bullish ? "rgba(34,197,94,0.07)" : "rgba(239,68,68,0.06)"}
                      />
                    );
                  })}
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e293b80" />
                  <XAxis
                    dataKey="idx"
                    type="number"
                    axisLine={false}
                    tickLine={false}
                    domain={["dataMin", "dataMax"]}
                    tickFormatter={(v: number) => chartData[v]?.dateLabel ?? ""}
                    tick={{ fill: "#64748b", fontSize: 9 }}
                    height={28}
                  />
                  <YAxis
                    yAxisId="left"
                    domain={["auto", "auto"]}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#64748b", fontSize: 10 }}
                    tickFormatter={(v: number) => `$${v}`}
                    width={50}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0d1118",
                      border: "1px solid rgba(255,255,255,0.08)",
                      borderRadius: "12px",
                      fontSize: 11,
                    }}
                    labelStyle={{ color: "#94a3b8" }}
                    itemStyle={{ color: "#22d3ee" }}
                    formatter={(value: number) => [`$${value?.toFixed?.(2) ?? value}`, "Close"]}
                    labelFormatter={(_, item: any) => item?.payload?.dateLabel ?? ""}
                  />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="close"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    fill="url(#closeFill2)"
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-600 text-sm rounded-xl border border-dashed border-white/10">
                {loading ? "Loading forecast..." : "No forecast data"}
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 mt-4 flex-1 min-h-[120px]">
            <div className="rounded-xl border border-white/[0.06] bg-black/25 p-2">
              <p className="text-[9px] uppercase tracking-wider text-slate-500 mb-1 px-1">
                Volatility factor (S_V)
              </p>
              <div className="h-[72px]">
                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 2, right: 2, bottom: 0, left: -18 }}>
                      <Area type="monotone" dataKey="S_V" stroke="#f472b6" strokeWidth={1.5} fill="rgba(244,114,182,0.12)" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : null}
              </div>
            </div>
            <div className="rounded-xl border border-white/[0.06] bg-black/25 p-2">
              <p className="text-[9px] uppercase tracking-wider text-slate-500 mb-1 px-1">
                Regime confidence
              </p>
              <div className="h-[72px]">
                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 2, right: 2, bottom: 0, left: -18 }}>
                      <Area
                        type="monotone"
                        dataKey="hmm_confidence"
                        stroke="#a78bfa"
                        strokeWidth={1.5}
                        fill="rgba(167,139,250,0.15)"
                        dot={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : null}
              </div>
            </div>
          </div>
        </GlassCard>
      </div>

      {/* ── ROW 3 · THREE COLUMNS ─────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* WHY RLM THINKS THIS */}
        <GlassCard className="p-6" delay={0.15}>
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <h2 className="text-sm font-bold uppercase tracking-wider">Why RLM Thinks This</h2>
          </div>

          {topPlan ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-500 uppercase">Strategy</span>
                  <span className="text-xs font-semibold text-foreground">{topPlan.strategy}</span>
                </div>
                {topPlan.coordinateStrategy && (
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-slate-500 uppercase">Coordinate</span>
                    <span className="text-xs font-mono text-purple-400">{topPlan.coordinateStrategy}</span>
                  </div>
                )}
                <div className="bg-white/[0.03] rounded-lg p-3 mt-2">
                  <p className="text-xs text-slate-400 leading-relaxed">{topPlan.rationale || "No rationale provided."}</p>
                </div>
              </div>

              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Top Drivers</p>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { k: "S_D", v: topPlan.S_D, label: "Direction" },
                    { k: "S_V", v: topPlan.S_V, label: "Volatility" },
                    { k: "S_L", v: topPlan.S_L, label: "Liquidity" },
                    { k: "S_G", v: topPlan.S_G, label: "Flow" },
                  ]
                    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
                    .map((d) => (
                      <div key={d.k} className="bg-white/[0.03] rounded-lg px-3 py-2">
                        <p className="text-[10px] text-slate-600">{d.label}</p>
                        <p className={cn("text-sm font-bold font-mono", d.v >= 0 ? "text-cyan-400" : "text-purple-400")}>
                          {d.v >= 0 ? "+" : ""}{d.v.toFixed(3)}
                        </p>
                      </div>
                    ))}
                </div>
              </div>

              {(topPlan.M_D != null || topPlan.M_V != null) && (
                <div>
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">MTF Confluence</p>
                  <div className="flex flex-wrap gap-2">
                    {topPlan.M_D != null && <Badge className="bg-white/5 border-white/10 text-slate-400">M_D: {topPlan.M_D.toFixed(2)}</Badge>}
                    {topPlan.M_V != null && <Badge className="bg-white/5 border-white/10 text-slate-400">M_V: {topPlan.M_V.toFixed(2)}</Badge>}
                    {topPlan.M_L != null && <Badge className="bg-white/5 border-white/10 text-slate-400">M_L: {topPlan.M_L.toFixed(2)}</Badge>}
                    {topPlan.M_G != null && <Badge className="bg-white/5 border-white/10 text-slate-400">M_G: {topPlan.M_G.toFixed(2)}</Badge>}
                    {topPlan.M_trend_strength != null && <Badge className="bg-white/5 border-white/10 text-slate-400">Trend: {topPlan.M_trend_strength.toFixed(2)}</Badge>}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-slate-600">Waiting for data…</p>
          )}
        </GlassCard>

        {/* RECENT TRANSITIONS / ACTIVE SIGNALS */}
        <GlassCard className="p-6" delay={0.2}>
          <div className="flex items-center gap-2 mb-4">
            <Radio className="w-4 h-4 text-amber-400" />
            <h2 className="text-sm font-bold uppercase tracking-wider">Active Signals</h2>
            <span className="text-[10px] text-slate-600 ml-auto font-mono">{symbols.length} sym</span>
          </div>

          <div className="space-y-1.5 max-h-[380px] overflow-y-auto pr-1 scrollbar-thin">
            {plans.length > 0 ? (
              [...plans]
                .sort((a, b) => (b.rankScore ?? b.confidence ?? 0) - (a.rankScore ?? a.confidence ?? 0))
                .map((p) => {
                  const ac = actionColor(p.action);
                  const rk = parseRegimeKey(p.regimeKey);
                  return (
                    <div
                      key={p.planId ?? p.symbol}
                      className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/[0.02] hover:bg-white/[0.05] transition-colors border border-transparent hover:border-white/5"
                    >
                      <div className={cn("w-9 h-9 rounded-lg flex items-center justify-center text-xs font-black border", ac.bg)}>
                        {p.symbol?.slice(0, 3)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-bold">{p.symbol}</span>
                          <span className={cn("text-[10px] font-bold", ac.text)}>{p.action?.toUpperCase()}</span>
                        </div>
                        <div className="flex items-center gap-1.5 mt-0.5">
                          <span className="text-[10px] text-slate-600 truncate">{p.strategy}</span>
                          <span className="text-[10px] text-slate-700">·</span>
                          <span className={cn("text-[10px]", dirColor(rk.direction))}>{rk.direction}</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-xs font-bold font-mono text-cyan-400">{pct(p.confidence)}</p>
                        <p className="text-[10px] text-slate-600">{p.regimeKey?.slice(0, 12)}</p>
                      </div>
                    </div>
                  );
                })
            ) : (
              <div className="text-center py-8 text-slate-600 text-xs">
                {loading ? "Loading signals…" : "No active signals"}
              </div>
            )}
          </div>
        </GlassCard>

        {/* QUICK STATS */}
        <GlassCard className="p-6 flex flex-col gap-4" delay={0.25}>
          <div className="flex items-center gap-2 mb-1">
            <Gauge className="w-4 h-4 text-green-400" />
            <h2 className="text-sm font-bold uppercase tracking-wider">Quick Stats</h2>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <StatBlock
              label="Win Rate"
              value={ts ? pct(ts.winRate) : (metrics?.winRate ?? "—")}
              icon={Target}
              color={ts && ts.winRate >= 0.5 ? "text-green-400" : "text-amber-400"}
            />
            <StatBlock
              label="Total Trades"
              value={ts ? String(ts.totalTrades) : "—"}
              icon={Layers}
              color="text-cyan-400"
            />
            <StatBlock
              label="Open Trades"
              value={ts ? String(ts.openTrades) : "—"}
              icon={Activity}
              color="text-purple-400"
            />
            <StatBlock
              label="Realized PnL"
              value={ts ? fmtUsd(ts.realizedPnl) : (metrics?.realizedPnl ?? "—")}
              icon={ts && ts.realizedPnl >= 0 ? TrendingUp : TrendingDown}
              color={ts && ts.realizedPnl >= 0 ? "text-green-400" : "text-red-400"}
            />
            <StatBlock
              label="Unrealized PnL"
              value={ts ? fmtUsd(ts.unrealizedPnl) : (metrics?.activeRisk ?? "—")}
              icon={Sigma}
              color={ts && ts.unrealizedPnl >= 0 ? "text-green-400" : "text-red-400"}
            />
            <StatBlock
              label="Best Strategy"
              value={topPlan?.strategy ?? "—"}
              icon={Zap}
              color="text-cyan-400"
            />
          </div>

          {/* Mini equity positions summary */}
          {eqTs && (
            <div className="mt-auto pt-3 border-t border-white/5">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Equity Positions</p>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-lg font-bold font-mono text-foreground">{eqTs.totalTrades}</p>
                  <p className="text-[10px] text-slate-600">Total</p>
                </div>
                <div>
                  <p className="text-lg font-bold font-mono text-green-400">{pct(eqTs.winRate)}</p>
                  <p className="text-[10px] text-slate-600">Win Rate</p>
                </div>
                <div>
                  <p className={cn("text-lg font-bold font-mono", eqTs.realizedPnl >= 0 ? "text-green-400" : "text-red-400")}>
                    {fmtUsd(eqTs.realizedPnl)}
                  </p>
                  <p className="text-[10px] text-slate-600">Realized</p>
                </div>
              </div>
            </div>
          )}
        </GlassCard>
      </div>
      </div>

      <aside className="hidden xl:flex w-[300px] shrink-0 flex-col gap-4 xl:sticky xl:top-4 xl:self-start">
        <GlassCard className="p-5 panel-hud" delay={0.08}>
          <div className="flex items-center gap-2 mb-4">
            <Brain className="w-4 h-4 text-violet-400" />
            <h2 className="text-xs font-bold uppercase tracking-[0.15em] text-slate-300">Current State</h2>
          </div>
          <div className="flex items-baseline gap-2 mb-4">
            <span className="text-5xl font-black font-[family-name:var(--font-mono)] tracking-tight text-white drop-shadow-[0_0_24px_rgba(167,139,250,0.35)]">
              {forecast.length > 0
                ? displayHmmState(forecast[forecast.length - 1])
                : spyPlan?.hmmState ?? "—"}
            </span>
          </div>
          <ul className="space-y-2 text-[11px] text-slate-400 mb-6 border-b border-white/[0.06] pb-4">
            <li className="flex gap-2">
              <span className="text-emerald-400">●</span>
              <span>{regime.direction} trend bias</span>
            </li>
            <li className="flex gap-2">
              <span className="text-cyan-400">●</span>
              <span>{regime.volatility.toLowerCase()} regime</span>
            </li>
            <li className="flex gap-2">
              <span className="text-violet-400">●</span>
              <span>{regime.liquidity} liquidity</span>
            </li>
          </ul>
          <ConfidenceArc value={Math.min(1, Math.max(0, blendConf))} label="Confidence" />
          <div className="mt-6 space-y-3">
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Key Metrics</p>
            {spyPlan ? (
              <>
                {factorBar(spyPlan.S_D, "S_D")}
                {factorBar(spyPlan.S_V, "S_V")}
              </>
            ) : (
              <p className="text-xs text-slate-600">No plan data</p>
            )}
          </div>
        </GlassCard>

        <GlassCard className="p-5 panel-hud border-amber-500/15" delay={0.12}>
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <h2 className="text-xs font-bold uppercase tracking-[0.15em]">Alerts</h2>
          </div>
          <div className="space-y-3">
            <div className="rounded-lg border border-amber-500/25 bg-amber-500/[0.06] px-3 py-2">
              <p className="text-[11px] font-semibold text-amber-200">Volatility clustering</p>
              <p className="text-[10px] text-slate-500 mt-1 leading-snug">
                Watch band width vs. forecast — widen reduces conviction on directional bets.
              </p>
            </div>
            <div className="rounded-lg border border-cyan-500/25 bg-cyan-500/[0.05] px-3 py-2">
              <p className="text-[11px] font-semibold text-cyan-200">Pipeline heartbeat</p>
              <p className="text-[10px] text-slate-500 mt-1 leading-snug">
                Forecast CSV drives SPY overlay; refresh dashboards after batch runs.
              </p>
            </div>
          </div>
        </GlassCard>
      </aside>
    </div>
  );
}
