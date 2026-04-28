"use client";

import React, { useEffect, useState, useMemo } from "react";
import {
  Activity,
  Layers,
  ArrowRightLeft,
  BarChart3,
  RefreshCcw,
} from "lucide-react";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  ReferenceLine,
} from "recharts";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

const STATE_COLORS: Record<string, string> = {
  "bull-quiet": "#22c55e",
  "bull-volatile": "#86efac",
  "bear-quiet": "#ef4444",
  "bear-volatile": "#fca5a5",
  "neutral": "#94a3b8",
  "crisis": "#f59e0b",
};

function stateColor(label: string | null): string {
  if (!label) return "#94a3b8";
  const key = label.toLowerCase().replace(/[\s_]+/g, "-");
  return STATE_COLORS[key] ?? "#00f5ff";
}

function fmt(v: number | null | undefined, decimals = 2): string {
  if (v == null || isNaN(v)) return "—";
  return v.toFixed(decimals);
}

function fmtDate(ts: string): string {
  if (!ts) return "";
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts.slice(0, 10);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

interface ForecastPoint {
  timestamp: string;
  close: number;
  S_D: number;
  S_V: number;
  S_L: number;
  S_G: number;
  mean_price: number | null;
  lower_1s: number | null;
  upper_1s: number | null;
  lower_2s: number | null;
  upper_2s: number | null;
  hmm_state: number | null;
  hmm_confidence: number | null;
  hmm_state_label: string | null;
  forecast_return: number | null;
  forecast_uncertainty: number | null;
}

export default function StateMapPage() {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const fetchMetrics = async () => {
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();
      setMetrics(data);
    } catch (err) {
      console.error("Failed to load metrics", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 15000);
    return () => clearInterval(interval);
  }, []);

  const ts: ForecastPoint[] = metrics?.forecastTimeseries ?? [];

  const chartData = useMemo(
    () =>
      ts.map((p) => ({
        ...p,
        dateLabel: fmtDate(p.timestamp),
        band1: p.lower_1s != null && p.upper_1s != null ? [p.lower_1s, p.upper_1s] : undefined,
        band2: p.lower_2s != null && p.upper_2s != null ? [p.lower_2s, p.upper_2s] : undefined,
        confColor:
          (p.hmm_confidence ?? 0) > 0.8
            ? "#22c55e"
            : (p.hmm_confidence ?? 0) >= 0.5
              ? "#f59e0b"
              : "#ef4444",
      })),
    [ts],
  );

  const latest = ts.length > 0 ? ts[ts.length - 1] : null;

  const transitions = useMemo(() => {
    const result: { from: string; to: string; timestamp: string }[] = [];
    for (let i = 1; i < ts.length; i++) {
      if (ts[i].hmm_state_label !== ts[i - 1].hmm_state_label) {
        result.push({
          from: ts[i - 1].hmm_state_label || "unknown",
          to: ts[i].hmm_state_label || "unknown",
          timestamp: ts[i].timestamp,
        });
      }
    }
    return result.slice(-10).reverse();
  }, [ts]);

  const priceDomain = useMemo(() => {
    if (chartData.length === 0) return [0, 100];
    const lows = chartData.map((d) => d.lower_2s ?? d.lower_1s ?? d.close);
    const highs = chartData.map((d) => d.upper_2s ?? d.upper_1s ?? d.close);
    const min = Math.min(...lows);
    const max = Math.max(...highs);
    const pad = (max - min) * 0.05;
    return [Math.floor(min - pad), Math.ceil(max + pad)];
  }, [chartData]);

  const regimeKey = latest
    ? (metrics?.activePlans?.[0]?.regimeKey || "—")
    : "—";

  return (
    <div className="space-y-6 max-w-[1600px] mx-auto">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">State Map</h1>
          <p className="text-muted-foreground mt-1">
            Regime transitions and price-state overlay
          </p>
        </div>
        <button
          onClick={() => {
            setLoading(true);
            fetchMetrics();
          }}
          className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-2 rounded-xl font-semibold hover:opacity-90 transition-opacity disabled:opacity-50"
          disabled={loading}
        >
          <RefreshCcw className={cn("w-4 h-4", loading && "animate-spin")} />
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ── Main area: 2/3 width ── */}
        <div className="lg:col-span-2 space-y-6">
          {/* Price & State Chart */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-2xl p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-secondary border border-border">
                <Layers className="w-5 h-5 text-primary neon-text" />
              </div>
              <h2 className="text-lg font-bold">Price & State Overlay</h2>
            </div>

            <div className="h-[400px] w-full">
              {chartData.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  No forecast data available
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                    <defs>
                      <linearGradient id="band2Fill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#00f5ff" stopOpacity={0.04} />
                        <stop offset="100%" stopColor="#00f5ff" stopOpacity={0.02} />
                      </linearGradient>
                      <linearGradient id="band1Fill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#00f5ff" stopOpacity={0.15} />
                        <stop offset="100%" stopColor="#00f5ff" stopOpacity={0.05} />
                      </linearGradient>
                      <linearGradient id="closeFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#00f5ff" stopOpacity={0.25} />
                        <stop offset="95%" stopColor="#00f5ff" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e1e24" />
                    <XAxis
                      dataKey="dateLabel"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      dy={10}
                    />
                    <YAxis
                      domain={priceDomain}
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      tickFormatter={(v: number) => `$${v}`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#111114",
                        border: "1px solid #1e1e24",
                        borderRadius: "12px",
                        fontSize: 12,
                      }}
                      labelStyle={{ color: "#94a3b8" }}
                      formatter={(value, name) => {
                        const key = String(name ?? "");
                        const labels: Record<string, string> = {
                          upper_2s: "Upper 2σ",
                          lower_2s: "Lower 2σ",
                          upper_1s: "Upper 1σ",
                          lower_1s: "Lower 1σ",
                          close: "Close",
                        };
                        return [`$${Number(value).toFixed(2)}`, labels[key] || key];
                      }}
                    />

                    <Area
                      type="monotone"
                      dataKey="upper_2s"
                      stroke="none"
                      fill="url(#band2Fill)"
                      fillOpacity={1}
                      isAnimationActive={false}
                    />
                    <Area
                      type="monotone"
                      dataKey="lower_2s"
                      stroke="none"
                      fill="#0a0a0b"
                      fillOpacity={1}
                      isAnimationActive={false}
                    />

                    <Area
                      type="monotone"
                      dataKey="upper_1s"
                      stroke="none"
                      fill="url(#band1Fill)"
                      fillOpacity={1}
                      isAnimationActive={false}
                    />
                    <Area
                      type="monotone"
                      dataKey="lower_1s"
                      stroke="none"
                      fill="#0a0a0b"
                      fillOpacity={1}
                      isAnimationActive={false}
                    />

                    <Area
                      type="monotone"
                      dataKey="close"
                      stroke="#00f5ff"
                      strokeWidth={2}
                      fill="url(#closeFill)"
                      fillOpacity={1}
                      dot={false}
                    />

                    {transitions.map((t, i) => (
                      <ReferenceLine
                        key={i}
                        x={fmtDate(t.timestamp)}
                        stroke={stateColor(t.to)}
                        strokeDasharray="4 4"
                        strokeOpacity={0.6}
                      />
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
              )}
            </div>

            {/* State color legend */}
            <div className="flex flex-wrap gap-4 mt-4 px-2">
              {Object.entries(STATE_COLORS).map(([label, color]) => (
                <div key={label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  {label}
                </div>
              ))}
            </div>
          </motion.div>

          {/* HMM Confidence Timeline */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass rounded-2xl p-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-secondary border border-border">
                <Activity className="w-5 h-5 text-accent" />
              </div>
              <h2 className="text-lg font-bold">HMM Confidence Timeline</h2>
            </div>

            <div className="h-[140px] w-full">
              {chartData.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                  No confidence data
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 10, bottom: 0 }}>
                    <defs>
                      <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e1e24" />
                    <XAxis
                      dataKey="dateLabel"
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#94a3b8", fontSize: 10 }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      axisLine={false}
                      tickLine={false}
                      tick={{ fill: "#94a3b8", fontSize: 10 }}
                      tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#111114",
                        border: "1px solid #1e1e24",
                        borderRadius: "12px",
                        fontSize: 12,
                      }}
                      formatter={(value) => [
                        `${(Number(value ?? 0) * 100).toFixed(1)}%`,
                        "Confidence",
                      ]}
                    />
                    <ReferenceLine y={0.8} stroke="#22c55e" strokeDasharray="3 3" strokeOpacity={0.4} />
                    <ReferenceLine y={0.5} stroke="#f59e0b" strokeDasharray="3 3" strokeOpacity={0.4} />
                    <Area
                      type="monotone"
                      dataKey="hmm_confidence"
                      stroke="#22c55e"
                      strokeWidth={2}
                      fill="url(#confGrad)"
                      fillOpacity={1}
                      dot={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
            <div className="flex gap-4 mt-2 px-2 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-green-500" /> &gt;80% High
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-amber-500" /> 50–80% Medium
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-red-500" /> &lt;50% Low
              </span>
            </div>
          </motion.div>
        </div>

        {/* ── Right sidebar: 1/3 width ── */}
        <div className="space-y-6">
          {/* Current State */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass rounded-2xl p-6 neon-border"
          >
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">
              Current State
            </h3>
            {latest ? (
              <div className="space-y-4">
                <div>
                  <p className="text-xs text-muted-foreground mb-1">HMM State</p>
                  <div className="flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: stateColor(latest.hmm_state_label) }}
                    />
                    <span className="text-xl font-bold neon-text">
                      {latest.hmm_state_label || "Unknown"}
                    </span>
                  </div>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Regime Key</p>
                  <p className="font-mono text-sm font-semibold text-accent">
                    {regimeKey}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">HMM Confidence</p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${(latest.hmm_confidence ?? 0) * 100}%`,
                          backgroundColor:
                            (latest.hmm_confidence ?? 0) > 0.8
                              ? "#22c55e"
                              : (latest.hmm_confidence ?? 0) >= 0.5
                                ? "#f59e0b"
                                : "#ef4444",
                        }}
                      />
                    </div>
                    <span className="text-sm font-bold">
                      {fmt((latest.hmm_confidence ?? 0) * 100, 0)}%
                    </span>
                  </div>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground mb-1">Close Price</p>
                  <p className="text-lg font-bold">${fmt(latest.close)}</p>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">Loading...</p>
            )}
          </motion.div>

          {/* Transition Table */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="glass rounded-2xl p-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <ArrowRightLeft className="w-4 h-4 text-accent" />
              <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                Recent Transitions
              </h3>
            </div>
            {transitions.length === 0 ? (
              <p className="text-sm text-muted-foreground">No transitions detected</p>
            ) : (
              <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
                {transitions.map((t, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 py-2 px-3 bg-secondary/30 rounded-lg border border-border text-sm"
                  >
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: stateColor(t.from) }}
                    />
                    <span className="truncate">{t.from}</span>
                    <ArrowRightLeft className="w-3 h-3 text-muted-foreground shrink-0" />
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: stateColor(t.to) }}
                    />
                    <span className="truncate font-semibold">{t.to}</span>
                    <span className="ml-auto text-xs text-muted-foreground whitespace-nowrap">
                      {fmtDate(t.timestamp)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </motion.div>

          {/* Forecast Bands */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="glass rounded-2xl p-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <BarChart3 className="w-4 h-4 text-primary" />
              <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                Forecast Bands
              </h3>
            </div>
            {latest ? (
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-secondary/30 rounded-lg p-3 border border-border">
                  <p className="text-xs text-muted-foreground">Upper 2σ</p>
                  <p className="font-bold">${fmt(latest.upper_2s)}</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-3 border border-border">
                  <p className="text-xs text-muted-foreground">Lower 2σ</p>
                  <p className="font-bold">${fmt(latest.lower_2s)}</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-3 border border-border">
                  <p className="text-xs text-muted-foreground">Upper 1σ</p>
                  <p className="font-bold text-primary">${fmt(latest.upper_1s)}</p>
                </div>
                <div className="bg-secondary/30 rounded-lg p-3 border border-border">
                  <p className="text-xs text-muted-foreground">Lower 1σ</p>
                  <p className="font-bold text-primary">${fmt(latest.lower_1s)}</p>
                </div>
                <div className="col-span-2 bg-secondary/30 rounded-lg p-3 border border-border">
                  <p className="text-xs text-muted-foreground">Sigma (σ)</p>
                  <p className="font-bold text-accent">{fmt(latest.S_D, 4)}</p>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">Loading...</p>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
