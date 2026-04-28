"use client";

import React, { useEffect, useState, useMemo } from "react";
import {
  ShieldAlert,
  Lock,
  Brain,
  Radio,
  RefreshCcw,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

function fmt(v: number | null | undefined, decimals = 2): string {
  if (v == null || isNaN(v)) return "—";
  return v.toFixed(decimals);
}

function fmtDollar(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/** CSV fractions are typically 0–1; avoids "-0.0%" float noise on tiny values */
function fmtPct(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  const pct = v * 100;
  if (Math.abs(pct) < 0.05 && pct !== 0) return `${pct.toFixed(2)}%`;
  return `${pct.toFixed(1)}%`;
}

function effectiveRegimeConfidence(p: { regimeConfidence?: number; hmmConfidence?: number }): number {
  return Math.max(
    Number(p.regimeConfidence) || 0,
    Number(p.hmmConfidence) || 0,
  );
}

function fmtDate(ts: string): string {
  if (!ts) return "";
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts.slice(0, 10);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function confColor(v: number): string {
  if (v >= 0.7) return "text-green-400";
  if (v >= 0.4) return "text-amber-400";
  return "text-red-400";
}

function confBg(v: number): string {
  if (v >= 0.7) return "bg-green-500";
  if (v >= 0.4) return "bg-amber-500";
  return "bg-red-500";
}

interface RiskCardProps {
  icon: React.ElementType;
  title: string;
  value: string;
  sub?: string;
  barPct?: number;
  barColor?: string;
  accent?: string;
  delay?: number;
}

function RiskCard({ icon: Icon, title, value, sub, barPct, barColor, accent = "primary", delay = 0 }: RiskCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="glass rounded-2xl p-5 relative overflow-hidden"
    >
      <div className="flex items-center gap-3 mb-3">
        <div className="p-2 rounded-lg bg-secondary border border-border">
          <Icon className={cn("w-5 h-5", `text-${accent}`)} />
        </div>
        <p className="text-sm font-medium text-muted-foreground">{title}</p>
      </div>
      <p className="text-2xl font-bold tracking-tight">{value}</p>
      {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
      {barPct != null && (
        <div className="mt-3 h-1.5 bg-secondary rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${Math.min(barPct, 100)}%`, backgroundColor: barColor || "#00f5ff" }}
          />
        </div>
      )}
    </motion.div>
  );
}

export default function RiskPage() {
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

  const plans: any[] = metrics?.activePlans ?? [];
  const equityPositions: any[] = metrics?.equityPositions ?? [];
  const backtestEquity: any[] = metrics?.backtestEquity ?? [];
  const walkforward: any[] = metrics?.walkforwardSummary ?? [];
  const eqSummary = metrics?.equityTradeSummary;

  const avgUncertainty = useMemo(() => {
    if (!plans.length) return 0;
    const sum = plans.reduce(
      (acc, p) => acc + (Number(p.forecastUncertainty) || 0),
      0,
    );
    return sum / plans.length;
  }, [plans]);

  const vaultCount = useMemo(
    () => plans.filter((p) => p.vaultTriggered).length,
    [plans],
  );

  const regimeSafe = useMemo(
    () =>
      plans.filter((p) => effectiveRegimeConfidence(p) > 0.7).length,
    [plans],
  );

  /** Universe marks funded trades with status active (action may still be enter on skipped runners). */
  const activeOptionPlans = useMemo(
    () =>
      plans.filter((p) => String(p.status || "").toLowerCase() === "active"),
    [plans],
  );

  const openEquity = useMemo(
    () =>
      equityPositions.filter(
        (p) => String(p.status || "").toLowerCase() === "open",
      ),
    [equityPositions],
  );

  const btData = useMemo(
    () =>
      backtestEquity.map((d) => ({
        ...d,
        dateLabel: fmtDate(d.timestamp),
      })),
    [backtestEquity],
  );

  const posture = metrics?.marketState?.posture || "UNKNOWN";
  const status = metrics?.marketState?.status || "UNKNOWN";

  return (
    <div className="space-y-6 max-w-[1600px] mx-auto">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Risk Center</h1>
          <p className="text-muted-foreground mt-1">
            Portfolio risk assessment and position monitoring
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

      {/* ── Row 1: Risk Overview Cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <RiskCard
          icon={ShieldAlert}
          title="Forecast Uncertainty"
          value={fmt(avgUncertainty, 4)}
          sub={`Avg across ${plans.length} plans`}
          barPct={avgUncertainty * 100}
          barColor={avgUncertainty > 0.05 ? "#ef4444" : avgUncertainty > 0.02 ? "#f59e0b" : "#22c55e"}
          accent="primary"
        />
        <RiskCard
          icon={Lock}
          title="Vault Status"
          value={`${vaultCount} / ${plans.length}`}
          sub="Vault triggered"
          barPct={plans.length > 0 ? (vaultCount / plans.length) * 100 : 0}
          barColor={vaultCount > 0 ? "#ef4444" : "#22c55e"}
          accent="accent"
          delay={0.05}
        />
        <RiskCard
          icon={Brain}
          title="Regime Safety"
          value={`${regimeSafe} / ${plans.length}`}
          sub="Confidence > 70%"
          barPct={plans.length > 0 ? (regimeSafe / plans.length) * 100 : 0}
          barColor={regimeSafe / (plans.length || 1) > 0.6 ? "#22c55e" : "#f59e0b"}
          delay={0.1}
        />
        <RiskCard
          icon={Radio}
          title="System Status"
          value={posture}
          sub={status}
          delay={0.15}
        />
      </div>

      {/* ── Row 2: Positions ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Options Positions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass rounded-2xl p-6"
        >
          <h2 className="text-lg font-bold mb-4">Active Options Positions</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-muted-foreground text-xs uppercase tracking-wider">
                  <th className="text-left py-3 px-2">Symbol</th>
                  <th className="text-left py-3 px-2">Strategy</th>
                  <th className="text-right py-3 px-2">Entry</th>
                  <th className="text-right py-3 px-2">Regime conf</th>
                  <th className="text-center py-3 px-2">Vault</th>
                  <th className="text-left py-3 px-2">Regime</th>
                </tr>
              </thead>
              <tbody>
                {activeOptionPlans.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-6 text-center text-muted-foreground">
                      No plans with status &quot;active&quot; in universe_trade_plans.json
                    </td>
                  </tr>
                ) : (
                  activeOptionPlans.map((p, i) => {
                    const erc = effectiveRegimeConfidence(p);
                    return (
                    <tr
                      key={p.planId || i}
                      className={cn(
                        "border-b border-border/50 hover:bg-secondary/30 transition-colors",
                        erc >= 0.7 && "bg-green-500/5",
                        erc < 0.4 && "bg-red-500/5",
                      )}
                    >
                      <td className="py-3 px-2 font-semibold">{p.symbol}</td>
                      <td className="py-3 px-2 font-mono text-xs">{p.strategy || "—"}</td>
                      <td className="py-3 px-2 text-right">
                        {p.entryDebit != null ? fmtDollar(p.entryDebit) : "—"}
                      </td>
                      <td className={cn("py-3 px-2 text-right font-bold", confColor(erc))}>
                        {fmtPct(erc)}
                      </td>
                      <td className="py-3 px-2 text-center">
                        {p.vaultTriggered ? (
                          <span className="text-red-400 font-bold">YES</span>
                        ) : (
                          <span className="text-green-400">—</span>
                        )}
                      </td>
                      <td className="py-3 px-2 font-mono text-xs text-accent">{p.regimeKey || "—"}</td>
                    </tr>
                  );
                  })
                )}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Equity Positions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="glass rounded-2xl p-6"
        >
          <h2 className="text-lg font-bold mb-4">Equity Positions</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-muted-foreground text-xs uppercase tracking-wider">
                  <th className="text-left py-3 px-2">Symbol</th>
                  <th className="text-left py-3 px-2">Direction</th>
                  <th className="text-right py-3 px-2">Qty</th>
                  <th className="text-right py-3 px-2">Entry</th>
                  <th className="text-left py-3 px-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {openEquity.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-6 text-center text-muted-foreground">
                      No open equity positions
                    </td>
                  </tr>
                ) : (
                  openEquity.map((p, i) => (
                    <tr
                      key={p.planId || i}
                      className="border-b border-border/50 hover:bg-secondary/30 transition-colors"
                    >
                      <td className="py-3 px-2 font-semibold">{p.symbol}</td>
                      <td className="py-3 px-2">
                        <span
                          className={cn(
                            "inline-flex items-center gap-1 text-xs font-bold px-2 py-0.5 rounded",
                            p.direction === "long"
                              ? "bg-green-500/20 text-green-400"
                              : "bg-red-500/20 text-red-400",
                          )}
                        >
                          {p.direction === "long" ? (
                            <TrendingUp className="w-3 h-3" />
                          ) : (
                            <TrendingDown className="w-3 h-3" />
                          )}
                          {p.direction?.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3 px-2 text-right font-mono">{p.quantity}</td>
                      <td className="py-3 px-2 text-right">{fmtDollar(p.entryPrice)}</td>
                      <td className="py-3 px-2">
                        <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">
                          {p.status}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {eqSummary && (
            <div className="mt-4 pt-3 border-t border-border flex justify-between text-sm">
              <span className="text-muted-foreground">
                Total Unrealized P/L
              </span>
              <span
                className={cn(
                  "font-bold",
                  eqSummary.unrealizedPnl >= 0 ? "text-green-400" : "text-red-400",
                )}
              >
                {fmtDollar(eqSummary.unrealizedPnl)}
              </span>
            </div>
          )}
        </motion.div>
      </div>

      {/* ── Row 3: Backtest & Walkforward ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Backtest Equity Curve */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass rounded-2xl p-6"
        >
          <h2 className="text-lg font-bold mb-4">Backtest Equity Curve</h2>
          <div className="h-[280px] w-full">
            {btData.length === 0 ? (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                No backtest data available
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={btData} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="eqFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00f5ff" stopOpacity={0.3} />
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
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    tickFormatter={(v: number) => `$${v.toLocaleString()}`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#111114",
                      border: "1px solid #1e1e24",
                      borderRadius: "12px",
                      fontSize: 12,
                    }}
                    formatter={(value) => [fmtDollar(Number(value ?? 0)), "Equity"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke="#00f5ff"
                    strokeWidth={2}
                    fill="url(#eqFill)"
                    fillOpacity={1}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </motion.div>

        {/* Walkforward Performance */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="glass rounded-2xl p-6"
        >
          <h2 className="text-lg font-bold mb-4">Walk-Forward Performance</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-muted-foreground text-xs uppercase tracking-wider">
                  <th className="text-left py-3 px-2">Window</th>
                  <th className="text-left py-3 px-2">Period</th>
                  <th className="text-right py-3 px-2">Return%</th>
                  <th className="text-right py-3 px-2">MaxDD</th>
                  <th className="text-right py-3 px-2">Sharpe</th>
                  <th className="text-right py-3 px-2">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {walkforward.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-6 text-center text-muted-foreground">
                      No walk-forward data available
                    </td>
                  </tr>
                ) : (
                  walkforward.map((w, i) => (
                    <tr
                      key={w.windowId ?? i}
                      className="border-b border-border/50 hover:bg-secondary/30 transition-colors"
                    >
                      <td className="py-3 px-2 font-mono text-xs">#{w.windowId}</td>
                      <td className="py-3 px-2 text-xs">
                        {fmtDate(w.oosStart)}–{fmtDate(w.oosEnd)}
                      </td>
                      <td
                        className={cn(
                          "py-3 px-2 text-right font-bold",
                          w.totalReturnPct >= 0 ? "text-green-400" : "text-red-400",
                        )}
                      >
                        {fmtPct(w.totalReturnPct)}
                      </td>
                      <td className="py-3 px-2 text-right text-red-400">
                        {fmtPct(Math.abs(w.maxDrawdown ?? 0))}
                      </td>
                      <td className="py-3 px-2 text-right">{fmt(w.sharpe, 2)}</td>
                      <td className="py-3 px-2 text-right">{fmtPct(w.winRate)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
