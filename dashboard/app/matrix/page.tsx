"use client";

import React, { useEffect, useState, useMemo, useCallback } from "react";
import { motion } from "framer-motion";
import { Info, AlertTriangle, Maximize2, RefreshCcw } from "lucide-react";
import { cn } from "@/lib/utils";

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
  S_D: number;
  S_V: number;
  S_L: number;
  S_G: number;
  M_regime: string;
  M_D: number;
  M_V: number;
  M_L: number;
  M_G: number;
  M_trend_strength: number;
  M_R_trans: number;
  hmmConfidence: number;
  vaultTriggered: boolean;
  forecastUncertainty: number;
  coordinateStrategy: string;
  rankScore: number | null;
  runAt: string;
}

const PLOT_PADDING = 60;
const PLOT_SIZE = 600;
const INNER = PLOT_SIZE - PLOT_PADDING * 2;

function toPlotX(mD: number) {
  return PLOT_PADDING + (mD / 10) * INNER;
}
function toPlotY(mV: number) {
  return PLOT_PADDING + ((10 - mV) / 10) * INNER;
}

function dotColor(plan: ActivePlan): string {
  const strat = plan.strategy.toLowerCase();
  if (plan.action === "enter" && (strat.includes("bull") || strat.includes("call"))) return "#22c55e";
  if (strat.includes("bear") || strat.includes("put")) return "#ef4444";
  return "#00f5ff";
}

function dotRadius(plan: ActivePlan): number {
  const rank = plan.rankScore ?? 0;
  return Math.max(6, Math.min(18, 6 + rank * 12));
}

function transitionLabel(val: number): { text: string; color: string } {
  if (val < 0.25) return { text: "LOW", color: "text-green-400" };
  if (val < 0.5) return { text: "MODERATE", color: "text-amber-400" };
  if (val < 0.75) return { text: "HIGH", color: "text-orange-400" };
  return { text: "CRITICAL", color: "text-red-400" };
}

function FactorBar({ label, value }: { label: string; value: number }) {
  const pct = (value / 10) * 100;
  const centerPct = 50;
  const barLeft = Math.min(pct, centerPct);
  const barWidth = Math.abs(pct - centerPct);
  const isAbove = value > 5;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground font-medium">{label}</span>
        <span className="font-mono font-bold">{value.toFixed(1)}</span>
      </div>
      <div className="relative h-2 bg-secondary rounded-full overflow-hidden">
        <div className="absolute top-0 left-1/2 w-px h-full bg-white/20 z-10" />
        <div
          className={cn("absolute top-0 h-full rounded-full", isAbove ? "bg-primary" : "bg-accent")}
          style={{ left: `${barLeft}%`, width: `${barWidth}%` }}
        />
      </div>
    </div>
  );
}

export default function LocusMatrixPage() {
  const [plans, setPlans] = useState<ActivePlan[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [hoveredSymbol, setHoveredSymbol] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();
      setPlans(data.activePlans || []);
      if (!selectedSymbol && data.activePlans?.length) {
        setSelectedSymbol(data.activePlans[0].symbol);
      }
    } catch (err) {
      console.error("Failed to fetch metrics", err);
    } finally {
      setLoading(false);
    }
  }, [selectedSymbol]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const symbols = useMemo(() => [...new Set(plans.map((p) => p.symbol))], [plans]);
  const selected = useMemo(() => plans.find((p) => p.symbol === selectedSymbol), [plans, selectedSymbol]);
  const transition = useMemo(() => transitionLabel(selected?.M_R_trans ?? 0), [selected]);

  const gridLines = [0, 2, 4, 5, 6, 8, 10];

  return (
    <div className="space-y-8 max-w-7xl mx-auto h-full flex flex-col">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Regime Locus Matrix</h1>
          <p className="text-muted-foreground mt-1">
            Projecting current state into the volatility-direction vector space.
          </p>
        </div>
        <div className="flex gap-3 items-center">
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="bg-secondary border border-border rounded-xl px-4 py-2 text-sm outline-none focus:ring-2 focus:ring-primary"
          >
            {symbols.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <button
            onClick={() => {
              setLoading(true);
              fetchData();
            }}
            className="p-2 glass rounded-xl hover:bg-secondary transition-colors"
          >
            <RefreshCcw className={cn("w-5 h-5 text-muted-foreground", loading && "animate-spin")} />
          </button>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-8 min-h-[600px]">
        {/* ─── SVG Scatter Plot ─── */}
        <motion.div
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="lg:col-span-3 glass rounded-3xl p-1 relative overflow-hidden flex flex-col"
        >
          <div className="absolute inset-0 pointer-events-none opacity-[0.08]">
            <div
              className="absolute inset-0"
              style={{
                background:
                  "linear-gradient(135deg, rgba(239,68,68,0.5) 0%, rgba(59,130,246,0.3) 40%, rgba(59,130,246,0.3) 60%, rgba(34,197,94,0.5) 100%)",
              }}
            />
          </div>

          <div className="relative flex-1 m-4">
            <svg
              viewBox={`0 0 ${PLOT_SIZE} ${PLOT_SIZE}`}
              className="w-full h-full"
              preserveAspectRatio="xMidYMid meet"
            >
              <defs>
                <radialGradient id="glow-green" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#22c55e" stopOpacity={0.6} />
                  <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
                </radialGradient>
                <radialGradient id="glow-red" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.6} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                </radialGradient>
                <radialGradient id="glow-cyan" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#00f5ff" stopOpacity={0.6} />
                  <stop offset="100%" stopColor="#00f5ff" stopOpacity={0} />
                </radialGradient>
                <filter id="blur-glow">
                  <feGaussianBlur stdDeviation="3" />
                </filter>
              </defs>

              {/* Quadrant backgrounds */}
              <rect x={PLOT_PADDING} y={PLOT_PADDING} width={INNER / 2} height={INNER / 2} fill="rgba(239,68,68,0.04)" />
              <rect x={PLOT_PADDING + INNER / 2} y={PLOT_PADDING} width={INNER / 2} height={INNER / 2} fill="rgba(34,197,94,0.04)" />
              <rect x={PLOT_PADDING} y={PLOT_PADDING + INNER / 2} width={INNER / 2} height={INNER / 2} fill="rgba(59,130,246,0.03)" />
              <rect x={PLOT_PADDING + INNER / 2} y={PLOT_PADDING + INNER / 2} width={INNER / 2} height={INNER / 2} fill="rgba(59,130,246,0.03)" />

              {/* Grid lines */}
              {gridLines.map((v) => (
                <React.Fragment key={v}>
                  <line
                    x1={toPlotX(v)}
                    y1={PLOT_PADDING}
                    x2={toPlotX(v)}
                    y2={PLOT_SIZE - PLOT_PADDING}
                    stroke={v === 5 ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.04)"}
                    strokeWidth={v === 5 ? 1.5 : 0.5}
                  />
                  <line
                    x1={PLOT_PADDING}
                    y1={toPlotY(v)}
                    x2={PLOT_SIZE - PLOT_PADDING}
                    y2={toPlotY(v)}
                    stroke={v === 5 ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.04)"}
                    strokeWidth={v === 5 ? 1.5 : 0.5}
                  />
                </React.Fragment>
              ))}

              {/* Axis labels */}
              <text x={PLOT_SIZE / 2} y={PLOT_PADDING - 16} textAnchor="middle" fill="#94a3b8" fontSize={11} fontWeight={600}>
                HIGH VOLATILITY
              </text>
              <text x={PLOT_SIZE / 2} y={PLOT_SIZE - PLOT_PADDING + 28} textAnchor="middle" fill="#94a3b8" fontSize={11} fontWeight={600}>
                LOW VOLATILITY
              </text>
              <text x={PLOT_PADDING - 16} y={PLOT_SIZE / 2} textAnchor="middle" fill="#94a3b8" fontSize={11} fontWeight={600} transform={`rotate(-90, ${PLOT_PADDING - 16}, ${PLOT_SIZE / 2})`}>
                BEARISH
              </text>
              <text x={PLOT_SIZE - PLOT_PADDING + 16} y={PLOT_SIZE / 2} textAnchor="middle" fill="#94a3b8" fontSize={11} fontWeight={600} transform={`rotate(90, ${PLOT_SIZE - PLOT_PADDING + 16}, ${PLOT_SIZE / 2})`}>
                BULLISH
              </text>

              {/* Tick labels along axes */}
              {[0, 2, 4, 6, 8, 10].map((v) => (
                <React.Fragment key={`tick-${v}`}>
                  <text x={toPlotX(v)} y={PLOT_SIZE - PLOT_PADDING + 14} textAnchor="middle" fill="#64748b" fontSize={9}>
                    {v}
                  </text>
                  <text x={PLOT_PADDING - 8} y={toPlotY(v) + 3} textAnchor="end" fill="#64748b" fontSize={9}>
                    {v}
                  </text>
                </React.Fragment>
              ))}

              {/* Data points */}
              {plans.map((plan, i) => {
                const cx = toPlotX(plan.M_D);
                const cy = toPlotY(plan.M_V);
                const r = dotRadius(plan);
                const color = dotColor(plan);
                const isActive = plan.action === "enter";
                const isSkipped = plan.status === "skipped" || plan.action === "skip";
                const isHovered = hoveredSymbol === plan.symbol;
                const isSelected = plan.symbol === selectedSymbol;

                return (
                  <motion.g
                    key={plan.symbol}
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.05, type: "spring", stiffness: 200 }}
                    onMouseEnter={() => setHoveredSymbol(plan.symbol)}
                    onMouseLeave={() => setHoveredSymbol(null)}
                    onClick={() => setSelectedSymbol(plan.symbol)}
                    style={{ cursor: "pointer" }}
                  >
                    {/* Pulsing glow for active plans */}
                    {isActive && (
                      <circle cx={cx} cy={cy} r={r * 2.5} fill={color} opacity={0.12} filter="url(#blur-glow)">
                        <animate attributeName="r" values={`${r * 2};${r * 3};${r * 2}`} dur="2s" repeatCount="indefinite" />
                        <animate attributeName="opacity" values="0.12;0.06;0.12" dur="2s" repeatCount="indefinite" />
                      </circle>
                    )}

                    <circle
                      cx={cx}
                      cy={cy}
                      r={r}
                      fill={color}
                      opacity={isSkipped ? 0.3 : 0.85}
                      stroke={isSelected ? "#fff" : isHovered ? "rgba(255,255,255,0.5)" : "none"}
                      strokeWidth={isSelected ? 2 : 1}
                    />

                    <text
                      x={cx}
                      y={cy - r - 5}
                      textAnchor="middle"
                      fill={isSkipped ? "#64748b" : "#ececf1"}
                      fontSize={isSelected ? 13 : 11}
                      fontWeight={isSelected ? 700 : 500}
                    >
                      {plan.symbol}
                    </text>

                    {/* Tooltip on hover */}
                    {isHovered && (
                      <g>
                        <rect
                          x={cx + r + 6}
                          y={cy - 32}
                          width={140}
                          height={56}
                          rx={8}
                          fill="rgba(17,17,20,0.95)"
                          stroke="rgba(255,255,255,0.1)"
                          strokeWidth={1}
                        />
                        <text x={cx + r + 14} y={cy - 16} fill="#ececf1" fontSize={10} fontWeight={600}>
                          {plan.strategy || "—"}
                        </text>
                        <text x={cx + r + 14} y={cy - 2} fill="#94a3b8" fontSize={9}>
                          D:{plan.M_D.toFixed(1)} V:{plan.M_V.toFixed(1)} L:{plan.M_L.toFixed(1)} G:{plan.M_G.toFixed(1)}
                        </text>
                        <text x={cx + r + 14} y={cy + 14} fill={color} fontSize={9} fontWeight={600}>
                          {plan.action.toUpperCase()} — conf {(plan.confidence * 100).toFixed(0)}%
                        </text>
                      </g>
                    )}
                  </motion.g>
                );
              })}
            </svg>
          </div>

          <div className="px-6 pb-4 flex justify-between items-center text-xs text-muted-foreground">
            <div className="flex gap-6">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-green-500" /> Bullish Entry
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500" /> Bearish / Put
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-[#00f5ff]" /> Neutral
              </span>
            </div>
            <span className="italic">Auto-refresh: 15s</span>
          </div>
        </motion.div>

        {/* ─── Right Sidebar ─── */}
        <div className="space-y-6">
          {/* Regime Analysis */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="glass rounded-3xl p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <Info className="w-4 h-4 text-primary" />
              <h3 className="font-bold">Regime Analysis</h3>
            </div>
            {selected ? (
              <div className="p-4 bg-secondary/50 rounded-2xl border border-border space-y-2">
                <p className="text-xs text-muted-foreground uppercase tracking-widest font-bold">Current State</p>
                <p className="text-lg font-bold neon-text">{selected.M_regime || "Unknown"}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {selected.symbol} @ ${selected.close.toFixed(2)} — {selected.strategy || "No strategy"}
                </p>
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-xs text-muted-foreground">Confidence:</span>
                  <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary rounded-full"
                      style={{ width: `${selected.regimeConfidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono font-bold">{(selected.regimeConfidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No symbol selected</p>
            )}
          </motion.div>

          {/* Transition Risk */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="glass rounded-3xl p-6"
          >
            <div className="flex items-center gap-2 mb-4">
              <AlertTriangle className="w-4 h-4 text-accent" />
              <h3 className="font-bold">Transition Risk</h3>
            </div>
            {selected ? (
              <div className="p-4 bg-secondary/30 rounded-2xl border border-border">
                <p className={cn("text-2xl font-bold", transition.color)}>{transition.text}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  M_R_trans: {selected.M_R_trans.toFixed(3)}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Trend strength: {selected.M_trend_strength.toFixed(2)}
                </p>
                {selected.vaultTriggered && (
                  <span className="inline-block mt-2 text-xs font-bold bg-red-500/20 text-red-400 px-2 py-0.5 rounded-lg">
                    VAULT TRIGGERED
                  </span>
                )}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">—</p>
            )}
          </motion.div>

          {/* Factor Breakdown */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="glass rounded-3xl p-6"
          >
            <h3 className="font-bold mb-4">Factor Breakdown</h3>
            {selected ? (
              <div className="space-y-3">
                <FactorBar label="M_D — Direction" value={selected.M_D} />
                <FactorBar label="M_V — Volatility" value={selected.M_V} />
                <FactorBar label="M_L — Liquidity" value={selected.M_L} />
                <FactorBar label="M_G — Gamma" value={selected.M_G} />
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">—</p>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
