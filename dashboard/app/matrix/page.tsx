"use client";

import React from "react";
import { motion } from "framer-motion";
import { Info, Maximize2 } from "lucide-react";

const matrixData = [
  { x: 0.2, y: 0.8, label: "SPY", time: "Now" },
  { x: 0.1, y: 0.7, label: "SPY", time: "1h ago" },
  { x: -0.1, y: 0.5, label: "SPY", time: "2h ago" },
  { x: -0.3, y: 0.2, label: "SPY", time: "3h ago" },
  { x: 0.6, y: -0.4, label: "TSLA", time: "Now" },
];

export default function LocusMatrixPage() {
  return (
    <div className="space-y-8 max-w-7xl mx-auto h-full flex flex-col">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Regime Locus Matrix</h1>
          <p className="text-muted-foreground mt-1">Projecting current state into the volatility-direction vector space.</p>
        </div>
        <div className="flex gap-3">
          <select className="bg-secondary border border-border rounded-xl px-4 py-2 text-sm outline-none focus:ring-2 focus:ring-primary">
            <option>SPY (Default)</option>
            <option>QQQ</option>
            <option>TSLA</option>
          </select>
          <button className="p-2 glass rounded-xl hover:bg-secondary">
            <Maximize2 className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-8 min-h-[600px]">
        <div className="lg:col-span-3 glass rounded-3xl p-1 relative overflow-hidden flex flex-col">
          {/* Background Gradient Map */}
          <div className="absolute inset-0 opacity-20 pointer-events-none">
            <div className="absolute inset-0 bg-gradient-to-tr from-red-500 via-blue-500 to-green-500" />
            <div className="absolute inset-0 backdrop-blur-[100px]" />
          </div>

          <div className="relative flex-1 m-8 border border-white/10 rounded-2xl overflow-hidden bg-black/40">
            {/* Axis Labels */}
            <div className="absolute left-1/2 top-4 -translate-x-1/2 text-xs font-bold text-muted-foreground tracking-widest uppercase">High Volatility</div>
            <div className="absolute left-1/2 bottom-4 -translate-x-1/2 text-xs font-bold text-muted-foreground tracking-widest uppercase">Low Volatility</div>
            <div className="absolute left-4 top-1/2 -translate-y-1/2 text-xs font-bold text-muted-foreground tracking-widest uppercase -rotate-90">Bearish Flow</div>
            <div className="absolute right-4 top-1/2 -translate-y-1/2 text-xs font-bold text-muted-foreground tracking-widest uppercase rotate-90">Bullish Flow</div>

            {/* Grid Lines */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-px h-full bg-white/10" />
              <div className="h-px w-full bg-white/10" />
            </div>

            {/* Matrix Content */}
            <svg viewBox="-1 -1 2 2" className="w-full h-full p-12">
              {/* Path line */}
              <motion.path
                d={`M ${matrixData.filter(d => d.label === "SPY").map(d => `${d.x} ${-d.y}`).join(" L ")}`}
                fill="none"
                stroke="rgba(0, 245, 255, 0.4)"
                strokeWidth="0.02"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 2 }}
              />

              {/* Data points */}
              {matrixData.map((point, i) => (
                <motion.g
                  key={i}
                  initial={{ opacity: 0, scale: 0 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.1 }}
                >
                  <circle
                    cx={point.x}
                    cy={-point.y}
                    r={i === 0 || i === 4 ? 0.04 : 0.02}
                    fill={point.label === "SPY" ? "#00f5ff" : "#a855f7"}
                    className={i === 0 || i === 4 ? "animate-pulse" : ""}
                    filter="blur(1px)"
                  />
                  {(i === 0 || i === 4) && (
                    <text
                      x={point.x + 0.06}
                      y={-point.y - 0.06}
                      fontSize="0.08"
                      fill="white"
                      fontWeight="bold"
                      className="select-none"
                    >
                      {point.label}
                    </text>
                  )}
                </motion.g>
              ))}
            </svg>
          </div>

          <div className="px-8 pb-8 flex justify-between items-center">
            <div className="flex gap-8">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#00f5ff] neon-border" />
                <span className="text-sm font-medium">Index Flow (SPY)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-[#a855f7]" />
                <span className="text-sm font-medium">Equities (TSLA)</span>
              </div>
            </div>
            <p className="text-xs text-muted-foreground italic">Update frequency: 60s</p>
          </div>
        </div>

        <div className="space-y-6">
          <div className="glass rounded-3xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Info className="w-4 h-4 text-primary" />
              <h3 className="font-bold">Regime Analysis</h3>
            </div>
            <div className="space-y-4">
              <div className="p-4 bg-secondary/50 rounded-2xl border border-border">
                <p className="text-xs text-muted-foreground uppercase tracking-widest font-bold">Current State</p>
                <p className="text-lg font-bold text-primary mt-1">Bullish Volatility</p>
                <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
                  Price is expanding with high directional flow. Confidence in this regime is 84%.
                </p>
              </div>
              <div className="p-4 bg-secondary/30 rounded-2xl border border-border">
                <p className="text-xs text-muted-foreground uppercase tracking-widest font-bold">Transition Risk</p>
                <p className="text-lg font-bold text-amber-400 mt-1">Moderate</p>
                <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
                  Signs of exhaustion in S_L (Liquidity) factor. Watch for range contraction.
                </p>
              </div>
            </div>
          </div>

          <div className="glass rounded-3xl p-6 flex-1 overflow-y-auto">
            <h3 className="font-bold mb-4">Transition History</h3>
            <div className="space-y-4">
              {[
                { time: "14:20", from: "Stable", to: "Expansion", type: "UP" },
                { time: "12:15", from: "Range", to: "Stable", type: "UP" },
                { time: "10:05", from: "Contraction", to: "Range", type: "UP" },
                { time: "Yesterday", from: "Crash", to: "Contraction", type: "DOWN" },
              ].map((item, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className={`w-1.5 h-10 rounded-full ${item.type === "UP" ? "bg-green-500" : "bg-red-500"}`} />
                  <div>
                    <p className="text-xs text-muted-foreground">{item.time}</p>
                    <p className="text-sm font-medium">{item.from} → {item.to}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
