"use client";

import React, { useMemo } from "react";
import { cn } from "@/lib/utils";

const ROW_LABELS = ["J", "I", "H", "G", "F", "E", "D", "C", "B", "A"];

export interface HeatPoint {
  sD: number;
  sV: number;
}

function clampIdx(v: number): number {
  const n = Math.round(v);
  return Math.min(9, Math.max(0, n));
}

/** Map S_V (0–10): top row = high vol */
function rowFromVol(sV: number): number {
  return clampIdx(Math.round(10 - sV));
}

/** Map S_D (0–10): left = bearish, right = bullish */
function colFromDir(sD: number): number {
  return clampIdx(Math.round(sD));
}

function cellColor(sd: number, sv: number): string {
  const t = (sd + sv) / 20;
  if (sd >= 6 && sv <= 5) return `rgba(34,197,94,${0.15 + t * 0.15})`;
  if (sd <= 4 && sv >= 5) return `rgba(239,68,68,${0.12 + (1 - t) * 0.12})`;
  return `rgba(139,92,246,${0.08 + t * 0.1})`;
}

export default function RegimeHeatmap({
  path,
  title = "Regime Locus Matrix",
  subtitle = "Volatility × Direction · last 20 bars",
  className,
}: {
  path: HeatPoint[];
  title?: string;
  subtitle?: string;
  className?: string;
}) {
  const pts = useMemo(() => {
    const slice = path.slice(-20);
    return slice.map((p) => ({
      row: rowFromVol(p.sV),
      col: colFromDir(p.sD),
      ...p,
    }));
  }, [path]);

  const current = pts.length ? pts[pts.length - 1] : null;

  const polylinePoints = useMemo(() => {
    return pts.map((p) => {
      const x = (p.col + 0.5) * (100 / 10);
      const y = (p.row + 0.5) * (100 / 10);
      return `${x},${y}`;
    }).join(" ");
  }, [pts]);

  return (
    <div
      className={cn(
        "relative rounded-2xl border border-cyan-500/15 bg-[#07090e]/90 shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_0_40px_rgba(34,211,238,0.06)] overflow-hidden flex flex-col min-h-[340px]",
        className,
      )}
    >
      <div className="absolute inset-0 pointer-events-none opacity-[0.07] bg-[radial-gradient(ellipse_at_top,_rgba(34,211,238,0.5),transparent_55%)]" />

      <div className="relative px-5 pt-4 pb-2 flex justify-between items-start gap-3">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-[0.15em] text-cyan-400/95 neon-text-soft">
            {title}
          </h3>
          <p className="text-[11px] text-slate-500 mt-1">{subtitle}</p>
        </div>
        {current && (
          <div className="text-right font-mono text-[10px] text-slate-500">
            <span className="text-emerald-400/90">●</span> Current cell{" "}
            <span className="text-foreground font-semibold">
              {ROW_LABELS[current.row]}
              {current.col + 1}
            </span>
          </div>
        )}
      </div>

      <div className="relative flex-1 px-4 pb-4 pt-1 flex gap-2">
        {/* Y labels */}
        <div className="flex flex-col justify-between py-6 text-[9px] font-mono text-slate-600 w-5 shrink-0 select-none">
          <span className="text-slate-400">HIGH</span>
          <span className="rotate-[-90deg] origin-center whitespace-nowrap text-[8px] tracking-widest opacity-70">
            VOL
          </span>
          <span className="text-slate-400">LOW</span>
        </div>

        <div className="flex-1 min-w-0">
          {/* Column numbers */}
          <div className="grid grid-cols-10 gap-px mb-1 px-0.5">
            {Array.from({ length: 10 }, (_, i) => (
              <div
                key={`cn-${i}`}
                className="text-center text-[9px] font-mono text-slate-600"
              >
                {i + 1}
              </div>
            ))}
          </div>

          <div className="relative aspect-square max-h-[min(52vw,420px)] w-full mx-auto">
            {/* Grid */}
            <div className="absolute inset-0 grid grid-cols-10 grid-rows-10 gap-px rounded-lg overflow-hidden border border-white/[0.07] bg-black/50">
              {Array.from({ length: 100 }).map((_, i) => {
                const row = Math.floor(i / 10);
                const col = i % 10;
                const fakeSd = col + 0.5;
                const fakeSv = 10 - row - 0.5;
                return (
                  <div
                    key={i}
                    className="relative border border-white/[0.04]"
                    style={{ backgroundColor: cellColor(fakeSd, fakeSv) }}
                  />
                );
              })}
            </div>

            {/* SVG path overlay */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              <defs>
                <filter id="hm-glow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="0.35" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              {polylinePoints && (
                <polyline
                  fill="none"
                  stroke="rgba(255,255,255,0.85)"
                  strokeWidth="0.35"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={polylinePoints}
                  filter="url(#hm-glow)"
                  vectorEffect="non-scaling-stroke"
                />
              )}
              {pts.map((p, idx) => {
                const cx = (p.col + 0.5) * 10;
                const cy = (p.row + 0.5) * 10;
                const isLast = idx === pts.length - 1;
                const hue = idx / Math.max(pts.length - 1, 1);
                const stroke = `hsla(${280 - hue * 120}, 85%, ${60 + hue * 15}%, 0.95)`;
                return (
                  <circle
                    key={`${p.row}-${p.col}-${idx}`}
                    cx={cx}
                    cy={cy}
                    r={isLast ? 1.35 : 0.55}
                    fill={stroke}
                    stroke={isLast ? "#fff" : "none"}
                    strokeWidth={isLast ? 0.25 : 0}
                  />
                );
              })}
            </svg>
          </div>

          <div className="flex justify-between mt-2 px-1 text-[9px] font-mono text-slate-500 uppercase tracking-wide">
            <span>Bearish</span>
            <span>Direction</span>
            <span>Bullish</span>
          </div>
        </div>

        {/* Row letters */}
        <div className="flex flex-col justify-between py-6 text-[9px] font-mono text-slate-600 w-4 shrink-0 select-none text-center">
          {ROW_LABELS.map((L) => (
            <span key={L}>{L}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
