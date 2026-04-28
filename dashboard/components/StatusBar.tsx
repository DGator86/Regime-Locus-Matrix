"use client";

import React, { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

function isMarketOpen(): boolean {
  const now = new Date();
  const et = new Date(
    now.toLocaleString("en-US", { timeZone: "America/New_York" }),
  );
  const day = et.getDay();
  if (day === 0 || day === 6) return false;
  const minutes = et.getHours() * 60 + et.getMinutes();
  return minutes >= 570 && minutes < 960;
}

type TickSpec = { label: string; price?: number | null; pct?: number | null };

export default function StatusBar() {
  const [metrics, setMetrics] = useState<any>(null);
  const [marketOpen, setMarketOpen] = useState(isMarketOpen);
  const [nowEt, setNowEt] = useState("");

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch("/api/metrics");
        const data = await res.json();
        setMetrics(data);
      } catch {
        // silent
      }
    };

    fetchMetrics();
    const dataInterval = setInterval(fetchMetrics, 15000);
    const clockInterval = setInterval(() => {
      setMarketOpen(isMarketOpen());
      setNowEt(
        new Date().toLocaleTimeString("en-US", {
          hour: "numeric",
          minute: "2-digit",
          second: "2-digit",
          hour12: true,
          timeZone: "America/New_York",
        }),
      );
    }, 1000);

    setNowEt(
      new Date().toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
        timeZone: "America/New_York",
      }),
    );

    return () => {
      clearInterval(dataInterval);
      clearInterval(clockInterval);
    };
  }, []);

  const ts = metrics?.forecastTimeseries ?? [];
  const latest = ts.length > 0 ? ts[ts.length - 1] : null;
  const prev = ts.length > 1 ? ts[ts.length - 2] : null;

  const spyPrice = latest?.close ?? null;
  const spyChange = latest && prev ? latest.close - prev.close : null;
  const spyPct =
    spyChange != null && prev?.close
      ? (spyChange / prev.close) * 100
      : null;

  const status = metrics?.marketState?.status || "—";
  const generated = metrics?.generatedAt
    ? new Date(metrics.generatedAt).toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "America/New_York",
      })
    : "—";
  const planCount = metrics?.activePlans?.length ?? 0;

  const statusLower = status.toLowerCase();
  const statusOk =
    statusLower.includes("operational") ||
    statusLower.includes("open") ||
    statusLower.includes("healthy") ||
    status === "OK";
  const statusWarn =
    statusLower.includes("degraded") ||
    statusLower.includes("warning") ||
    statusLower.includes("caution");

  const ticks: TickSpec[] = [
    { label: "SPY", price: spyPrice, pct: spyPct },
    { label: "QQQ", price: null, pct: null },
    { label: "IWM", price: null, pct: null },
    { label: "VIX", price: null, pct: null },
    { label: "10Y", price: null, pct: null },
    { label: "DXY", price: null, pct: null },
  ];

  return (
    <div
      className="min-h-10 shrink-0 border-t border-white/[0.06] bg-[#07090d]/95 backdrop-blur-md px-3 py-2 text-[11px] select-none font-[family-name:var(--font-mono)]"
      style={{ boxShadow: "inset 0 1px 0 rgba(255,255,255,0.04)" }}
    >
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 max-w-[1920px] mx-auto">
        <span className="text-slate-600 shrink-0">RLM · dashboard</span>

        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 justify-center flex-1 min-w-0 text-slate-400">
          {ticks.map((t) => (
            <span key={t.label} className="flex items-center gap-2 whitespace-nowrap">
              <span className="font-semibold text-slate-300">{t.label}</span>
              {t.price != null ? (
                <>
                  <span className="text-slate-200">${t.price.toFixed(2)}</span>
                  {t.pct != null && (
                    <span
                      className={cn(
                        (t.pct ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400",
                      )}
                    >
                      {(t.pct ?? 0) >= 0 ? "+" : ""}
                      {(t.pct ?? 0).toFixed(2)}%
                    </span>
                  )}
                </>
              ) : (
                <span className="text-slate-600">—</span>
              )}
            </span>
          ))}
        </div>

        <div className="flex flex-wrap items-center justify-end gap-x-4 gap-y-1 text-slate-500 shrink-0">
          <span className="flex items-center gap-1.5">
            <span
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                statusOk
                  ? "bg-emerald-400 animate-pulse"
                  : statusWarn
                    ? "bg-amber-400"
                    : "bg-rose-400",
              )}
            />
            <span
              className={cn(
                statusOk
                  ? "text-emerald-400"
                  : statusWarn
                    ? "text-amber-400"
                    : "text-rose-400",
              )}
            >
              {status}
            </span>
          </span>

          <span className="hidden sm:inline">
            Upd{" "}
            <span className="text-slate-300">{generated}</span> ET ·{" "}
            <span className="text-slate-300">{planCount}</span> plans
          </span>

          <span
            className={cn(
              "flex items-center gap-1.5 font-semibold",
              marketOpen ? "text-emerald-400" : "text-rose-400",
            )}
          >
            <span
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                marketOpen ? "bg-emerald-400 animate-pulse" : "bg-rose-400",
              )}
            />
            {marketOpen ? "MARKET OPEN" : "MARKET CLOSED"}
            <span className="text-slate-500 font-normal ml-1">{nowEt}</span>
          </span>
        </div>
      </div>
    </div>
  );
}
