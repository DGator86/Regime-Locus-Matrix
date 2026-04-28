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
  return minutes >= 570 && minutes < 960; // 9:30–16:00
}

export default function StatusBar() {
  const [metrics, setMetrics] = useState<any>(null);
  const [marketOpen, setMarketOpen] = useState(isMarketOpen);

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
    const clockInterval = setInterval(() => setMarketOpen(isMarketOpen()), 30000);

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

  return (
    <div
      className="h-10 flex items-center justify-between px-4 text-xs border-t border-border shrink-0 select-none"
      style={{ backgroundColor: "#111114" }}
    >
      {/* Left */}
      <span className="text-muted-foreground font-mono">RLM v2.4</span>

      {/* Center */}
      <div className="flex items-center gap-5 text-muted-foreground">
        {spyPrice != null && (
          <span className="flex items-center gap-1.5">
            <span className="font-semibold text-foreground">SPY</span>
            <span className="font-mono">${spyPrice.toFixed(2)}</span>
            {spyChange != null && spyPct != null && (
              <span
                className={cn(
                  "font-mono",
                  spyChange >= 0 ? "text-green-400" : "text-red-400",
                )}
              >
                {spyChange >= 0 ? "+" : ""}
                {spyChange.toFixed(2)} ({spyPct.toFixed(2)}%)
              </span>
            )}
          </span>
        )}

        <span className="flex items-center gap-1.5">
          <span
            className={cn(
              "w-1.5 h-1.5 rounded-full",
              statusOk ? "bg-green-400 animate-pulse"
                : statusWarn ? "bg-amber-400"
                : "bg-red-400",
            )}
          />
          <span
            className={cn(
              statusOk ? "text-green-400"
                : statusWarn ? "text-amber-400"
                : "text-red-400",
            )}
          >
            {status}
          </span>
        </span>

        <span>
          Updated <span className="font-mono text-foreground">{generated}</span> ET
        </span>

        <span>
          <span className="font-mono text-foreground">{planCount}</span> plans
        </span>
      </div>

      {/* Right */}
      <span
        className={cn(
          "flex items-center gap-1.5 font-semibold",
          marketOpen ? "text-green-400" : "text-red-400",
        )}
      >
        <span
          className={cn(
            "w-1.5 h-1.5 rounded-full",
            marketOpen ? "bg-green-400 animate-pulse" : "bg-red-400",
          )}
        />
        {marketOpen ? "MARKET OPEN" : "MARKET CLOSED"}
      </span>
    </div>
  );
}
