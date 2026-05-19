"use client";

import React, { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { Database, LineChart, RefreshCw } from "lucide-react";

type FeedsPayload = {
  generatedAt?: string;
  stockSource?: string;
  eodhd?: {
    pollWindow?: string;
    pollIntervalSec?: number;
    collectorActive?: boolean;
    lastSymbol?: string;
    symbols?: {
      symbol: string;
      parquetMtime: string | null;
      lastPollEt: string | null;
      parquetExists: boolean;
    }[];
  };
  options?: {
    provider?: string;
    tradeLogMtime?: string | null;
    universePlansMtime?: string | null;
  };
  pipeline?: {
    primaryBarSize?: string;
    primaryDuration?: string;
    plansGeneratedAt?: string;
    activeCount?: number;
    symbolCount?: number;
  };
};

function formatAge(iso: string | null | undefined): string {
  if (!iso) return "â€”";
  const ts = new Date(iso).getTime();
  if (!Number.isFinite(ts)) return "â€”";
  const mins = Math.floor((Date.now() - ts) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const h = Math.floor(mins / 60);
  if (h < 48) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function formatEtShort(iso: string | null | undefined): string {
  if (!iso) return "â€”";
  try {
    return new Date(iso).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "America/New_York",
    });
  } catch {
    return "â€”";
  }
}

export default function DataFeedsPanel() {
  const [data, setData] = useState<FeedsPayload | null>(null);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    fetch("/api/data-feeds", { cache: "no-store" })
      .then((r) => r.json())
      .then((j) => setData(j))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  const eodhd = data?.eodhd;
  const stocks = eodhd?.symbols ?? [];
  const withLake = stocks.filter((s) => s.parquetExists).length;
  const freshPolls = stocks.filter((s) => {
    if (!s.lastPollEt) return false;
    const ts = new Date(s.lastPollEt).getTime();
    return Date.now() - ts < 10 * 60 * 1000;
  }).length;

  return (
    <div className="shrink-0 border-b border-cyan-500/20 bg-cyan-950/20 px-4 py-3">
      <div className="max-w-[1920px] mx-auto space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-cyan-200">
            <Database className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-semibold uppercase tracking-wider">Data feeds</span>
            {loading && <RefreshCw className="w-3 h-3 animate-spin text-cyan-500/70" />}
          </div>
          <button
            type="button"
            onClick={load}
            className="text-[10px] uppercase tracking-wide text-cyan-400/80 hover:text-cyan-300"
          >
            Refresh
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 text-[11px]">
          <div className="rounded-lg border border-cyan-500/15 bg-black/30 p-3 space-y-2">
            <p className="font-semibold text-cyan-300 flex items-center gap-1.5">
              <LineChart className="w-3.5 h-3.5" />
              Stocks / ETFs (EODHD 1m)
            </p>
            <p className="text-slate-400">
              Source: <span className="text-slate-200">{data?.stockSource || "eodhd"}</span> Â· Window{" "}
              {eodhd?.pollWindow || "09:00â€“16:30 ET"} Â· {eodhd?.pollIntervalSec ?? 5}s rotation
            </p>
            <p className="text-slate-400">
              Lake:{" "}
              <span className={cn(withLake >= 10 ? "text-emerald-400" : "text-amber-400")}>
                {withLake}/{stocks.length}
              </span>{" "}
              symbols Â· Polls &lt;10m:{" "}
              <span className={cn(freshPolls >= 8 ? "text-emerald-400" : "text-amber-400")}>
                {freshPolls}
              </span>
              {eodhd?.lastSymbol ? (
                <span className="text-slate-500"> Â· last {eodhd.lastSymbol}</span>
              ) : null}
            </p>
            <div className="flex flex-wrap gap-1.5 max-h-16 overflow-y-auto">
              {stocks.slice(0, 12).map((s) => (
                <span
                  key={s.symbol}
                  className={cn(
                    "font-mono px-1.5 py-0.5 rounded border text-[10px]",
                    s.parquetExists
                      ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-200"
                      : "border-rose-500/25 bg-rose-500/10 text-rose-200",
                  )}
                  title={`poll ${formatEtShort(s.lastPollEt)} ET Â· lake ${formatAge(s.parquetMtime)}`}
                >
                  {s.symbol}
                </span>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-violet-500/15 bg-black/30 p-3 space-y-2">
            <p className="font-semibold text-violet-300">Options (Massive)</p>
            <p className="text-slate-400">
              Chains / marks via Massive API. Trade log updated{" "}
              <span className="text-slate-200">{formatAge(data?.options?.tradeLogMtime)}</span>
            </p>
            <p className="text-slate-400">
              Universe plans{" "}
              <span className="text-slate-200">{formatAge(data?.options?.universePlansMtime)}</span>
            </p>
          </div>

          <div className="rounded-lg border border-white/10 bg-black/30 p-3 space-y-2">
            <p className="font-semibold text-slate-200">Regime pipeline</p>
            <p className="text-slate-400">
              Bars:{" "}
              <span className="text-cyan-200 font-mono">{data?.pipeline?.primaryBarSize || "â€”"}</span>
              {data?.pipeline?.primaryDuration ? (
                <span className="text-slate-500"> Â· {data.pipeline.primaryDuration}</span>
              ) : null}
            </p>
            <p className="text-slate-400">
              Plans: {data?.pipeline?.symbolCount ?? 0} symbols Â· {data?.pipeline?.activeCount ?? 0}{" "}
              active Â· {formatAge(data?.pipeline?.plansGeneratedAt)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
