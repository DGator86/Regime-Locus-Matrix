"use client";

import React, { useEffect, useState } from "react";

interface DataMeta {
  processedDir: string;
  resolved: boolean;
  resolutionSource: string;
  hasForecastFeaturesCsv: boolean;
  hasUniversePlansJson: boolean;
  checkedPathsSample?: string[];
}

export default function DataDirBanner() {
  const [meta, setMeta] = useState<DataMeta | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/metrics")
      .then((r) => r.json())
      .then((j) => {
        if (!cancelled && j?.dataMeta) setMeta(j.dataMeta);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  if (!meta) return null;

  const missingDir = !meta.resolved;
  const missingForecast = meta.resolved && !meta.hasForecastFeaturesCsv;
  const missingPlans = meta.resolved && !meta.hasUniversePlansJson;

  if (!missingDir && !missingForecast && !missingPlans) return null;

  let msg = "";
  if (missingDir) {
    msg =
      "Processed data folder was not found. From repo root run pipelines so data/processed exists, or set RLM_DATA_DIR (or RLM_DATA_ROOT) in dashboard/.env.local pointing at your processed outputs.";
  } else {
    const parts: string[] = [];
    if (missingForecast) parts.push("forecast_features_SPY.csv (run forecast pipeline)");
    if (missingPlans) parts.push("universe_trade_plans.json (trade plans)");
    msg = `Reading ${meta.processedDir} — missing: ${parts.join("; ")}.`;
  }

  return (
    <div className="shrink-0 border-b border-amber-500/35 bg-amber-950/50 px-4 py-2.5 text-[11px] sm:text-xs text-amber-100/95 leading-snug">
      <span className="font-semibold text-amber-300">Data:</span>{" "}
      <span className="text-amber-50/95">{msg}</span>
      <span className="block mt-1 font-mono text-[10px] text-amber-200/70 break-all">
        Dir: {meta.processedDir} ({meta.resolutionSource})
      </span>
    </div>
  );
}
