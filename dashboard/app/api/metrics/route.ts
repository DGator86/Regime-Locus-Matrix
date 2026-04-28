import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import {
  optionalNum,
  parseHmmStateIndex,
  sanitizeHmmStateLabel,
} from "@/lib/hmmDisplay";

export const dynamic = "force-dynamic";

/**
 * Resolve processed outputs directory.
 * - Prefer `RLM_DATA_DIR` when set.
 * - Else `RLM_DATA_ROOT/processed` (matches Python `RLM_DATA_ROOT`).
 * - Else repo `data/processed` when Next cwd is `dashboard/` or repo root.
 * - Fallback VPS path used only when nothing else exists.
 */
function resolveProcessedDir(): {
  dir: string;
  found: boolean;
  tried: string[];
  source: string;
} {
  const tried: string[] = [];
  const candidates: { label: string; abs: string }[] = [];

  const envDir = process.env.RLM_DATA_DIR?.trim();
  if (envDir) {
    candidates.push({ label: "RLM_DATA_DIR", abs: path.resolve(envDir) });
  }

  const dataRoot = process.env.RLM_DATA_ROOT?.trim();
  if (dataRoot) {
    candidates.push({
      label: "RLM_DATA_ROOT/processed",
      abs: path.resolve(path.join(dataRoot, "processed")),
    });
  }

  const cwd = process.cwd();
  candidates.push(
    { label: "cwd/../data/processed", abs: path.resolve(cwd, "..", "data", "processed") },
    { label: "cwd/data/processed", abs: path.resolve(cwd, "data", "processed") },
    { label: "VPS default", abs: "/opt/Regime-Locus-Matrix/data/processed" },
  );

  for (const { label, abs } of candidates) {
    tried.push(abs);
    try {
      if (fs.existsSync(abs) && fs.statSync(abs).isDirectory()) {
        return { dir: abs, found: true, tried, source: label };
      }
    } catch {
      /* ignore */
    }
  }

  const fallback =
    envDir ? path.resolve(envDir)
    : dataRoot ? path.resolve(path.join(dataRoot, "processed"))
    : path.resolve(cwd, "..", "data", "processed");

  return {
    dir: fallback,
    found: false,
    tried,
    source: "fallback-unreadable",
  };
}

function parseCsv(text: string): Record<string, string>[] {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = (vals[i] ?? "").trim();
    });
    return row;
  });
}

function readJson(filePath: string): any | null {
  if (!fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readCsvFile(filePath: string): Record<string, string>[] {
  if (!fs.existsSync(filePath)) return [];
  return parseCsv(fs.readFileSync(filePath, "utf8"));
}

function num(v: any): number {
  const n = parseFloat(v);
  return isNaN(n) ? 0 : n;
}

function buildMarketState(dataDir: string) {
  const gate = readJson(path.join(dataDir, "gate_state.json"));
  if (!gate) return { posture: "UNKNOWN", status: "UNKNOWN", lastUpdated: "" };
  return {
    posture: gate.posture || gate.gate_posture || "UNKNOWN",
    status: gate.status || gate.gate_status || "UNKNOWN",
    lastUpdated: gate.last_updated || gate.timestamp || "",
  };
}

function buildActivePlans(dataDir: string) {
  const plans = readJson(path.join(dataDir, "universe_trade_plans.json"));
  if (!plans) return { activePlans: [], topRanked: [], symbolsInUniverse: [] };

  const results: any[] = plans.results || [];
  const symbols = [...new Set(results.map((r: any) => r.symbol).filter(Boolean))];

  const activeRanked: string[] = (plans.active_ranked || []).map((r: any) =>
    typeof r === "string" ? r : r.symbol || ""
  );

  const activePlans = results.map((r: any) => {
    const decision = r.decision || {};
    const pipeline = r.pipeline || {};
    const meta = decision.metadata || {};

    return {
      symbol: r.symbol || "",
      status: r.status || "unknown",
      strategy: decision.strategy_name || r.strategy || "",
      action: decision.action || "",
      rationale: decision.rationale || "",
      confidence: num(meta.confidence || decision.confidence || r.confidence),
      regimeConfidence: num(
        meta.regime_confidence ??
          meta.hmm_confidence ??
          r.regime_confidence ??
          decision.regime_confidence
      ),
      regimeKey: pipeline.regime_key || r.regime_key || "",
      close: num(pipeline.close),
      sigma: num(pipeline.sigma),
      lower1s: num(pipeline.lower_1s),
      upper1s: num(pipeline.upper_1s),
      lower2s: num(pipeline.lower_2s),
      upper2s: num(pipeline.upper_2s),
      S_D: num(pipeline.S_D),
      S_V: num(pipeline.S_V),
      S_L: num(pipeline.S_L),
      S_G: num(pipeline.S_G),
      sizeFraction: decision.size_fraction != null ? num(decision.size_fraction) : null,
      vaultTriggered: Boolean(meta.vault_triggered),
      forecastUncertainty: num(
        meta.forecast_uncertainty ??
          pipeline.forecast_uncertainty ??
          r.forecast_uncertainty
      ),
      hmmConfidence: num(meta.hmm_confidence),
      hmmTradeAllowed: Boolean(meta.hmm_trade_allowed),
      M_regime: meta.M_regime || "",
      M_D: num(meta.M_D),
      M_V: num(meta.M_V),
      M_L: num(meta.M_L),
      M_G: num(meta.M_G),
      M_trend_strength: num(meta.M_trend_strength),
      M_R_trans: num(meta.M_R_trans),
      coordinateStrategy: decision.coordinate_strategy || meta.coordinate_strategy || "",
      legs: r.matched_legs || decision.legs || [],
      entryDebit:
        r.entry_debit_dollars != null
          ? num(r.entry_debit_dollars)
          : r.entry_debit != null
            ? num(r.entry_debit)
            : null,
      thresholds: r.thresholds || decision.thresholds || null,
      planId: r.plan_id || null,
      rankScore: r.rank_score != null ? num(r.rank_score) : null,
      runAt: r.run_at_utc || r.run_at || "",
    };
  });

  return { activePlans, topRanked: activeRanked, symbolsInUniverse: symbols };
}

function buildTradeSummary(dataDir: string) {
  const rows = readCsvFile(path.join(dataDir, "trade_log.csv"));
  if (rows.length === 0) {
    return {
      totalTrades: 0,
      openTrades: 0,
      closedTrades: 0,
      realizedPnl: 0,
      unrealizedPnl: 0,
      winRate: 0,
      wins: 0,
      losses: 0,
    };
  }

  const latestByPid: Record<string, Record<string, string>> = {};
  const closedByPid: Record<string, Record<string, string>> = {};

  for (const row of rows) {
    const pid = row.plan_id || "";
    if (!pid) continue;
    latestByPid[pid] = row;
    if ((row.closed || "0").trim() === "1") {
      closedByPid[pid] = row;
    }
  }

  let totalRealized = 0;
  let wins = 0;
  let losses = 0;
  const closedPids = Object.keys(closedByPid);

  for (const pid of closedPids) {
    const pnl = num(closedByPid[pid].realized_pnl || closedByPid[pid].unrealized_pnl);
    totalRealized += pnl;
    if (pnl > 0) wins++;
    else if (pnl < 0) losses++;
  }

  let openCount = 0;
  let totalUnrealized = 0;
  for (const pid of Object.keys(latestByPid)) {
    if ((latestByPid[pid].closed || "0").trim() === "1") continue;
    openCount++;
    totalUnrealized += num(latestByPid[pid].unrealized_pnl);
  }

  const totalTrades = closedPids.length + openCount;

  return {
    totalTrades,
    openTrades: openCount,
    closedTrades: closedPids.length,
    realizedPnl: totalRealized,
    unrealizedPnl: totalUnrealized,
    winRate: closedPids.length > 0 ? wins / closedPids.length : 0,
    wins,
    losses,
  };
}

function buildEquityPositions(dataDir: string) {
  const data = readJson(path.join(dataDir, "equity_positions_state.json"));
  if (!data) return [];
  let positions: any[];
  if (Array.isArray(data)) {
    positions = data;
  } else if (typeof data === "object" && data.positions) {
    positions = Array.isArray(data.positions) ? data.positions : [];
  } else if (typeof data === "object") {
    positions = Object.values(data);
  } else {
    positions = [];
  }
  return positions.map((p: any) => ({
    planId: p.plan_id || "",
    symbol: p.symbol || "",
    direction: p.direction || "",
    side: p.side || "",
    quantity: num(p.quantity),
    entryPrice: num(p.entry_price),
    entryTs: p.entry_ts || p.entry_timestamp || "",
    status: p.status || "",
    exitPrice: p.exit_price != null ? num(p.exit_price) : null,
    exitTs: p.exit_ts || p.exit_timestamp || null,
    exitReason: p.exit_reason || null,
  }));
}

function buildEquityTradeSummary(dataDir: string) {
  const rows = readCsvFile(path.join(dataDir, "equity_trade_log.csv"));
  if (rows.length === 0) {
    return {
      totalTrades: 0,
      openTrades: 0,
      closedTrades: 0,
      realizedPnl: 0,
      unrealizedPnl: 0,
      winRate: 0,
    };
  }

  let open = 0;
  let closed = 0;
  let realized = 0;
  let unrealized = 0;
  let wins = 0;

  for (const row of rows) {
    const isClosed = (row.closed || row.status || "").trim() === "1" ||
      (row.status || "").toLowerCase() === "closed";
    if (isClosed) {
      closed++;
      const pnl = num(row.realized_pnl || row.pnl);
      realized += pnl;
      if (pnl > 0) wins++;
    } else {
      open++;
      unrealized += num(row.unrealized_pnl);
    }
  }

  return {
    totalTrades: open + closed,
    openTrades: open,
    closedTrades: closed,
    realizedPnl: realized,
    unrealizedPnl: unrealized,
    winRate: closed > 0 ? wins / closed : 0,
  };
}

function buildForecastTimeseries(dataDir: string) {
  const rows = readCsvFile(path.join(dataDir, "forecast_features_SPY.csv"));
  if (rows.length === 0) return [];

  const tail = rows.slice(-60);
  return tail.map((r) => {
    const hmmIdx = parseHmmStateIndex(r.hmm_state ?? r.HMM_State);
    const hmmLblRaw = r.hmm_state_label ?? r.HMM_State_Label;
    return {
      timestamp: r.timestamp || r.date || r.Date || "",
      close: num(r.close || r.Close),
      S_D: num(r.S_D),
      S_V: num(r.S_V),
      S_L: num(r.S_L),
      S_G: num(r.S_G),
      sigma: optionalNum(r.sigma),
      mean_price: r.mean_price ? num(r.mean_price) : null,
      lower_1s: r.lower_1s ? num(r.lower_1s) : null,
      upper_1s: r.upper_1s ? num(r.upper_1s) : null,
      lower_2s: r.lower_2s ? num(r.lower_2s) : null,
      upper_2s: r.upper_2s ? num(r.upper_2s) : null,
      hmm_state: hmmIdx,
      hmm_confidence: optionalNum(r.hmm_confidence),
      hmm_state_label: sanitizeHmmStateLabel(hmmLblRaw, hmmIdx),
      forecast_return: r.forecast_return ? num(r.forecast_return) : null,
      forecast_uncertainty: r.forecast_uncertainty
        ? num(r.forecast_uncertainty)
        : null,
    };
  });
}

function buildBacktestEquity(dataDir: string) {
  const rows = readCsvFile(path.join(dataDir, "backtest_equity_SPY.csv"));
  if (rows.length === 0) return [];

  const tail = rows.slice(-200);
  return tail.map((r) => ({
    timestamp: r.timestamp || r.date || r.Date || "",
    equity: num(r.equity || r.Equity || r.cumulative_equity),
  }));
}

function buildWalkforwardSummary(dataDir: string) {
  const data = readJson(path.join(dataDir, "walkforward_summary.json"));
  if (data && Array.isArray(data)) {
    return data.map((w: any) => ({
      windowId: num(w.window_id),
      oosStart: w.oos_start || "",
      oosEnd: w.oos_end || "",
      finalEquity: num(w.final_equity),
      totalReturnPct: num(w.total_return_pct),
      maxDrawdown: num(w.max_drawdown),
      sharpe: num(w.sharpe),
      numTrades: num(w.num_trades),
      winRate: num(w.win_rate),
      profitFactor: num(w.profit_factor),
    }));
  }

  const rows = readCsvFile(path.join(dataDir, "walkforward_summary.csv"));
  return rows.map((r) => ({
    windowId: num(r.window_id),
    oosStart: r.oos_start || "",
    oosEnd: r.oos_end || "",
    finalEquity: num(r.final_equity),
    totalReturnPct: num(r.total_return_pct),
    maxDrawdown: num(r.max_drawdown),
    sharpe: num(r.sharpe),
    numTrades: num(r.num_trades),
    winRate: num(r.win_rate),
    profitFactor: num(r.profit_factor),
  }));
}

export async function GET() {
  try {
    const resolved = resolveProcessedDir();
    const dataDir = resolved.dir;

    const forecastPath = path.join(dataDir, "forecast_features_SPY.csv");
    const plansPath = path.join(dataDir, "universe_trade_plans.json");
    const hasForecast = fs.existsSync(forecastPath);
    const hasPlans = fs.existsSync(plansPath);

    const marketState = buildMarketState(dataDir);
    const { activePlans, topRanked, symbolsInUniverse } = buildActivePlans(dataDir);
    const tradeSummary = buildTradeSummary(dataDir);
    const equityPositions = buildEquityPositions(dataDir);
    const equityTradeSummary = buildEquityTradeSummary(dataDir);
    const forecastTimeseries = buildForecastTimeseries(dataDir);
    const backtestEquity = buildBacktestEquity(dataDir);
    const walkforwardSummary = buildWalkforwardSummary(dataDir);

    return NextResponse.json({
      marketState,
      activePlans,
      topRanked,
      tradeSummary,
      equityPositions,
      equityTradeSummary,
      forecastTimeseries,
      backtestEquity,
      walkforwardSummary,
      generatedAt: new Date().toISOString(),
      symbolsInUniverse,
      dataMeta: {
        processedDir: dataDir,
        resolved: resolved.found,
        resolutionSource: resolved.source,
        hasForecastFeaturesCsv: hasForecast,
        hasUniversePlansJson: hasPlans,
        checkedPathsSample: resolved.tried.slice(0, 5),
      },
    });
  } catch (error) {
    console.error("Dashboard metrics API error:", error);
    return NextResponse.json(
      { error: "Failed to fetch metrics", detail: String(error) },
      { status: 500 }
    );
  }
}
