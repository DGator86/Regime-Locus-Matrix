import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import {
  challengeSymbolSet,
  formatLegs,
  isWithinRollingDays,
  nowEtYmd,
  num,
  readCsvFile,
  readJson,
  resolveChallengePaths,
  resolveOptionsTradeLogPath,
  resolveRepoRootFromProcessed,
  summarizeEquityTradeLog,
  summarizeOptionsTradeLog,
  timestampToEtYmd,
} from "@/lib/tradingOverview";

export const dynamic = "force-dynamic";

function resolveProcessedDir(): string {
  const candidates: { abs: string }[] = [];
  const envDir = process.env.RLM_DATA_DIR?.trim();
  if (envDir) candidates.push({ abs: path.resolve(envDir) });
  const dataRoot = process.env.RLM_DATA_ROOT?.trim();
  if (dataRoot) candidates.push({ abs: path.resolve(path.join(dataRoot, "processed")) });
  const cwd = process.cwd();
  const cwdBase = path.basename(cwd).toLowerCase();
  if (cwdBase === "dashboard") {
    candidates.push({ abs: path.resolve(cwd, "..", "data", "processed") });
    candidates.push({ abs: path.resolve(cwd, "data", "processed") });
  } else {
    candidates.push({ abs: path.resolve(cwd, "data", "processed") });
    candidates.push({ abs: path.resolve(cwd, "..", "data", "processed") });
  }
  candidates.push({ abs: "/opt/Regime-Locus-Matrix/data/processed" });

  for (const { abs } of candidates) {
    try {
      if (fs.existsSync(abs) && fs.statSync(abs).isDirectory()) return abs;
    } catch {
      /* ignore */
    }
  }
  return candidates[0]?.abs || path.resolve(process.cwd(), "..", "data", "processed");
}

type Snapshot = ReturnType<typeof buildPlanSnapshot>;

function buildPlanSnapshot(item: Record<string, unknown>) {
  const dec = (item.decision as Record<string, unknown>) || {};
  const pipe = (item.pipeline as Record<string, unknown>) || {};
  const meta = (dec.metadata as Record<string, unknown>) || {};
  const legs = (item.matched_legs as unknown) || dec.legs;

  const regimeConf = num(
    String(meta.regime_confidence ?? meta.hmm_confidence ?? dec.regime_confidence ?? ""),
  );
  const strategyConf = num(String(meta.confidence ?? dec.confidence ?? ""));

  return {
    symbol: String(item.symbol || "").toUpperCase(),
    status: String(item.status || ""),
    skipReason: String(item.skip_reason || ""),
    regimeKey: String(pipe.regime_key || item.regime_key || ""),
    regimeConfidence: regimeConf,
    mtfRegime: String(meta.M_regime || ""),
    mtfConfidence: strategyConf,
    predictors: {
      M_D: num(String(meta.M_D)),
      M_V: num(String(meta.M_V)),
      M_L: num(String(meta.M_L)),
      M_G: num(String(meta.M_G)),
      M_trend_strength: num(String(meta.M_trend_strength)),
      M_R_trans: num(String(meta.M_R_trans)),
    },
    strategyName: String(dec.strategy_name || ""),
    action: String(dec.action || ""),
    legsHuman: formatLegs(legs),
    rationale: String(dec.rationale || "").slice(0, 500),
    regimeDirection: String(item.regime_direction || "").toLowerCase(),
    planId: String(item.plan_id || ""),
    rankScore: item.rank_score != null ? num(String(item.rank_score)) : null,
  };
}

function orderSnapshotsByRank(snaps: Snapshot[], activeRanked: unknown[]): Snapshot[] {
  const rankOrder = activeRanked.map((r) =>
    typeof r === "string" ? r.toUpperCase() : String((r as { symbol?: string }).symbol || "").toUpperCase(),
  );
  const rankIdx = new Map(rankOrder.map((s, i) => [s, i]));
  return [...snaps].sort((a, b) => {
    const ia = rankIdx.has(a.symbol) ? rankIdx.get(a.symbol)! : 999;
    const ib = rankIdx.has(b.symbol) ? rankIdx.get(b.symbol)! : 999;
    if (ia !== ib) return ia - ib;
    return a.symbol.localeCompare(b.symbol);
  });
}

type ChallengeStateFile = {
  balance?: number;
  seed?: number;
  target?: number;
  open_positions?: Record<string, unknown>[];
  trade_history?: Record<string, unknown>[];
};

function summarizeChallengeState(state: ChallengeStateFile | null) {
  if (!state) {
    return {
      loaded: false,
      balance: null as number | null,
      seed: null as number | null,
      target: null as number | null,
      totalReturnDollarsNet: 0,
      totalReturnPct: 0,
      dailyRealized: 0,
      weeklyRealizedRolling7d: 0,
      openPositions: [] as Record<string, unknown>[],
      closedTrades: [] as Record<string, unknown>[],
    };
  }

  const seed = Number(state.seed ?? 0);
  const balance = Number(state.balance ?? seed);
  const history = Array.isArray(state.trade_history) ? state.trade_history : [];
  const open = Array.isArray(state.open_positions) ? state.open_positions : [];

  const today = nowEtYmd();
  let dailyRealized = 0;
  let weeklyRealized = 0;

  for (const t of history) {
    const exitRaw = String(t.exit_date || "");
    const pnl = Number(t.pnl ?? 0);
    const isoGuess = exitRaw.includes("T") ? exitRaw : `${exitRaw}T23:59:59Z`;
    if (timestampToEtYmd(isoGuess) === today) dailyRealized += pnl;
    if (isWithinRollingDays(isoGuess, 7)) weeklyRealized += pnl;
  }

  return {
    loaded: true,
    balance,
    seed: seed || null,
    target: state.target != null ? Number(state.target) : null,
    totalReturnDollarsNet: balance - seed,
    totalReturnPct: seed > 0 ? ((balance - seed) / seed) * 100 : 0,
    dailyRealized,
    weeklyRealizedRolling7d: weeklyRealized,
    openPositions: open,
    closedTrades: [...history].reverse(),
  };
}

export async function GET() {
  try {
    const processedDir = resolveProcessedDir();
    const repoRoot = resolveRepoRootFromProcessed(processedDir);
    const plansPath = path.join(processedDir, "universe_trade_plans.json");
    const plans = readJson<Record<string, unknown>>(plansPath);
    const resultsRaw = Array.isArray(plans?.results) ? (plans!.results as Record<string, unknown>[]) : [];
    const activeRanked = Array.isArray(plans?.active_ranked) ? plans!.active_ranked : [];

    const allSnapshots = resultsRaw.map(buildPlanSnapshot).filter((s) => s.symbol);
    const chSet = challengeSymbolSet();
    const challengeSnapshots = orderSnapshotsByRank(
      allSnapshots.filter((s) => chSet.has(s.symbol)),
      activeRanked,
    );
    const nonChallenge = allSnapshots.filter((s) => !chSet.has(s.symbol));
    const mergedLarge =
      nonChallenge.length > 0 ? orderSnapshotsByRank(nonChallenge, activeRanked) : orderSnapshotsByRank([...allSnapshots], activeRanked);

    const equitySnapshots = orderSnapshotsByRank(
      allSnapshots.filter(
        (s) =>
          s.regimeDirection === "bull" ||
          s.regimeDirection === "bear" ||
          (s.regimeKey || "").toLowerCase().includes("bull") ||
          (s.regimeKey || "").toLowerCase().includes("bear"),
      ),
      activeRanked,
    );

    const optionsLogPath = resolveOptionsTradeLogPath(repoRoot, processedDir);
    const optionsRows = readCsvFile(optionsLogPath);
    const largeOptionsPnl = summarizeOptionsTradeLog(optionsRows);

    const chPaths = resolveChallengePaths(repoRoot);
    const challengeRows = readCsvFile(chPaths.tradeLogCsv);
    const challengeState = readJson<ChallengeStateFile>(chPaths.stateJson);
    const challengePnL = summarizeChallengeState(challengeState);

    const equityLogPath = path.join(processedDir, "equity_trade_log.csv");
    const equityRows = readCsvFile(equityLogPath);
    const equityPnl = summarizeEquityTradeLog(equityRows);

    const equityStatePath = path.join(processedDir, "equity_positions_state.json");
    const equityStateRaw = readJson<unknown>(equityStatePath);
    let equityStatePositions: Record<string, unknown>[] = [];
    if (equityStateRaw && typeof equityStateRaw === "object") {
      const o = equityStateRaw as Record<string, unknown>;
      if (Array.isArray(o.positions)) equityStatePositions = o.positions as Record<string, unknown>[];
      else if (!Array.isArray(o))
        equityStatePositions = Object.values(o).filter((x) => x && typeof x === "object") as Record<
          string,
          unknown
        >[];
    }

    const equitySymFromState = equityStatePositions
      .map((p) => String((p as { symbol?: string }).symbol || "").toUpperCase())
      .filter(Boolean);

    return NextResponse.json({
      generatedAt: new Date().toISOString(),
      paths: {
        processedDir,
        universePlans: plansPath,
        optionsTradeLog: optionsLogPath,
        equityTradeLog: equityLogPath,
        challengeState: chPaths.stateJson,
      },
      challengeSymbols: [...chSet].sort(),
      largeAccountOptions: {
        tickers: [...new Set(mergedLarge.map((s) => s.symbol))],
        symbols: mergedLarge,
        positionsOpen: largeOptionsPnl.openRows,
        positionsClosed: largeOptionsPnl.closedRows.slice(0, 200),
        pnl: {
          dailyRealized: largeOptionsPnl.dailyRealized,
          weeklyRealizedRolling7d: largeOptionsPnl.weeklyRealized,
          allTimeRealizedClosed: largeOptionsPnl.allTimeRealized,
          openMarkToMarket: largeOptionsPnl.openMtmPnl,
          combinedOpenPlusRealized: largeOptionsPnl.allTimeRealized + largeOptionsPnl.openMtmPnl,
        },
      },
      pdtChallengeOptions: {
        tickers: [...new Set(challengeSnapshots.map((s) => s.symbol))],
        symbols: challengeSnapshots,
        challengeAccount: challengePnL,
        csvRecentRows: challengeRows.slice(-80),
      },
      equities: {
        tickers: [...new Set([...equitySnapshots.map((s) => s.symbol), ...equitySymFromState])].sort(),
        symbols: equitySnapshots,
        positionsOpen: equityPnl.openRows,
        positionsClosed: equityPnl.closedRows.slice(0, 200),
        statePositions: equityStatePositions,
        pnl: {
          dailyRealized: equityPnl.dailyRealized,
          weeklyRealizedRolling7d: equityPnl.weeklyRealized,
          allTimeRealizedClosed: equityPnl.allTimeRealized,
          openMarkToMarket: equityPnl.openMtmPnl,
          combinedOpenPlusRealized: equityPnl.allTimeRealized + equityPnl.openMtmPnl,
        },
      },
    });
  } catch (error) {
    console.error("trading-overview API error:", error);
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}
