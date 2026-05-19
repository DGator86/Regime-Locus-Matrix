import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export const dynamic = "force-dynamic";

const DEFAULT_SYMBOLS = [
  "SPY",
  "QQQ",
  "AAPL",
  "AMZN",
  "GOOGL",
  "META",
  "MSFT",
  "NVDA",
  "TSLA",
  "AMD",
  "AVGO",
  "JPM",
];

function resolveRepoRoot(): string {
  const rlmRoot = process.env.RLM_ROOT?.trim();
  if (rlmRoot) return path.resolve(rlmRoot);
  const cwd = process.cwd();
  return path.basename(cwd).toLowerCase() === "dashboard"
    ? path.resolve(cwd, "..")
    : cwd;
}

function readJson(filePath: string): Record<string, unknown> | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    const data = JSON.parse(fs.readFileSync(filePath, "utf8"));
    return data && typeof data === "object" ? (data as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function fileMtimeIso(filePath: string): string | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    return fs.statSync(filePath).mtime.toISOString();
  } catch {
    return null;
  }
}

function stock1mPath(repoRoot: string, symbol: string): string {
  const sym = symbol.toUpperCase();
  return path.join(repoRoot, "data", "stocks", sym, "1m", `${sym.toLowerCase()}_full_1m.parquet`);
}

export async function GET() {
  try {
    const repoRoot = resolveRepoRoot();
    const processedDir = path.join(repoRoot, "data", "processed");
    const collectorState = readJson(path.join(processedDir, "eodhd_collector_state.json"));
    const polls =
      collectorState?.polls && typeof collectorState.polls === "object"
        ? (collectorState.polls as Record<string, string>)
        : {};

    const plans = readJson(path.join(processedDir, "universe_trade_plans.json"));
    const results = Array.isArray(plans?.results) ? plans.results : [];
    const firstPipe =
      results.length > 0 && typeof results[0] === "object"
        ? ((results[0] as Record<string, unknown>).pipeline as Record<string, unknown> | undefined)
        : undefined;

    const symbols: string[] = [];
    if (Array.isArray(plans?.symbols)) {
      for (const s of plans.symbols) {
        if (typeof s === "string" && s.trim()) symbols.push(s.trim().toUpperCase());
      }
    }
    if (symbols.length === 0) symbols.push(...DEFAULT_SYMBOLS);

    const stocks = symbols.map((symbol) => {
      const parquet = stock1mPath(repoRoot, symbol);
      const polledAt = polls[symbol] || null;
      return {
        symbol,
        parquetPath: parquet,
        parquetMtime: fileMtimeIso(parquet),
        lastPollEt: polledAt,
        parquetExists: fs.existsSync(parquet),
      };
    });

    const optionsLogCandidates = [
      path.join(processedDir, "options_large_account_trade_log.csv"),
      path.join(processedDir, "trade_log.csv"),
    ];
    let optionsLogPath = optionsLogCandidates[0];
    for (const p of optionsLogCandidates) {
      if (fs.existsSync(p)) {
        optionsLogPath = p;
        break;
      }
    }

    return NextResponse.json({
      generatedAt: new Date().toISOString(),
      repoRoot,
      stockSource: process.env.RLM_STOCK_BARS_SOURCE || "eodhd",
      eodhd: {
        pollWindow: `${process.env.RLM_EODHD_POLL_START || "09:00"}–${process.env.RLM_EODHD_POLL_END || "16:30"} ET`,
        pollIntervalSec: Number(process.env.RLM_EODHD_POLL_INTERVAL_SEC || "5"),
        collectorActive: Boolean(collectorState),
        lastSymbol: String(collectorState?.last_symbol || ""),
        symbols: stocks,
      },
      options: {
        provider: "massive",
        tradeLogPath: optionsLogPath,
        tradeLogMtime: fileMtimeIso(optionsLogPath),
        universePlansMtime: fileMtimeIso(path.join(processedDir, "universe_trade_plans.json")),
      },
      pipeline: {
        primaryBarSize: String(firstPipe?.primary_bar_size || plans?.bar_size || "1 min"),
        primaryDuration: String(firstPipe?.primary_duration || ""),
        plansGeneratedAt: String(plans?.generated_at_utc || plans?.run_at_utc || ""),
        activeCount: Array.isArray(plans?.active_ranked) ? plans.active_ranked.length : 0,
        symbolCount: results.length,
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to read data feeds", detail: String(error) },
      { status: 500 },
    );
  }
}
