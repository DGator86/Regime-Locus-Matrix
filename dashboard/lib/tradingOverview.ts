import fs from "fs";
import path from "path";

/** ET calendar day YYYY-MM-DD */
export function timestampToEtYmd(iso: string): string | null {
  const ms = Date.parse(iso.replace(/Z$/, "+00:00"));
  if (!Number.isFinite(ms)) return null;
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date(ms));
}

export function nowEtYmd(): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}

/** Rolling window by UTC ms (weekly PNL = closed trades in the last 7×24h from exit timestamp). */
export function isWithinRollingDays(iso: string, days: number): boolean {
  const t = Date.parse(iso.replace(/Z$/, "+00:00"));
  if (!Number.isFinite(t)) return false;
  return t >= Date.now() - days * 86400000 * 1000;
}

export function parseCsv(text: string): Record<string, string>[] {
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

export function num(v: string | undefined): number {
  const n = parseFloat(v ?? "");
  return Number.isFinite(n) ? n : 0;
}

export function readCsvFile(filePath: string): Record<string, string>[] {
  try {
    if (!fs.existsSync(filePath)) return [];
    return parseCsv(fs.readFileSync(filePath, "utf8"));
  } catch {
    return [];
  }
}

export function readJson<T>(filePath: string): T | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as T;
  } catch {
    return null;
  }
}

/** Latest row per plan_id by max timestamp_utc (file order tie-break). */
export function latestRowPerPlan(rows: Record<string, string>[]): Record<string, Record<string, string>> {
  const best = new Map<string, { t: number; row: Record<string, string> }>();
  for (const row of rows) {
    const pid = (row.plan_id || "").trim();
    if (!pid) continue;
    const t = Date.parse((row.timestamp_utc || "").replace(/Z$/, "+00:00"));
    const prev = best.get(pid);
    if (!prev || t >= prev.t) best.set(pid, { t: Number.isFinite(t) ? t : 0, row });
  }
  const out: Record<string, Record<string, string>> = {};
  best.forEach((v, k) => {
    out[k] = v.row;
  });
  return out;
}

export function isClosedRow(row: Record<string, string>): boolean {
  const c = (row.closed || "").trim();
  if (c === "1") return true;
  const sig = (row.signal || "").toLowerCase();
  return sig === "closed" || sig.includes("closed");
}

export type OptionsPnlSummary = {
  openPositionCount: number;
  closedTradeCount: number;
  openMtmPnl: number;
  dailyRealized: number;
  weeklyRealized: number;
  allTimeRealized: number;
};

/** Options-style trade_log.csv (monitor). Closed PNL taken from final row unrealized_pnl per plan. */
export function summarizeOptionsTradeLog(rows: Record<string, string>[]): OptionsPnlSummary & {
  openRows: Record<string, string>[];
  closedRows: Record<string, string>[];
} {
  const latest = latestRowPerPlan(rows);
  const openRows: Record<string, string>[] = [];
  const closedRows: Record<string, string>[] = [];
  let openMtmPnl = 0;
  let allTimeRealized = 0;
  let dailyRealized = 0;
  let weeklyRealized = 0;

  const today = nowEtYmd();

  Object.values(latest).forEach((row) => {
    const iso = row.timestamp_utc || "";
    if (isClosedRow(row)) {
      closedRows.push(row);
      const pnl = num(row.unrealized_pnl);
      allTimeRealized += pnl;
      const ymd = timestampToEtYmd(iso);
      if (ymd === today) dailyRealized += pnl;
      if (isWithinRollingDays(iso, 7)) weeklyRealized += pnl;
    } else {
      openRows.push(row);
      openMtmPnl += num(row.unrealized_pnl);
    }
  });

  openRows.sort((a, b) => (a.symbol || "").localeCompare(b.symbol || ""));
  closedRows.sort((a, b) => (b.timestamp_utc || "").localeCompare(a.timestamp_utc || ""));

  return {
    openRows,
    closedRows,
    openPositionCount: openRows.length,
    closedTradeCount: closedRows.length,
    openMtmPnl,
    dailyRealized,
    weeklyRealized,
    allTimeRealized,
  };
}

/** Equity trade_log: rows have action + quantity */
export function summarizeEquityTradeLog(rows: Record<string, string>[]): OptionsPnlSummary & {
  openRows: Record<string, string>[];
  closedRows: Record<string, string>[];
} {
  const latest = latestRowPerPlan(rows);
  const openRows: Record<string, string>[] = [];
  const closedRows: Record<string, string>[] = [];
  let openMtmPnl = 0;
  let allTimeRealized = 0;
  let dailyRealized = 0;
  let weeklyRealized = 0;

  const today = nowEtYmd();

  Object.values(latest).forEach((row) => {
    const iso = row.timestamp_utc || "";
    const closed =
      (row.closed || "").trim() === "1" ||
      (row.signal || "").toLowerCase() === "closed" ||
      (row.status || "").toLowerCase() === "closed";

    if (closed) {
      closedRows.push(row);
      const pnl = num(row.unrealized_pnl || row.realized_pnl);
      allTimeRealized += pnl;
      const ymd = timestampToEtYmd(iso);
      if (ymd === today) dailyRealized += pnl;
      if (isWithinRollingDays(iso, 7)) weeklyRealized += pnl;
    } else {
      openRows.push(row);
      openMtmPnl += num(row.unrealized_pnl);
    }
  });

  openRows.sort((a, b) => (a.symbol || "").localeCompare(b.symbol || ""));
  closedRows.sort((a, b) => (b.timestamp_utc || "").localeCompare(a.timestamp_utc || ""));

  return {
    openRows,
    closedRows,
    openPositionCount: openRows.length,
    closedTradeCount: closedRows.length,
    openMtmPnl,
    dailyRealized,
    weeklyRealized,
    allTimeRealized,
  };
}

export function resolveRepoRootFromProcessed(processedDir: string): string {
  return path.resolve(processedDir, "..", "..");
}

export function resolveChallengePaths(repoRoot: string): {
  stateJson: string;
  tradeLogCsv: string;
} {
  return {
    stateJson: path.join(repoRoot, "data", "challenge", "state.json"),
    tradeLogCsv: path.join(repoRoot, "data", "challenge", "trade_log.csv"),
  };
}

export function resolveOptionsTradeLogPath(repoRoot: string, processedDir: string): string {
  const raw = process.env.RLM_OPTIONS_TRADE_LOG_PATH?.trim();
  if (raw) return path.isAbsolute(raw) ? raw : path.resolve(repoRoot, raw);
  return path.join(processedDir, "trade_log.csv");
}

export function challengeSymbolSet(): Set<string> {
  const raw = process.env.RLM_CHALLENGE_SYMBOLS?.trim();
  const parts = (raw || "SPY,QQQ")
    .split(/[,\s]+/)
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean);
  return new Set(parts);
}

export function formatLegs(legs: unknown): string {
  if (!Array.isArray(legs) || legs.length === 0) return "—";
  return legs
    .map((leg) => {
      if (!leg || typeof leg !== "object") return "";
      const o = leg as Record<string, unknown>;
      const sym = String(o.symbol || o.contract_symbol || "");
      const side = String(o.side || o.action || "");
      const qty = o.quantity != null ? String(o.quantity) : "";
      const strike = o.strike != null ? String(o.strike) : "";
      const right = String(o.right || o.option_type || "");
      const expiry = String(o.expiry || o.expiration || "");
      const bits = [side, qty, sym || strike + right, expiry].filter(Boolean);
      return bits.join(" ");
    })
    .filter(Boolean)
    .join(" | ");
}
