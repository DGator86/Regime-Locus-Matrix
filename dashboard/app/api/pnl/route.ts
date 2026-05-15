import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// Directory resolution
// ---------------------------------------------------------------------------

function resolveRepoRoot(): string {
  const envRoot = process.env.RLM_DATA_ROOT?.trim();
  if (envRoot) return path.resolve(envRoot, "..");

  const cwd = process.cwd();
  const cwdBase = path.basename(cwd).toLowerCase();
  return cwdBase === "dashboard" ? path.resolve(cwd, "..") : cwd;
}

function resolveLedgerDir(repoRoot: string): string {
  const rlmRoot = process.env.RLM_ROOT?.trim();
  if (rlmRoot) return path.join(path.resolve(rlmRoot), "data", "processed", "ledgers");
  const envDir = process.env.RLM_DATA_DIR?.trim();
  if (envDir) return path.join(envDir, "ledgers");
  const dataRoot = process.env.RLM_DATA_ROOT?.trim();
  if (dataRoot) return path.join(dataRoot, "processed", "ledgers");
  return path.join(repoRoot, "data", "processed", "ledgers");
}

// ---------------------------------------------------------------------------
// CSV parsing — ledger files have a comment row first, then headers
// ---------------------------------------------------------------------------

function parseCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuote = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (inQuote && line[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuote = !inQuote;
      }
    } else if (c === "," && !inQuote) {
      result.push(current.trim());
      current = "";
    } else {
      current += c;
    }
  }
  result.push(current.trim());
  return result;
}

function parseLedgerCsv(text: string): Record<string, string>[] {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  // Find the header row: starts with "book,"
  const headerIdx = lines.findIndex((l) => l.startsWith("book,"));
  if (headerIdx === -1) return [];
  const headers = parseCsvLine(lines[headerIdx]);
  return lines.slice(headerIdx + 1).map((line) => {
    const vals = parseCsvLine(line);
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = vals[i] ?? "";
    });
    return row;
  });
}

function readLedgerCsv(filePath: string): { rows: Record<string, string>[]; found: boolean; generatedAt: string } {
  try {
    if (!fs.existsSync(filePath)) return { rows: [], found: false, generatedAt: "" };
    const text = fs.readFileSync(filePath, "utf8");
    // Extract generated_utc from the comment line
    const firstLine = text.split("\n")[0] ?? "";
    const genMatch = firstLine.match(/generated_utc=([^\s,]+)/);
    const generatedAt = genMatch?.[1] ?? "";
    return { rows: parseLedgerCsv(text), found: true, generatedAt };
  } catch {
    return { rows: [], found: false, generatedAt: "" };
  }
}

function readJson<T>(filePath: string): T | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as T;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Time helpers (ET calendar)
// ---------------------------------------------------------------------------

function nowEtYmd(): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}

function timestampToEtYmd(iso: string): string | null {
  // Handle plain date strings (YYYY-MM-DD) and ISO timestamps
  const s = (iso || "").trim();
  if (!s) return null;
  // Plain date
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
  const ms = Date.parse(s.replace(/Z$/, "+00:00"));
  if (!Number.isFinite(ms)) return null;
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date(ms));
}

function isWithin7Days(iso: string): boolean {
  const s = (iso || "").trim();
  if (!s) return false;
  // Plain date: parse as midnight ET
  const raw = /^\d{4}-\d{2}-\d{2}$/.test(s) ? `${s}T12:00:00-05:00` : s.replace(/Z$/, "+00:00");
  const t = Date.parse(raw);
  if (!Number.isFinite(t)) return false;
  return t >= Date.now() - 7 * 86_400_000;
}

function num(v: string | undefined): number {
  const n = parseFloat(v ?? "");
  return Number.isFinite(n) ? n : 0;
}

// ---------------------------------------------------------------------------
// Book summary extraction
// ---------------------------------------------------------------------------

type LedgerRow = Record<string, string>;

type ChartPoint = { timestamp: string; value: number };

type BookSummary = {
  found: boolean;
  generatedAt: string;
  seed: number;
  bookValue: number;
  closedRealized: number;
  openMtm: number;
  netVsSeed: number;
  todayRealized: number;
  weeklyRealized: number;
  allTimeRealized: number;
  equityCurve: ChartPoint[];
};

function extractBookSummary(rows: LedgerRow[], generatedAt: string, found: boolean): BookSummary {
  const today = nowEtYmd();

  const seedRow = rows.find((r) => r.event === "SEED");
  const mtmRow = rows.find((r) => r.event === "OPEN_MTM_SUM");
  const closedRows = rows.filter((r) => r.event === "CLOSED" || r.event === "ROUND_TRIP");

  const seed = num(seedRow?.running_book_usd);
  const bookValue = num(mtmRow?.running_book_usd);
  const openMtm = num(mtmRow?.pnl_usd);
  // all-time realized = running_closed_usd of MTM row minus seed
  const allTimeRealized = closedRows.reduce((s, r) => s + num(r.pnl_usd), 0);

  let todayRealized = 0;
  let weeklyRealized = 0;
  for (const r of closedRows) {
    const ts = r.timestamp_utc;
    const pnl = num(r.pnl_usd);
    if (timestampToEtYmd(ts) === today) todayRealized += pnl;
    if (isWithin7Days(ts)) weeklyRealized += pnl;
  }

  // Equity curve: SEED point + each CLOSED/ROUND_TRIP running_closed_usd
  const equityCurve: ChartPoint[] = [];
  if (seedRow) {
    equityCurve.push({ timestamp: seedRow.timestamp_utc, value: seed });
  }
  for (const r of closedRows) {
    equityCurve.push({ timestamp: r.timestamp_utc, value: num(r.running_closed_usd) });
  }
  if (mtmRow) {
    equityCurve.push({ timestamp: mtmRow.timestamp_utc, value: num(mtmRow.running_closed_usd) });
  }

  return {
    found,
    generatedAt,
    seed,
    bookValue,
    closedRealized: allTimeRealized,
    openMtm,
    netVsSeed: bookValue - seed,
    todayRealized,
    weeklyRealized,
    allTimeRealized,
    equityCurve,
  };
}

type PdtSummary = BookSummary & {
  balance: number;
  target: number | null;
  progressPct: number;
  phase: string;
  closedTradeCount: number;
};

function extractPdtSummary(rows: LedgerRow[], generatedAt: string, found: boolean): PdtSummary {
  const base = extractBookSummary(rows, generatedAt, found);
  const mtmRow = rows.find((r) => r.event === "OPEN_MTM_SUM");
  // For PDT, running_closed_usd of OPEN_MTM_SUM = current balance
  const balance = num(mtmRow?.running_closed_usd);
  const closedTradeCount = rows.filter((r) => r.event === "ROUND_TRIP").length;

  // Phase classification (PDT $1K→$25K)
  const target = 25_000;
  const progressPct = base.seed > 0 && target > base.seed
    ? Math.max(0, Math.min(100, ((balance - base.seed) / (target - base.seed)) * 100))
    : 0;
  const phase =
    balance >= 25_000 ? "Complete"
    : balance >= 10_000 ? "Phase III — Scale"
    : balance >= 3_000 ? "Phase II — Growth"
    : "Phase I — Foundation";

  return {
    ...base,
    balance,
    target,
    progressPct,
    phase,
    closedTradeCount,
  };
}

// ---------------------------------------------------------------------------
// IBKR account cache (optional)
// ---------------------------------------------------------------------------

type IbkrAccount = {
  netLiq: number | null;
  cash: number | null;
  buyingPower: number | null;
  unrealizedPnl: number | null;
  realizedPnl: number | null;
  updatedAt: string | null;
};

function readIbkrAccountCache(repoRoot: string): IbkrAccount | null {
  const candidates = [
    path.join(repoRoot, "data", "processed", "ibkr_account_cache.json"),
    path.join(repoRoot, "data", "ibkr_account_cache.json"),
  ];
  for (const cand of candidates) {
    const data = readJson<Record<string, unknown>>(cand);
    if (!data) continue;
    const tag = (tagName: string): number | null => {
      // Format: { account_summary: [{tag, value}, ...] } or flat {NetLiquidation, ...}
      if (typeof data[tagName] === "number") return data[tagName] as number;
      if (typeof data[tagName] === "string") {
        const n = parseFloat(data[tagName] as string);
        return Number.isFinite(n) ? n : null;
      }
      const summary = data.account_summary;
      if (Array.isArray(summary)) {
        const row = summary.find((r: unknown) =>
          typeof r === "object" && r !== null && (r as Record<string, unknown>).tag === tagName,
        );
        if (row) {
          const v = parseFloat(String((row as Record<string, unknown>).value ?? ""));
          return Number.isFinite(v) ? v : null;
        }
      }
      return null;
    };
    return {
      netLiq: tag("NetLiquidation"),
      cash: tag("TotalCashValue"),
      buyingPower: tag("BuyingPower"),
      unrealizedPnl: tag("UnrealizedPnL"),
      realizedPnl: tag("RealizedPnL"),
      updatedAt: (data.updated_at as string | null) ?? (data.generatedAt as string | null) ?? null,
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Route handler
// ---------------------------------------------------------------------------

export async function GET() {
  try {
    const repoRoot = resolveRepoRoot();
    const ledgerDir = resolveLedgerDir(repoRoot);

    const equityFile = path.join(ledgerDir, "large_equities_book.csv");
    const optionsFile = path.join(ledgerDir, "large_options_book.csv");
    const pdtFile = path.join(ledgerDir, "pdt_robinhood_challenge_book.csv");

    const equityLedger = readLedgerCsv(equityFile);
    const optionsLedger = readLedgerCsv(optionsFile);
    const pdtLedger = readLedgerCsv(pdtFile);

    const largeEquities = extractBookSummary(equityLedger.rows, equityLedger.generatedAt, equityLedger.found);
    const largeOptions = extractBookSummary(optionsLedger.rows, optionsLedger.generatedAt, optionsLedger.found);
    const pdtChallenge = extractPdtSummary(pdtLedger.rows, pdtLedger.generatedAt, pdtLedger.found);

    const ibkrAccount = readIbkrAccountCache(repoRoot);

    return NextResponse.json({
      generatedAt: new Date().toISOString(),
      ledgerDir,
      largeEquities,
      largeOptions,
      pdtChallenge,
      ibkrAccount,
    });
  } catch (err) {
    console.error("[pnl] route error:", err);
    return NextResponse.json({ error: String(err) }, { status: 500 });
  }
}
