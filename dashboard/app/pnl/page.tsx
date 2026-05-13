"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { PageHero } from "@/components/PageHero";
import { cn } from "@/lib/utils";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  DollarSign,
  RefreshCw,
  Server,
  TrendingDown,
  TrendingUp,
  Wallet,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

type PdtSummary = BookSummary & {
  balance: number;
  target: number | null;
  progressPct: number;
  phase: string;
  closedTradeCount: number;
};

type IbkrAccount = {
  netLiq: number | null;
  cash: number | null;
  buyingPower: number | null;
  unrealizedPnl: number | null;
  realizedPnl: number | null;
  updatedAt: string | null;
};

type PnlPayload = {
  generatedAt: string;
  ledgerDir: string;
  largeEquities: BookSummary;
  largeOptions: BookSummary;
  pdtChallenge: PdtSummary;
  ibkrAccount: IbkrAccount | null;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtMoney(v: number | null | undefined, opts?: { sign?: boolean; decimals?: number }) {
  if (v == null || !Number.isFinite(v)) return "—";
  const decimals = opts?.decimals ?? 2;
  const abs = Math.abs(v).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  if (opts?.sign) {
    return v >= 0 ? `+$${abs}` : `-$${abs}`;
  }
  return v < 0 ? `-$${abs}` : `$${abs}`;
}

function pnlColor(v: number) {
  if (v > 0) return "text-emerald-400";
  if (v < 0) return "text-rose-400";
  return "text-slate-400";
}

function pnlBg(v: number) {
  if (v > 0) return "border-emerald-500/25 bg-emerald-500/[0.05]";
  if (v < 0) return "border-rose-500/25 bg-rose-500/[0.05]";
  return "border-white/[0.07] bg-white/[0.02]";
}

function fmtPct(v: number, decimals = 2) {
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(decimals)}%`;
}

function fmtTs(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function chartLabel(iso: string): string {
  if (!iso) return "";
  const plain = /^\d{4}-\d{2}-\d{2}$/.test(iso);
  try {
    const d = plain ? new Date(`${iso}T12:00:00`) : new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  } catch {
    return iso.slice(0, 10);
  }
}

// ---------------------------------------------------------------------------
// Mini equity curve
// ---------------------------------------------------------------------------

function EquityCurve({ data, seed }: { data: ChartPoint[]; seed: number }) {
  if (data.length < 2) {
    return (
      <div className="h-[72px] flex items-center justify-center text-[10px] text-slate-600 border border-dashed border-white/[0.06] rounded-lg">
        No curve data
      </div>
    );
  }
  const chartData = data.map((pt) => ({
    ...pt,
    label: chartLabel(pt.timestamp),
  }));
  const last = chartData[chartData.length - 1]?.value ?? seed;
  const positive = last >= seed;
  const stroke = positive ? "#34d399" : "#f43f5e";
  const fill = positive ? "rgba(52,211,153,0.12)" : "rgba(244,63,94,0.1)";

  return (
    <div className="h-[72px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 2, right: 2, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={`curve-${positive ? "pos" : "neg"}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={stroke} stopOpacity={0.3} />
              <stop offset="100%" stopColor={stroke} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="label" hide />
          <YAxis domain={["auto", "auto"]} hide />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0d1118",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8,
              fontSize: 10,
            }}
            labelStyle={{ color: "#64748b" }}
            itemStyle={{ color: stroke }}
            formatter={(v: number) => [fmtMoney(v), "Book"]}
          />
          <Area
            type="monotone"
            dataKey="value"
            stroke={stroke}
            strokeWidth={1.5}
            fill={`url(#curve-${positive ? "pos" : "neg"})`}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// P&L metric row
// ---------------------------------------------------------------------------

function PnlRow({
  label,
  value,
  sub,
}: {
  label: string;
  value: number;
  sub?: string;
}) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-white/[0.04] last:border-0">
      <span className="text-[11px] text-slate-500">{label}</span>
      <div className="text-right">
        <span className={cn("text-[12px] font-bold font-[family-name:var(--font-mono)]", pnlColor(value))}>
          {fmtMoney(value, { sign: true })}
        </span>
        {sub && <p className="text-[10px] text-slate-600">{sub}</p>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Book header (seed / book value / net)
// ---------------------------------------------------------------------------

function BookHeader({ seed, bookValue, netVsSeed }: { seed: number; bookValue: number; netVsSeed: number }) {
  const retPct = seed > 0 ? (netVsSeed / seed) * 100 : 0;
  return (
    <div className="flex items-start justify-between gap-4 pb-3 border-b border-white/[0.06]">
      <div>
        <p className="text-[10px] uppercase tracking-wider text-slate-500">Book Value</p>
        <p className="text-2xl font-black font-[family-name:var(--font-mono)] text-white">
          {fmtMoney(bookValue)}
        </p>
        <p className="text-[11px] text-slate-500 mt-0.5">Seed {fmtMoney(seed)}</p>
      </div>
      <div className="text-right">
        <p className="text-[10px] uppercase tracking-wider text-slate-500">Net vs Seed</p>
        <p className={cn("text-lg font-bold font-[family-name:var(--font-mono)] mt-0.5", pnlColor(netVsSeed))}>
          {fmtMoney(netVsSeed, { sign: true })}
        </p>
        <p className={cn("text-[11px] font-[family-name:var(--font-mono)]", pnlColor(retPct))}>
          {fmtPct(retPct)}
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Large book card (equities or options)
// ---------------------------------------------------------------------------

function LargeBookCard({
  title,
  subtitle,
  book,
  accent = "cyan",
}: {
  title: string;
  subtitle: string;
  book: BookSummary;
  accent?: "cyan" | "violet";
}) {
  const accentBorder = accent === "violet" ? "border-violet-500/20" : "border-cyan-500/20";
  const accentBg = accent === "violet" ? "from-violet-500/[0.05]" : "from-cyan-500/[0.05]";
  const accentIcon = accent === "violet" ? "text-violet-400" : "text-cyan-400";

  return (
    <div
      className={cn(
        "rounded-2xl border p-5 space-y-4 bg-gradient-to-br to-transparent",
        accentBorder,
        accentBg,
      )}
    >
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className={cn("p-2 rounded-lg bg-white/[0.04] mt-0.5")}>
          <BarChart3 className={cn("w-4 h-4", accentIcon)} />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">{title}</p>
          <p className="text-[10px] text-slate-600 mt-0.5">{subtitle}</p>
        </div>
        {!book.found && (
          <span className="text-[10px] bg-amber-500/15 border border-amber-500/30 text-amber-300 px-2 py-0.5 rounded-md">
            No ledger
          </span>
        )}
      </div>

      <BookHeader seed={book.seed} bookValue={book.bookValue} netVsSeed={book.netVsSeed} />

      {/* P&L rows */}
      <div>
        <PnlRow label="Today (realized)" value={book.todayRealized} />
        <PnlRow label="This week (realized)" value={book.weeklyRealized} />
        <PnlRow label="All-time realized" value={book.allTimeRealized} />
        <PnlRow
          label="Open MTM"
          value={book.openMtm}
          sub="unrealized"
        />
      </div>

      {/* Equity curve */}
      <div>
        <p className="text-[10px] uppercase tracking-wider text-slate-600 mb-2">Realized equity curve</p>
        <EquityCurve data={book.equityCurve} seed={book.seed} />
      </div>

      {book.generatedAt && (
        <p className="text-[10px] text-slate-700 font-[family-name:var(--font-mono)] text-right">
          Ledger as of {fmtTs(book.generatedAt)}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// PDT challenge card
// ---------------------------------------------------------------------------

function PdtCard({ pdt }: { pdt: PdtSummary }) {
  const retPct = pdt.seed > 0 ? ((pdt.balance - pdt.seed) / pdt.seed) * 100 : 0;
  const netCash = pdt.balance - pdt.seed;

  return (
    <div className="rounded-2xl border border-amber-500/20 bg-gradient-to-br from-amber-500/[0.04] to-transparent p-5 space-y-4">
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-lg bg-white/[0.04] mt-0.5">
          <Activity className="w-4 h-4 text-amber-400" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">PDT Challenge</p>
          <p className="text-[10px] text-slate-600 mt-0.5">Robinhood $1K→$25K · elite paper track</p>
        </div>
        {!pdt.found && (
          <span className="text-[10px] bg-amber-500/15 border border-amber-500/30 text-amber-300 px-2 py-0.5 rounded-md">
            No ledger
          </span>
        )}
      </div>

      {/* Balance vs seed */}
      <div className="flex items-start justify-between gap-4 pb-3 border-b border-white/[0.06]">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-slate-500">Balance</p>
          <p className="text-2xl font-black font-[family-name:var(--font-mono)] text-white">
            {fmtMoney(pdt.balance || pdt.bookValue)}
          </p>
          <p className="text-[11px] text-slate-500 mt-0.5">Seed {fmtMoney(pdt.seed)}</p>
        </div>
        <div className="text-right">
          <p className="text-[10px] uppercase tracking-wider text-slate-500">Net vs Seed</p>
          <p className={cn("text-lg font-bold font-[family-name:var(--font-mono)] mt-0.5", pnlColor(netCash))}>
            {fmtMoney(netCash, { sign: true })}
          </p>
          <p className={cn("text-[11px] font-[family-name:var(--font-mono)]", pnlColor(retPct))}>
            {fmtPct(retPct)}
          </p>
        </div>
      </div>

      {/* Progress bar */}
      {pdt.target != null && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] text-slate-500 font-[family-name:var(--font-mono)]">
            <span>Seed {fmtMoney(pdt.seed, { decimals: 0 })}</span>
            <span className="text-amber-400/80 font-semibold">
              {pdt.progressPct.toFixed(1)}% · {pdt.phase}
            </span>
            <span>Target {fmtMoney(pdt.target, { decimals: 0 })}</span>
          </div>
          <div className="relative h-2 w-full rounded-full bg-white/[0.06] overflow-hidden">
            <div
              className="absolute left-0 top-0 h-full rounded-full bg-gradient-to-r from-amber-500 to-orange-400 transition-all duration-700"
              style={{ width: `${Math.max(0.5, pdt.progressPct)}%` }}
            />
          </div>
        </div>
      )}

      {/* P&L rows */}
      <div>
        <PnlRow label="Today (realized)" value={pdt.todayRealized} />
        <PnlRow label="This week (realized)" value={pdt.weeklyRealized} />
        <PnlRow label="All-time realized" value={pdt.allTimeRealized} />
        <PnlRow label="Open MTM" value={pdt.openMtm} sub="unrealized" />
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-2 text-center">
        <div className="rounded-lg bg-white/[0.03] border border-white/[0.05] py-2">
          <p className="text-lg font-bold font-[family-name:var(--font-mono)] text-white">
            {pdt.closedTradeCount}
          </p>
          <p className="text-[10px] text-slate-500">Closed trades</p>
        </div>
        <div className="rounded-lg bg-white/[0.03] border border-white/[0.05] py-2">
          <p className={cn("text-lg font-bold font-[family-name:var(--font-mono)]", pnlColor(pdt.openMtm))}>
            {fmtMoney(pdt.openMtm, { sign: true })}
          </p>
          <p className="text-[10px] text-slate-500">Open MTM</p>
        </div>
      </div>

      {/* Equity curve */}
      <div>
        <p className="text-[10px] uppercase tracking-wider text-slate-600 mb-2">Realized equity curve</p>
        <EquityCurve data={pdt.equityCurve} seed={pdt.seed} />
      </div>

      {pdt.generatedAt && (
        <p className="text-[10px] text-slate-700 font-[family-name:var(--font-mono)] text-right">
          Ledger as of {fmtTs(pdt.generatedAt)}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// IBKR account panel
// ---------------------------------------------------------------------------

function IbkrPanel({ account }: { account: IbkrAccount | null }) {
  return (
    <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-5">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 rounded-lg bg-white/[0.04]">
          <Server className="w-4 h-4 text-slate-400" />
        </div>
        <div>
          <p className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">IBKR Account</p>
          <p className="text-[10px] text-slate-600">Live gateway data · TWS / IB Gateway</p>
        </div>
        {account?.updatedAt && (
          <p className="ml-auto text-[10px] text-slate-600 font-[family-name:var(--font-mono)]">
            {fmtTs(account.updatedAt)}
          </p>
        )}
      </div>

      {account ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          <IbkrMetric label="Net Liq" value={account.netLiq} />
          <IbkrMetric label="Cash" value={account.cash} />
          <IbkrMetric label="Buying Power" value={account.buyingPower} />
          <IbkrMetric label="Unrealized P&L" value={account.unrealizedPnl} signed />
          <IbkrMetric label="Realized P&L" value={account.realizedPnl} signed />
        </div>
      ) : (
        <div className="flex items-center gap-3 py-3 px-4 rounded-xl border border-amber-500/20 bg-amber-500/[0.04]">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
          <div>
            <p className="text-[12px] font-semibold text-amber-200">No IBKR account cache</p>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Write{" "}
              <code className="text-[10px] bg-white/[0.05] px-1 rounded">
                data/processed/ibkr_account_cache.json
              </code>{" "}
              from the Telegram bot <code className="text-[10px] bg-white/[0.05] px-1 rounded">/balance</code>{" "}
              command to populate this panel.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function IbkrMetric({ label, value, signed }: { label: string; value: number | null; signed?: boolean }) {
  return (
    <div className="rounded-xl bg-white/[0.02] border border-white/[0.05] px-3 py-2.5">
      <p className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</p>
      <p
        className={cn(
          "text-sm font-bold font-[family-name:var(--font-mono)] mt-1",
          signed ? pnlColor(value ?? 0) : "text-slate-200",
        )}
      >
        {value != null ? fmtMoney(value, signed ? { sign: true } : undefined) : "—"}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Summary strip
// ---------------------------------------------------------------------------

function SummaryStrip({ data }: { data: PnlPayload }) {
  const { largeEquities: eq, largeOptions: opt, pdtChallenge: pdt } = data;

  // Large accounts consolidated (equities + options)
  const largeSeed = eq.seed + opt.seed;
  const largeBookValue = eq.bookValue + opt.bookValue;
  const largeNetVsSeed = eq.netVsSeed + opt.netVsSeed;
  const largeRetPct = largeSeed > 0 ? (largeNetVsSeed / largeSeed) * 100 : 0;

  const todayAll = eq.todayRealized + opt.todayRealized + pdt.todayRealized;
  const weekAll = eq.weeklyRealized + opt.weeklyRealized + pdt.weeklyRealized;
  const realizedAll = eq.allTimeRealized + opt.allTimeRealized + pdt.allTimeRealized;
  const openMtmAll = eq.openMtm + opt.openMtm + pdt.openMtm;

  return (
    <div className="glass rounded-2xl p-5 panel-hud">
      <div className="flex items-center gap-2 mb-4">
        <Wallet className="w-4 h-4 text-cyan-400" />
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200">Consolidated Books</h2>
        <span className="text-[10px] text-slate-600 ml-auto font-[family-name:var(--font-mono)]">
          Ledger as of {fmtTs(data.generatedAt)}
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <SummaryMetric
          label="Large accts book value"
          value={largeBookValue}
          sub={`Seed ${fmtMoney(largeSeed)}`}
        />
        <SummaryMetric
          label="Large accts net vs seed"
          value={largeNetVsSeed}
          signed
          sub={fmtPct(largeRetPct)}
          subColor={pnlColor(largeRetPct)}
        />
        <SummaryMetric label="Today's realized" value={todayAll} signed />
        <SummaryMetric label="7-day realized" value={weekAll} signed />
        <SummaryMetric
          label="All-time realized"
          value={realizedAll}
          signed
          sub={`Open MTM ${fmtMoney(openMtmAll, { sign: true })}`}
          subColor={pnlColor(openMtmAll)}
        />
      </div>
    </div>
  );
}

function SummaryMetric({
  label,
  value,
  sub,
  signed,
  subColor,
}: {
  label: string;
  value: number;
  sub?: string;
  signed?: boolean;
  subColor?: string;
}) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className={cn("text-xl font-black font-[family-name:var(--font-mono)] mt-1", signed ? pnlColor(value) : "text-slate-100")}>
        {fmtMoney(value, signed ? { sign: true } : undefined)}
      </p>
      {sub && (
        <p className={cn("text-[10px] mt-0.5 font-[family-name:var(--font-mono)]", subColor ?? "text-slate-600")}>
          {sub}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Data freshness note
// ---------------------------------------------------------------------------

function DataNote({ ledgerDir, found }: { ledgerDir: string; found: boolean }) {
  if (found) return null;
  return (
    <div className="rounded-xl border border-amber-500/25 bg-amber-500/[0.05] px-4 py-3 flex items-start gap-3">
      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
      <div className="text-[12px] text-slate-400 space-y-1">
        <p className="font-semibold text-amber-200">Ledger files not found</p>
        <p>
          Expected in{" "}
          <code className="text-[11px] bg-white/[0.05] px-1 rounded">{ledgerDir}</code>
        </p>
        <p>
          Run{" "}
          <code className="text-[11px] bg-white/[0.05] px-1 rounded">python scripts/sync_trading_ledgers.py</code>{" "}
          on the VPS, then sync with{" "}
          <code className="text-[11px] bg-white/[0.05] px-1 rounded">
            powershell ./scripts/sync_dashboard_data_from_vps.ps1
          </code>
          .
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function PnlDashboardPage() {
  const [data, setData] = useState<PnlPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const inFlight = useRef(false);

  const load = useCallback(async (showSpinner = false) => {
    if (inFlight.current) return;
    inFlight.current = true;
    if (showSpinner) setRefreshing(true);
    try {
      const res = await fetch("/api/pnl");
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error || res.statusText);
      setData(json);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      inFlight.current = false;
      if (showSpinner) setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    const boot = window.setTimeout(() => void load(), 0);
    const iv = window.setInterval(() => void load(), 60_000);
    return () => {
      window.clearTimeout(boot);
      window.clearInterval(iv);
    };
  }, [load]);

  const anyLedgerFound = data
    ? data.largeEquities.found || data.largeOptions.found || data.pdtChallenge.found
    : true;

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      <PageHero
        eyebrow="Performance"
        title="P&amp;L Dashboard"
        subtitle="Book-level accounting across all trading sleeves. Sourced from ledger CSVs — refreshes every 60 s."
      />

      {/* Controls */}
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => void load(true)}
          disabled={refreshing}
          className="flex items-center gap-2 text-[12px] px-3 py-2 rounded-lg border border-white/10 bg-white/[0.03] hover:bg-white/[0.06] text-slate-300 disabled:opacity-50 transition-colors"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", refreshing && "animate-spin")} />
          Refresh
        </button>
      </div>

      {err && (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/[0.07] px-5 py-3 text-rose-200 text-sm">
          {err}
        </div>
      )}

      {!data && !err && (
        <div className="py-20 text-center text-slate-500 text-sm">Loading…</div>
      )}

      {data && (
        <div className="space-y-6">
          {/* Ledger not found warning */}
          <DataNote ledgerDir={data.ledgerDir} found={anyLedgerFound} />

          {/* Consolidated summary */}
          <SummaryStrip data={data} />

          {/* Three book cards */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <LargeBookCard
              title="Large Equities"
              subtitle="IBKR · prop-style book"
              book={data.largeEquities}
              accent="cyan"
            />
            <LargeBookCard
              title="Large Options"
              subtitle="Advanced book · local monitor"
              book={data.largeOptions}
              accent="violet"
            />
            <PdtCard pdt={data.pdtChallenge} />
          </div>

          {/* IBKR account */}
          <IbkrPanel account={data.ibkrAccount} />

          {/* Ledger paths */}
          <div className="rounded-xl border border-white/[0.05] bg-white/[0.01] px-4 py-3 space-y-1">
            <p className="text-[10px] uppercase tracking-wider text-slate-600 mb-2">Ledger files</p>
            {(
              [
                ["large_equities_book.csv", data.largeEquities],
                ["large_options_book.csv", data.largeOptions],
                ["pdt_robinhood_challenge_book.csv", data.pdtChallenge],
              ] as [string, BookSummary][]
            ).map(([name, book]) => (
              <div key={name} className="flex items-center gap-3">
                <span
                  className={cn(
                    "w-1.5 h-1.5 rounded-full shrink-0",
                    book.found ? "bg-emerald-400" : "bg-rose-400",
                  )}
                />
                <code className="text-[11px] text-slate-500 font-[family-name:var(--font-mono)]">
                  {data.ledgerDir}/{name}
                </code>
                {book.generatedAt && (
                  <span className="text-[10px] text-slate-700 ml-auto font-[family-name:var(--font-mono)]">
                    {fmtTs(book.generatedAt)}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
