"use client";

import React, { useCallback, useEffect, useState } from "react";
import { PageHero } from "@/components/PageHero";
import { cn } from "@/lib/utils";

// ─── Types ───────────────────────────────────────────────────────────────────

type Snapshot = {
  symbol: string;
  status: string;
  regimeKey: string;
  regimeConfidence: number;
  mtfRegime: string;
  mtfConfidence: number;
  predictors: Record<string, number>;
  strategyName: string;
  action: string;
  legsHuman: string;
  rationale: string;
  regimeDirection: string;
};

type OverviewPayload = {
  generatedAt: string;
  paths: Record<string, string>;
  challengeSymbols: string[];
  largeAccountOptions: {
    tickers: string[];
    symbols: Snapshot[];
    positionsOpen: Record<string, string>[];
    positionsClosed: Record<string, string>[];
    pnl: Record<string, number>;
  };
  pdtChallengeOptions: {
    tickers: string[];
    symbols: Snapshot[];
    challengeAccount: Record<string, unknown>;
    csvRecentRows: Record<string, string>[];
  };
  equities: {
    tickers: string[];
    symbols: Snapshot[];
    positionsOpen: Record<string, string>[];
    positionsClosed: Record<string, string>[];
    statePositions: Record<string, unknown>[];
    pnl: Record<string, number>;
  };
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtMoney(n: number) {
  const x = Number.isFinite(n) ? n : 0;
  const sign = x < 0 ? "−" : "";
  return `${sign}$${Math.abs(x).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function fmtNum(n: number, d = 3) {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(d);
}

function actionColors(action: string) {
  const a = (action ?? "").toUpperCase();
  if (a === "LONG" || a === "ENTER" || a === "BUY")
    return {
      text: "text-emerald-300",
      bg: "bg-emerald-500/[0.08] border-emerald-500/20",
      dot: "bg-emerald-400",
      glow: "shadow-[0_0_18px_rgba(52,211,153,0.12)]",
    };
  if (a === "SHORT" || a === "EXIT" || a === "CLOSE" || a === "SELL")
    return {
      text: "text-rose-300",
      bg: "bg-rose-500/[0.08] border-rose-500/20",
      dot: "bg-rose-400",
      glow: "shadow-[0_0_18px_rgba(244,63,94,0.12)]",
    };
  return {
    text: "text-amber-300",
    bg: "bg-amber-500/[0.07] border-amber-500/20",
    dot: "bg-amber-400",
    glow: "",
  };
}

function regimeLabel(key: string) {
  if (!key) return "—";
  return key.split("|")[0]?.trim() ?? key;
}

function confColor(conf: number) {
  if (conf >= 0.7) return "text-emerald-400";
  if (conf >= 0.4) return "text-amber-400";
  return "text-slate-500";
}

// ─── Challenge Hero ───────────────────────────────────────────────────────────

function ChallengeHero({ account }: { account: Record<string, unknown> }) {
  const loaded = Boolean(account.loaded);

  if (!loaded) {
    return (
      <div className="rounded-2xl border border-amber-500/25 bg-amber-500/[0.05] px-5 py-4 space-y-1">
        <p className="text-sm font-semibold text-amber-200">Challenge state not found</p>
        <p className="text-[12px] text-slate-500">
          Run{" "}
          <code className="text-[11px] bg-white/[0.05] px-1.5 py-0.5 rounded">
            rlm challenge --reset
          </code>{" "}
          on the VPS, then sync with{" "}
          <code className="text-[11px] bg-white/[0.05] px-1.5 py-0.5 rounded">
            powershell ./scripts/sync_dashboard_data_from_vps.ps1
          </code>
          .
        </p>
      </div>
    );
  }

  const bal = Number(account.balance ?? 0);
  const seed = Number(account.seed ?? 0);
  const tgt = account.target != null ? Number(account.target) : null;
  const retPct = Number(account.totalReturnPct ?? 0);
  const netPnl = Number(account.totalReturnDollarsNet ?? 0);
  const progressPct =
    tgt != null && tgt > seed
      ? Math.max(0, Math.min(100, ((bal - seed) / (tgt - seed)) * 100))
      : 0;

  return (
    <div className="rounded-2xl border border-cyan-500/20 bg-gradient-to-br from-cyan-500/[0.06] to-violet-600/[0.03] p-5 space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[10px] uppercase tracking-[0.2em] text-cyan-400/70 font-semibold">
            PDT Challenge Account
          </p>
          <div className="flex items-baseline gap-3 mt-1.5">
            <span className="text-3xl font-black font-[family-name:var(--font-mono)] text-white">
              {fmtMoney(bal)}
            </span>
            <span
              className={cn(
                "text-sm font-bold font-[family-name:var(--font-mono)]",
                retPct >= 0 ? "text-emerald-400" : "text-rose-400",
              )}
            >
              {retPct >= 0 ? "+" : ""}
              {fmtNum(retPct, 2)}%
            </span>
          </div>
        </div>

        <div className="flex gap-5 text-right">
          <div>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Net P&amp;L</p>
            <p
              className={cn(
                "text-sm font-bold font-[family-name:var(--font-mono)] mt-0.5",
                netPnl >= 0 ? "text-emerald-300" : "text-rose-300",
              )}
            >
              {netPnl >= 0 ? "+" : ""}
              {fmtMoney(netPnl)}
            </p>
          </div>
          <div>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider">Seed</p>
            <p className="text-sm font-bold font-[family-name:var(--font-mono)] text-slate-300 mt-0.5">
              {fmtMoney(seed)}
            </p>
          </div>
          {tgt != null && (
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider">Target</p>
              <p className="text-sm font-bold font-[family-name:var(--font-mono)] text-violet-300 mt-0.5">
                {fmtMoney(tgt)}
              </p>
            </div>
          )}
        </div>
      </div>

      {tgt != null && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] text-slate-500 font-[family-name:var(--font-mono)]">
            <span>Seed {fmtMoney(seed)}</span>
            <span className="text-cyan-400/80 font-semibold">
              {progressPct.toFixed(1)}% to target
            </span>
            <span>Target {fmtMoney(tgt)}</span>
          </div>
          <div className="relative h-2 w-full rounded-full bg-white/[0.06] overflow-hidden">
            <div
              className="absolute left-0 top-0 h-full rounded-full bg-gradient-to-r from-cyan-500 to-violet-500 transition-all duration-700"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Signal Cards ─────────────────────────────────────────────────────────────

function SignalCards({
  snapshots,
  equityMode,
}: {
  snapshots: Snapshot[];
  equityMode?: boolean;
}) {
  if (!snapshots.length) {
    return <EmptyState message="No signals in this universe yet." />;
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {snapshots.map((s) => {
        const colors = actionColors(s.action);
        const conf = s.regimeConfidence || s.mtfConfidence;
        const regime = regimeLabel(s.regimeKey);

        return (
          <div
            key={s.symbol}
            className={cn("rounded-2xl border p-4 space-y-3", colors.bg, colors.glow)}
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider font-[family-name:var(--font-mono)]">
                  {s.symbol}
                </p>
                <p className={cn("text-2xl font-black tracking-tight mt-0.5", colors.text)}>
                  {(s.action || "—").toUpperCase()}
                </p>
              </div>
              <div className="text-right">
                <p className="text-[10px] text-slate-500 uppercase tracking-wider">Confidence</p>
                <p className={cn("text-sm font-bold font-[family-name:var(--font-mono)] mt-0.5", confColor(conf))}>
                  {Number.isFinite(conf) ? `${(conf * 100).toFixed(0)}%` : "—"}
                </p>
              </div>
            </div>

            <div>
              <p className="text-[12px] font-semibold text-slate-200">
                {s.strategyName || s.mtfRegime || "—"}
              </p>
              {!equityMode && s.legsHuman && (
                <p className="text-[10px] text-slate-500 font-[family-name:var(--font-mono)] mt-1 break-all leading-relaxed">
                  {s.legsHuman}
                </p>
              )}
            </div>

            <div className="flex items-center gap-2 pt-1 border-t border-white/[0.06]">
              <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", colors.dot)} />
              <span
                className="text-[10px] text-slate-500 truncate"
                title={s.regimeKey}
              >
                {regime}
              </span>
              {s.regimeDirection && (
                <span className="text-[10px] text-slate-600 ml-auto capitalize">
                  {s.regimeDirection}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Regime Table ─────────────────────────────────────────────────────────────

function RegimeTable({ snapshots }: { snapshots: Snapshot[] }) {
  if (!snapshots.length) return null;

  return (
    <div className="rounded-2xl border border-white/[0.06] overflow-hidden">
      <div className="px-4 py-3 border-b border-white/[0.05]">
        <p className="text-[11px] font-semibold uppercase tracking-[0.15em] text-slate-400">
          Regime &amp; Factors
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="border-b border-white/[0.04] text-left">
              <th className="px-3 py-2.5 text-slate-500">Symbol</th>
              <th className="px-3 py-2.5 text-slate-500">Regime</th>
              <th className="px-3 py-2.5 text-slate-500">Conf.</th>
              <th
                className="px-3 py-2.5 text-slate-500 cursor-help"
                title="Direction factor — positive is bullish"
              >
                M_D
              </th>
              <th
                className="px-3 py-2.5 text-slate-500 cursor-help"
                title="Volatility factor"
              >
                M_V
              </th>
              <th
                className="px-3 py-2.5 text-slate-500 cursor-help"
                title="Multi-timeframe regime label"
              >
                MTF
              </th>
            </tr>
          </thead>
          <tbody>
            {snapshots.map((s) => {
              const conf = s.regimeConfidence || s.mtfConfidence;
              const md = s.predictors?.M_D ?? NaN;
              return (
                <tr key={s.symbol} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                  <td className="px-3 py-2 font-semibold text-slate-200">{s.symbol}</td>
                  <td
                    className="px-3 py-2 text-slate-400 font-[family-name:var(--font-mono)] text-[10px] max-w-[120px] truncate"
                    title={s.regimeKey}
                  >
                    {regimeLabel(s.regimeKey) || "—"}
                  </td>
                  <td className={cn("px-3 py-2 font-[family-name:var(--font-mono)]", confColor(conf))}>
                    {Number.isFinite(conf) ? `${(conf * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td
                    className={cn(
                      "px-3 py-2 font-[family-name:var(--font-mono)]",
                      md >= 0 ? "text-cyan-400" : "text-violet-400",
                    )}
                  >
                    {fmtNum(md, 2)}
                  </td>
                  <td className="px-3 py-2 font-[family-name:var(--font-mono)] text-slate-400">
                    {fmtNum(s.predictors?.M_V ?? NaN, 2)}
                  </td>
                  <td className="px-3 py-2 text-slate-500 text-[10px] truncate">
                    {s.mtfRegime || "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Position Tables ──────────────────────────────────────────────────────────

function ChallengeOpenTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows.length) return <EmptyState message="No open positions." />;

  return (
    <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="border-b border-white/[0.05] text-left">
            <th className="px-3 py-2.5 text-slate-500">Symbol</th>
            <th className="px-3 py-2.5 text-slate-500">Type</th>
            <th className="px-3 py-2.5 text-slate-500">Strike</th>
            <th className="px-3 py-2.5 text-slate-500">Qty</th>
            <th className="px-3 py-2.5 text-slate-500">Unrealized P&amp;L</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const pnl = Number((r as { unrealised_pnl?: number }).unrealised_pnl ?? 0);
            return (
              <tr key={i} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                <td className="px-3 py-2 font-semibold text-slate-200">{String(r.symbol)}</td>
                <td className="px-3 py-2 text-slate-400">{String(r.option_type)}</td>
                <td className="px-3 py-2 font-[family-name:var(--font-mono)]">{String(r.strike)}</td>
                <td className="px-3 py-2">{String(r.qty)}</td>
                <td
                  className={cn(
                    "px-3 py-2 font-[family-name:var(--font-mono)] font-semibold",
                    pnl >= 0 ? "text-emerald-300" : "text-rose-300",
                  )}
                >
                  {fmtMoney(pnl)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ChallengeClosedTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows.length) return <EmptyState message="No closed trades yet." />;

  return (
    <div className="overflow-x-auto rounded-xl border border-white/[0.06] max-h-[300px] overflow-y-auto">
      <table className="w-full text-[11px]">
        <thead className="sticky top-0 bg-[#0a0e14]">
          <tr className="border-b border-white/[0.05] text-left">
            <th className="px-3 py-2.5 text-slate-500">Symbol</th>
            <th className="px-3 py-2.5 text-slate-500">Exit date</th>
            <th className="px-3 py-2.5 text-slate-500">P&amp;L</th>
            <th className="px-3 py-2.5 text-slate-500">Reason</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const pnl = Number((r as { pnl?: number }).pnl ?? 0);
            return (
              <tr key={i} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                <td className="px-3 py-2 font-semibold text-slate-200">{String(r.symbol)}</td>
                <td className="px-3 py-2 text-slate-400 font-[family-name:var(--font-mono)]">
                  {String(r.exit_date)}
                </td>
                <td
                  className={cn(
                    "px-3 py-2 font-[family-name:var(--font-mono)] font-semibold",
                    pnl >= 0 ? "text-emerald-300" : "text-rose-300",
                  )}
                >
                  {fmtMoney(pnl)}
                </td>
                <td className="px-3 py-2 text-slate-500">{String(r.exit_reason)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function OptionsPositionsTable({
  rows,
  closed,
}: {
  rows: Record<string, string>[];
  closed?: boolean;
}) {
  if (!rows.length)
    return (
      <EmptyState message={closed ? "No closed positions." : "No open positions."} />
    );

  const pv = (v: string | undefined) => {
    const n = parseFloat(v ?? "");
    return Number.isFinite(n) ? n : 0;
  };

  return (
    <div className="overflow-x-auto rounded-xl border border-white/[0.06] max-h-[300px] overflow-y-auto">
      <table className="w-full text-[11px]">
        <thead className="sticky top-0 bg-[#0a0e14]">
          <tr className="border-b border-white/[0.05] text-left">
            <th className="px-3 py-2.5 text-slate-500">Symbol</th>
            <th className="px-3 py-2.5 text-slate-500">Strategy</th>
            <th className="px-3 py-2.5 text-slate-500">P&amp;L</th>
            <th className="px-3 py-2.5 text-slate-500">Signal</th>
            <th className="px-3 py-2.5 text-slate-500">DTE</th>
            <th className="px-3 py-2.5 text-slate-500">When</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const pnl = pv(r.unrealized_pnl);
            return (
              <tr key={`${r.plan_id}-${i}`} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                <td className="px-3 py-2 font-semibold text-slate-200">{r.symbol}</td>
                <td className="px-3 py-2 text-slate-400 truncate max-w-[140px]">{r.strategy}</td>
                <td
                  className={cn(
                    "px-3 py-2 font-[family-name:var(--font-mono)] font-semibold",
                    pnl >= 0 ? "text-emerald-300" : "text-rose-300",
                  )}
                >
                  {fmtMoney(pnl)}
                </td>
                <td className="px-3 py-2 text-slate-500">{closed ? "closed" : r.signal}</td>
                <td className="px-3 py-2 font-[family-name:var(--font-mono)] text-slate-400">{r.dte}</td>
                <td className="px-3 py-2 text-slate-600 whitespace-nowrap font-[family-name:var(--font-mono)]">
                  {r.timestamp_utc?.slice(0, 16) ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function EquityPositionsTable({
  rows,
  closed,
}: {
  rows: Record<string, string>[];
  closed?: boolean;
}) {
  if (!rows.length)
    return (
      <EmptyState
        message={closed ? "No closed equity positions." : "No open equity positions."}
      />
    );

  const pv = (v: string | undefined) => {
    const n = parseFloat(v ?? "");
    return Number.isFinite(n) ? n : 0;
  };

  return (
    <div className="overflow-x-auto rounded-xl border border-white/[0.06] max-h-[300px] overflow-y-auto">
      <table className="w-full text-[11px]">
        <thead className="sticky top-0 bg-[#0a0e14]">
          <tr className="border-b border-white/[0.05] text-left">
            <th className="px-3 py-2.5 text-slate-500">Symbol</th>
            <th className="px-3 py-2.5 text-slate-500">Strategy</th>
            <th className="px-3 py-2.5 text-slate-500">P&amp;L</th>
            <th className="px-3 py-2.5 text-slate-500">Qty</th>
            <th className="px-3 py-2.5 text-slate-500">When</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const pnl = pv(r.unrealized_pnl);
            return (
              <tr
                key={`${r.plan_id}-${i}`}
                className="border-b border-white/[0.03] hover:bg-white/[0.02]"
              >
                <td className="px-3 py-2 font-semibold text-slate-200">{r.symbol}</td>
                <td className="px-3 py-2 text-slate-400 truncate max-w-[140px]">{r.strategy}</td>
                <td
                  className={cn(
                    "px-3 py-2 font-[family-name:var(--font-mono)] font-semibold",
                    pnl >= 0 ? "text-emerald-300" : "text-rose-300",
                  )}
                >
                  {fmtMoney(pnl)}
                </td>
                <td className="px-3 py-2 text-slate-400">{r.quantity}</td>
                <td className="px-3 py-2 text-slate-600 whitespace-nowrap font-[family-name:var(--font-mono)]">
                  {r.timestamp_utc?.slice(0, 16) ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─── P&L Strip ────────────────────────────────────────────────────────────────

function PnlStrip({
  daily,
  weekly,
  realized,
  openMtm,
  combined,
}: {
  daily: number;
  weekly: number;
  realized: number;
  openMtm: number;
  combined: number;
}) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <PnlCard label="Today" value={daily} sub="Realized (US/Eastern)" />
      <PnlCard label="7-Day Rolling" value={weekly} sub="Rolling 7×24 h window" />
      <PnlCard label="All-time Realized" value={realized} sub="Closed trades only" />
      <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/[0.04] p-4">
        <p className="text-[10px] uppercase tracking-wider text-emerald-300/70">
          Total (Realized + MTM)
        </p>
        <p
          className={cn(
            "text-xl font-black font-[family-name:var(--font-mono)] mt-1",
            combined >= 0 ? "text-emerald-200" : "text-rose-300",
          )}
        >
          {fmtMoney(combined)}
        </p>
        <p className="text-[10px] text-slate-600 mt-1 font-[family-name:var(--font-mono)]">
          Open MTM {fmtMoney(openMtm)}
        </p>
      </div>
    </div>
  );
}

function PnlCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: number;
  sub: string;
}) {
  return (
    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-4">
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p
        className={cn(
          "text-xl font-black font-[family-name:var(--font-mono)] mt-1",
          value >= 0 ? "text-emerald-300" : "text-rose-300",
        )}
      >
        {fmtMoney(value)}
      </p>
      <p className="text-[10px] text-slate-600 mt-1">{sub}</p>
    </div>
  );
}

// ─── Layout Primitives ────────────────────────────────────────────────────────

function SectionBlock({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="space-y-3">
      <h3 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
        {title}
      </h3>
      {children}
    </section>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-dashed border-white/[0.07] py-8 text-center text-slate-600 text-sm">
      {message}
    </div>
  );
}

// ─── Tab Contents ─────────────────────────────────────────────────────────────

function ChallengeTabContent({ data }: { data: OverviewPayload }) {
  const { pdtChallengeOptions } = data;
  const account = pdtChallengeOptions.challengeAccount;
  const openPositions = (account.openPositions ?? []) as Record<string, unknown>[];
  const closedTrades = (account.closedTrades ?? []) as Record<string, unknown>[];

  const daily = Number(account.dailyRealized ?? 0);
  const weekly = Number(account.weeklyRealizedRolling7d ?? 0);
  const realized = Number(account.totalReturnDollarsNet ?? 0);
  const openMtm = openPositions.reduce(
    (s, p) => s + Number((p as { unrealised_pnl?: number }).unrealised_pnl ?? 0),
    0,
  );
  const combined = realized + openMtm;

  return (
    <div className="space-y-6">
      <ChallengeHero account={account} />

      {pdtChallengeOptions.symbols.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2">
            <SectionBlock title="Current Signals">
              <SignalCards snapshots={pdtChallengeOptions.symbols} />
            </SectionBlock>
          </div>
          <div>
            <SectionBlock title="Regime &amp; Factors">
              <RegimeTable snapshots={pdtChallengeOptions.symbols} />
            </SectionBlock>
          </div>
        </div>
      )}

      <SectionBlock title="Open Positions">
        <ChallengeOpenTable rows={openPositions} />
      </SectionBlock>

      <SectionBlock title="Closed Trades">
        <ChallengeClosedTable rows={closedTrades} />
      </SectionBlock>

      <SectionBlock title="Performance">
        <PnlStrip
          daily={daily}
          weekly={weekly}
          realized={realized}
          openMtm={openMtm}
          combined={combined}
        />
      </SectionBlock>
    </div>
  );
}

function LargeAccountTabContent({ data }: { data: OverviewPayload }) {
  const { largeAccountOptions } = data;
  const pnl = largeAccountOptions.pnl;

  return (
    <div className="space-y-6">
      {largeAccountOptions.symbols.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2">
            <SectionBlock title="Current Signals">
              <SignalCards snapshots={largeAccountOptions.symbols} />
            </SectionBlock>
          </div>
          <div>
            <SectionBlock title="Regime &amp; Factors">
              <RegimeTable snapshots={largeAccountOptions.symbols} />
            </SectionBlock>
          </div>
        </div>
      )}

      <SectionBlock title="Open Positions">
        <OptionsPositionsTable rows={largeAccountOptions.positionsOpen} />
      </SectionBlock>

      <SectionBlock title="Closed Positions">
        <OptionsPositionsTable rows={largeAccountOptions.positionsClosed} closed />
      </SectionBlock>

      <SectionBlock title="Performance">
        <PnlStrip
          daily={pnl.dailyRealized ?? 0}
          weekly={pnl.weeklyRealizedRolling7d ?? 0}
          realized={pnl.allTimeRealizedClosed ?? 0}
          openMtm={pnl.openMarkToMarket ?? 0}
          combined={pnl.combinedOpenPlusRealized ?? 0}
        />
      </SectionBlock>
    </div>
  );
}

function EquitiesTabContent({ data }: { data: OverviewPayload }) {
  const { equities } = data;
  const pnl = equities.pnl;

  return (
    <div className="space-y-6">
      {equities.symbols.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2">
            <SectionBlock title="Current Signals">
              <SignalCards snapshots={equities.symbols} equityMode />
            </SectionBlock>
          </div>
          <div>
            <SectionBlock title="Regime &amp; Factors">
              <RegimeTable snapshots={equities.symbols} />
            </SectionBlock>
          </div>
        </div>
      )}

      <SectionBlock title="Open Positions">
        <EquityPositionsTable rows={equities.positionsOpen} />
      </SectionBlock>

      <SectionBlock title="Closed Positions">
        <EquityPositionsTable rows={equities.positionsClosed} closed />
      </SectionBlock>

      <SectionBlock title="Performance">
        <PnlStrip
          daily={pnl.dailyRealized ?? 0}
          weekly={pnl.weeklyRealizedRolling7d ?? 0}
          realized={pnl.allTimeRealizedClosed ?? 0}
          openMtm={pnl.openMarkToMarket ?? 0}
          combined={pnl.combinedOpenPlusRealized ?? 0}
        />
      </SectionBlock>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function TradingOverviewPage() {
  const [tab, setTab] = useState<"opt-large" | "opt-challenge" | "equities">("opt-large");
  const [data, setData] = useState<OverviewPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async (showSpinner = false) => {
    if (showSpinner) setRefreshing(true);
    try {
      const res = await fetch("/api/trading-overview");
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error || res.statusText);
      setData(json);
      setErr(null);
      const loaded = Boolean(
        (json?.pdtChallengeOptions?.challengeAccount as { loaded?: boolean })?.loaded,
      );
      if (loaded) {
        setTab((prev) => (prev === "opt-large" ? "opt-challenge" : prev));
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(() => load(), 30000);
    return () => clearInterval(t);
  }, [load]);

  const challengeLoaded = Boolean(
    (data?.pdtChallengeOptions?.challengeAccount as { loaded?: boolean })?.loaded,
  );

  const TABS = [
    { id: "opt-large" as const, label: "Large Account" },
    { id: "opt-challenge" as const, label: "PDT Challenge", dot: challengeLoaded },
    { id: "equities" as const, label: "Equities" },
  ];

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      <PageHero
        eyebrow="Operations"
        title="Trading"
        subtitle="Live signals, positions, and P&amp;L for options and equity strategies. Refreshes every 30 s."
      />

      <div className="flex flex-wrap gap-3 items-center justify-between">
        <div className="flex gap-1 p-1 rounded-xl bg-white/[0.03] border border-white/[0.07]">
          {TABS.map(({ id, label, dot }) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={cn(
                "px-4 py-2 rounded-lg text-[13px] font-medium transition-all flex items-center gap-2",
                tab === id
                  ? "bg-violet-600/40 text-violet-100 border border-violet-500/30 shadow-[0_0_12px_rgba(139,92,246,0.18)]"
                  : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]",
              )}
            >
              {label}
              {dot && (
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.8)]" />
              )}
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={() => load(true)}
          disabled={refreshing}
          className="flex items-center gap-2 text-[12px] px-3 py-2 rounded-lg border border-white/10 bg-white/[0.03] hover:bg-white/[0.06] text-slate-300 disabled:opacity-50 transition-colors"
        >
          <span
            className={cn(
              "inline-block text-[14px] leading-none",
              refreshing && "animate-spin",
            )}
          >
            ⟳
          </span>
          Refresh
        </button>
      </div>

      {err && (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/[0.07] px-5 py-3 text-rose-200 text-sm">
          {err}
        </div>
      )}

      {!data && !err && (
        <div className="py-16 text-center text-slate-500 text-sm">Loading…</div>
      )}

      {data && tab === "opt-challenge" && <ChallengeTabContent data={data} />}
      {data && tab === "opt-large" && <LargeAccountTabContent data={data} />}
      {data && tab === "equities" && <EquitiesTabContent data={data} />}
    </div>
  );
}
