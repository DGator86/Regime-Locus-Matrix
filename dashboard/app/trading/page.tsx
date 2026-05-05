"use client";

import React, { useCallback, useEffect, useState } from "react";
import { PageHero } from "@/components/PageHero";

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

function fmtMoney(n: number) {
  const x = Number.isFinite(n) ? n : 0;
  const sign = x < 0 ? "−" : "";
  return `${sign}$${Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtNum(n: number, d = 3) {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(d);
}

export default function TradingOverviewPage() {
  const [tab, setTab] = useState<"opt-large" | "opt-challenge" | "equities">("opt-large");
  const [data, setData] = useState<OverviewPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/trading-overview");
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error || res.statusText);
      setData(json);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, [load]);

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      <PageHero
        eyebrow="Operations"
        title="Trading overview"
        subtitle="Options (large account vs PDT challenge) and equities — fed from on-disk universe plans, trade logs, and challenge state. Refresh every 30s."
      />

      <div className="flex flex-wrap gap-2 items-center justify-between">
        <div className="flex gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.08]">
          {(
            [
              ["opt-large", "Options · Large account"],
              ["opt-challenge", "Options · PDT challenge"],
              ["equities", "Equities"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setTab(id)}
              className={`px-4 py-2 rounded-lg text-[13px] font-medium transition-colors ${
                tab === id
                  ? "bg-violet-600/40 text-violet-100 border border-violet-500/30"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => load()}
          className="text-[12px] px-3 py-2 rounded-lg border border-white/10 bg-white/[0.03] hover:bg-white/[0.06] text-slate-300"
        >
          Refresh now
        </button>
      </div>

      {err && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-red-200 text-sm">{err}</div>
      )}

      {data && (
        <p className="text-[11px] text-slate-500 font-[family-name:var(--font-mono)]">
          Generated {new Date(data.generatedAt).toLocaleString()} · processedDir {data.paths.processedDir}
        </p>
      )}

      {!data && !err && <p className="text-slate-400 text-sm">Loading…</p>}

      {data && tab === "opt-large" && (
        <TradingSections
          title="Options — Large account"
          snapshots={data.largeAccountOptions.symbols}
          positionsOpen={data.largeAccountOptions.positionsOpen}
          positionsClosed={data.largeAccountOptions.positionsClosed}
          pnl={data.largeAccountOptions.pnl}
          tickers={data.largeAccountOptions.tickers}
          extraNote="Universe rows exclude challenge symbols when other symbols exist; otherwise shows full universe. P&amp;L from monitor trade_log (latest row per plan)."
        />
      )}

      {data && tab === "opt-challenge" && (
        <div className="space-y-8">
          <ChallengeAccountSection account={data.pdtChallengeOptions.challengeAccount} />
          <TradingSections
            title="Options — PDT challenge (regime / strategy)"
            snapshots={data.pdtChallengeOptions.symbols}
            positionsOpen={[]} 
            positionsClosed={[]} 
            pnl={{
              dailyRealized: Number(data.pdtChallengeOptions.challengeAccount.dailyRealized ?? 0),
              weeklyRealizedRolling7d: Number(data.pdtChallengeOptions.challengeAccount.weeklyRealizedRolling7d ?? 0),
              allTimeRealizedClosed: Number(data.pdtChallengeOptions.challengeAccount.totalReturnDollarsNet ?? 0),
              openMarkToMarket: (data.pdtChallengeOptions.challengeAccount.openPositions as Record<string, unknown>[] | undefined)?.reduce(
                (s, p) => s + Number((p as { unrealised_pnl?: number }).unrealised_pnl ?? 0),
                0,
              ) ?? 0,
              combinedOpenPlusRealized: Number(data.pdtChallengeOptions.challengeAccount.totalReturnDollarsNet ?? 0),
            }}
            tickers={data.pdtChallengeOptions.tickers}
            extraNote={`Challenge universe symbols: ${data.challengeSymbols.join(", ")} (override with RLM_CHALLENGE_SYMBOLS).`}
            openFromChallenge={(data.pdtChallengeOptions.challengeAccount.openPositions || []) as Record<string, unknown>[]}
            closedFromChallenge={(data.pdtChallengeOptions.challengeAccount.closedTrades || []) as Record<string, unknown>[]}
          />
        </div>
      )}

      {data && tab === "equities" && (
        <TradingSections
          title="Equities"
          snapshots={data.equities.symbols}
          positionsOpen={data.equities.positionsOpen}
          positionsClosed={data.equities.positionsClosed}
          pnl={data.equities.pnl}
          tickers={data.equities.tickers}
          extraNote="Direction/confidence from universe regime_direction + regime confidence where present."
          equityMode
        />
      )}
    </div>
  );
}

function ChallengeAccountSection({ account }: { account: Record<string, unknown> }) {
  const loaded = Boolean(account.loaded);
  if (!loaded) {
    return (
      <div className="rounded-xl border border-amber-500/25 bg-amber-500/5 px-4 py-3 text-amber-100 text-sm">
        No challenge state at <code className="text-[11px]">data/challenge/state.json</code> — run{" "}
        <code className="text-[11px]">rlm challenge --reset</code> on the machine that owns this data directory.
      </div>
    );
  }
  const bal = Number(account.balance ?? 0);
  const seed = Number(account.seed ?? 0);
  const tgt = account.target != null ? Number(account.target) : null;
  const pct = Number(account.totalReturnPct ?? 0);
  return (
    <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/[0.06] p-4 space-y-2">
      <h3 className="text-sm font-semibold text-cyan-100">PDT challenge account</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[13px] font-[family-name:var(--font-mono)]">
        <div>
          <div className="text-slate-500 text-[10px] uppercase tracking-wider">Balance</div>
          <div className="text-slate-100">{fmtMoney(bal)}</div>
        </div>
        <div>
          <div className="text-slate-500 text-[10px] uppercase tracking-wider">Seed → Target</div>
          <div className="text-slate-100">
            {fmtMoney(seed)}
            {tgt != null ? ` → ${fmtMoney(tgt)}` : ""}
          </div>
        </div>
        <div>
          <div className="text-slate-500 text-[10px] uppercase tracking-wider">Return %</div>
          <div className={pct >= 0 ? "text-emerald-300" : "text-red-300"}>{fmtNum(pct, 2)}%</div>
        </div>
        <div>
          <div className="text-slate-500 text-[10px] uppercase tracking-wider">Net P&amp;L (balance − seed)</div>
          <div className="text-slate-100">{fmtMoney(Number(account.totalReturnDollarsNet ?? 0))}</div>
        </div>
      </div>
    </div>
  );
}

function TradingSections({
  title,
  snapshots,
  positionsOpen,
  positionsClosed,
  pnl,
  tickers,
  extraNote,
  equityMode,
  openFromChallenge,
  closedFromChallenge,
}: {
  title: string;
  snapshots: Snapshot[];
  positionsOpen: Record<string, string>[];
  positionsClosed: Record<string, string>[];
  pnl: Record<string, number>;
  tickers: string[];
  extraNote?: string;
  equityMode?: boolean;
  openFromChallenge?: Record<string, unknown>[];
  closedFromChallenge?: Record<string, unknown>[];
}) {
  return (
    <div className="space-y-8">
      <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
      {extraNote && <p className="text-[12px] text-slate-500">{extraNote}</p>}

      <Section num={1} title="Tickers">
        <div className="flex flex-wrap gap-2">
          {tickers.length === 0 ? (
            <span className="text-slate-500 text-sm">No symbols in this view.</span>
          ) : (
            tickers.map((t) => (
              <span
                key={t}
                className="px-2.5 py-1 rounded-lg bg-white/[0.06] border border-white/[0.08] text-[12px] font-[family-name:var(--font-mono)] text-slate-200"
              >
                {t}
              </span>
            ))
          )}
        </div>
      </Section>

      <Section num={2} title="MTF regime state & confidence">
        <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-white/[0.06] text-left text-slate-500">
                <th className="px-3 py-2">Symbol</th>
                <th className="px-3 py-2">Pipeline regime</th>
                <th className="px-3 py-2">Regime conf.</th>
                <th className="px-3 py-2">MTF label</th>
                <th className="px-3 py-2">MTF conf.</th>
                <th className="px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {snapshots.map((s) => (
                <tr key={s.symbol} className="border-b border-white/[0.04] hover:bg-white/[0.02]">
                  <td className="px-3 py-2 font-medium text-slate-200">{s.symbol}</td>
                  <td className="px-3 py-2 text-slate-300 font-[family-name:var(--font-mono)] text-[11px]">{s.regimeKey || "—"}</td>
                  <td className="px-3 py-2">{fmtNum(s.regimeConfidence)}</td>
                  <td className="px-3 py-2 text-slate-300">{s.mtfRegime || "—"}</td>
                  <td className="px-3 py-2">{fmtNum(s.mtfConfidence)}</td>
                  <td className="px-3 py-2 text-slate-400">{s.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section num={3} title="MTF regime predictors (metadata factors)">
        <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-white/[0.06] text-left text-slate-500">
                <th className="px-3 py-2">Symbol</th>
                <th className="px-3 py-2">M_D</th>
                <th className="px-3 py-2">M_V</th>
                <th className="px-3 py-2">M_L</th>
                <th className="px-3 py-2">M_G</th>
                <th className="px-3 py-2">Trend</th>
                <th className="px-3 py-2">R_trans</th>
              </tr>
            </thead>
            <tbody>
              {snapshots.map((s) => (
                <tr key={s.symbol} className="border-b border-white/[0.04]">
                  <td className="px-3 py-2 font-medium">{s.symbol}</td>
                  <td className="px-3 py-2 font-[family-name:var(--font-mono)]">{fmtNum(s.predictors?.M_D ?? 0)}</td>
                  <td className="px-3 py-2 font-[family-name:var(--font-mono)]">{fmtNum(s.predictors?.M_V ?? 0)}</td>
                  <td className="px-3 py-2 font-[family-name:var(--font-mono)]">{fmtNum(s.predictors?.M_L ?? 0)}</td>
                  <td className="px-3 py-2 font-[family-name:var(--font-mono)]">{fmtNum(s.predictors?.M_G ?? 0)}</td>
                  <td className="px-3 py-2 font-[family-name:var(--font-mono)]">{fmtNum(s.predictors?.M_trend_strength ?? 0)}</td>
                  <td className="px-3 py-2 font-[family-name:var(--font-mono)]">{fmtNum(s.predictors?.M_R_trans ?? 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {!equityMode && (
        <Section num={4} title="Current best options strategy & legs">
          <div className="space-y-3">
            {snapshots.map((s) => (
              <div key={s.symbol} className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
                <div className="text-[13px] font-semibold text-slate-100">
                  {s.symbol}{" "}
                  <span className="text-violet-300 font-normal">{s.strategyName || "—"}</span>{" "}
                  <span className="text-slate-500 text-[11px]">{s.action}</span>
                </div>
                <div className="text-[11px] text-slate-400 mt-1 font-[family-name:var(--font-mono)] break-all">{s.legsHuman}</div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {equityMode && (
        <Section num={4} title="Current direction & confidence">
          <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-b border-white/[0.06] text-left text-slate-500">
                  <th className="px-3 py-2">Symbol</th>
                  <th className="px-3 py-2">Direction</th>
                  <th className="px-3 py-2">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {snapshots.map((s) => (
                  <tr key={s.symbol} className="border-b border-white/[0.04]">
                    <td className="px-3 py-2">{s.symbol}</td>
                    <td className="px-3 py-2 capitalize text-slate-200">{s.regimeDirection || inferDirection(s.regimeKey)}</td>
                    <td className="px-3 py-2">{fmtNum(s.regimeConfidence || s.mtfConfidence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      <Section num={5} title="Current positions — position P&amp;L">
        {openFromChallenge && openFromChallenge.length > 0 ? (
          <ChallengeOpenTable rows={openFromChallenge} />
        ) : positionsOpen.length > 0 ? (
          <PositionsTable rows={positionsOpen} equityMode={equityMode} />
        ) : (
          <EmptyRows />
        )}
      </Section>

      <Section num={6} title="Closed positions">
        {closedFromChallenge && closedFromChallenge.length > 0 ? (
          <ChallengeClosedTable rows={closedFromChallenge} />
        ) : positionsClosed.length > 0 ? (
          <PositionsTable rows={positionsClosed} equityMode={equityMode} closed />
        ) : (
          <EmptyRows />
        )}
      </Section>

      <Section num={7} title="Daily P&amp;L">
        <PnlCallout value={pnl.dailyRealized} subtitle="Realized from exits whose timestamp is today (US/Eastern calendar day)." />
      </Section>

      <Section num={8} title="Weekly P&amp;L">
        <PnlCallout
          value={pnl.weeklyRealizedRolling7d}
          subtitle="Rolling 7×24h window from exit timestamps (not ISO calendar week)."
        />
      </Section>

      <Section num={9} title="All-time P&amp;L">
        <div className="grid md:grid-cols-2 gap-3">
          <div className="rounded-xl border border-white/[0.06] p-3 bg-white/[0.02]">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Realized (closed trades)</div>
            <div className="text-xl font-[family-name:var(--font-mono)] text-slate-100">{fmtMoney(pnl.allTimeRealizedClosed ?? 0)}</div>
          </div>
          <div className="rounded-xl border border-white/[0.06] p-3 bg-white/[0.02]">
            <div className="text-[10px] uppercase tracking-wider text-slate-500">Open mark-to-market</div>
            <div className="text-xl font-[family-name:var(--font-mono)] text-slate-100">{fmtMoney(pnl.openMarkToMarket ?? 0)}</div>
          </div>
          <div className="rounded-xl border border-emerald-500/20 p-3 bg-emerald-500/[0.04] md:col-span-2">
            <div className="text-[10px] uppercase tracking-wider text-emerald-300/80">Combined (realized + open MTM)</div>
            <div className="text-2xl font-[family-name:var(--font-mono)] text-emerald-100">{fmtMoney(pnl.combinedOpenPlusRealized ?? 0)}</div>
          </div>
        </div>
      </Section>
    </div>
  );
}

function inferDirection(regimeKey: string): string {
  const k = regimeKey.toLowerCase();
  if (k.includes("bull")) return "bull";
  if (k.includes("bear")) return "bear";
  return "—";
}

function Section({ num, title, children }: { num: number; title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2">
      <h3 className="text-[13px] font-semibold text-slate-200">
        <span className="text-violet-400 mr-2">{num}.</span>
        {title}
      </h3>
      {children}
    </section>
  );
}

function EmptyRows() {
  return <p className="text-slate-500 text-sm">No rows.</p>;
}

function PnlCallout({ value, subtitle }: { value: number; subtitle: string }) {
  return (
    <div className="rounded-xl border border-white/[0.06] p-4 bg-white/[0.02]">
      <div className={`text-2xl font-[family-name:var(--font-mono)] ${value >= 0 ? "text-emerald-300" : "text-red-300"}`}>
        {fmtMoney(value)}
      </div>
      <p className="text-[11px] text-slate-500 mt-2">{subtitle}</p>
    </div>
  );
}

function PositionsTable({
  rows,
  equityMode,
  closed,
}: {
  rows: Record<string, string>[];
  equityMode?: boolean;
  closed?: boolean;
}) {
  const pv = (v: string | undefined) => {
    const n = parseFloat(v ?? "");
    return Number.isFinite(n) ? n : 0;
  };
  return (
    <div className="overflow-x-auto rounded-xl border border-white/[0.06] max-h-[320px] overflow-y-auto">
      <table className="w-full text-[11px]">
        <thead className="sticky top-0 bg-[#0a0f14]">
          <tr className="border-b border-white/[0.06] text-left text-slate-500">
            <th className="px-2 py-2">Symbol</th>
            <th className="px-2 py-2">Strategy</th>
            <th className="px-2 py-2">P&amp;L</th>
            <th className="px-2 py-2">Signal</th>
            {!equityMode && <th className="px-2 py-2">DTE</th>}
            {equityMode && <th className="px-2 py-2">Qty</th>}
            <th className="px-2 py-2">When</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.plan_id}-${i}`} className="border-b border-white/[0.04]">
              <td className="px-2 py-1.5 font-medium">{r.symbol}</td>
              <td className="px-2 py-1.5 truncate max-w-[140px]">{r.strategy}</td>
              <td className={`px-2 py-1.5 font-[family-name:var(--font-mono)] ${pv(r.unrealized_pnl) >= 0 ? "text-emerald-300" : "text-red-300"}`}>
                {fmtMoney(pv(r.unrealized_pnl))}
              </td>
              <td className="px-2 py-1.5 text-slate-400">{closed ? "closed" : r.signal}</td>
              {!equityMode && <td className="px-2 py-1.5">{r.dte}</td>}
              {equityMode && <td className="px-2 py-1.5">{r.quantity}</td>}
              <td className="px-2 py-1.5 text-slate-500 whitespace-nowrap">{r.timestamp_utc?.slice(0, 19) ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChallengeOpenTable({ rows }: { rows: Record<string, unknown>[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="border-b border-white/[0.06] text-left text-slate-500">
            <th className="px-2 py-2">Symbol</th>
            <th className="px-2 py-2">Type</th>
            <th className="px-2 py-2">Strike</th>
            <th className="px-2 py-2">Qty</th>
            <th className="px-2 py-2">Unrealized</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-white/[0.04]">
              <td className="px-2 py-1.5">{String(r.symbol)}</td>
              <td className="px-2 py-1.5">{String(r.option_type)}</td>
              <td className="px-2 py-1.5">{String(r.strike)}</td>
              <td className="px-2 py-1.5">{String(r.qty)}</td>
              <td className="px-2 py-1.5 font-[family-name:var(--font-mono)]">{fmtMoney(Number(r.unrealised_pnl ?? 0))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChallengeClosedTable({ rows }: { rows: Record<string, unknown>[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-white/[0.06] max-h-[280px] overflow-y-auto">
      <table className="w-full text-[11px]">
        <thead className="sticky top-0 bg-[#0a0f14]">
          <tr className="border-b border-white/[0.06] text-left text-slate-500">
            <th className="px-2 py-2">Symbol</th>
            <th className="px-2 py-2">Exit</th>
            <th className="px-2 py-2">P&amp;L</th>
            <th className="px-2 py-2">Reason</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-white/[0.04]">
              <td className="px-2 py-1.5">{String(r.symbol)}</td>
              <td className="px-2 py-1.5">{String(r.exit_date)}</td>
              <td className={`px-2 py-1.5 font-[family-name:var(--font-mono)] ${Number(r.pnl) >= 0 ? "text-emerald-300" : "text-red-300"}`}>
                {fmtMoney(Number(r.pnl ?? 0))}
              </td>
              <td className="px-2 py-1.5 text-slate-400">{String(r.exit_reason)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
