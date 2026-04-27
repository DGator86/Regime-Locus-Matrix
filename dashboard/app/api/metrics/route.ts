import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

function parseCsv(text: string): Record<string, string>[] {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h.trim()] = (vals[i] ?? "").trim();
    });
    return row;
  });
}

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const DATA_DIR =
      process.env.RLM_DATA_DIR ||
      "/root/Regime-Locus-Matrix/data/processed";

    const tradeLogPath = path.join(DATA_DIR, "trade_log.csv");
    const plansPath = path.join(DATA_DIR, "universe_trade_plans.json");

    let accountBalance = 0;
    let realizedPnl = 0;
    let unrealizedPnl = 0;
    let winRate = "0.0%";
    let tradesCount = 0;
    let openCount = 0;

    if (fs.existsSync(tradeLogPath)) {
      const rows = parseCsv(fs.readFileSync(tradeLogPath, "utf8"));

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
      const closedPids = Object.keys(closedByPid);
      tradesCount = closedPids.length;

      for (const pid of closedPids) {
        const pnl = parseFloat(closedByPid[pid].unrealized_pnl || "0") || 0;
        totalRealized += pnl;
        if (pnl > 0) wins++;
      }
      realizedPnl = totalRealized;
      winRate =
        tradesCount > 0
          ? ((wins / tradesCount) * 100).toFixed(1) + "%"
          : "0.0%";

      let totalUnrealized = 0;
      for (const pid of Object.keys(latestByPid)) {
        if ((latestByPid[pid].closed || "0").trim() === "1") continue;
        openCount++;
        totalUnrealized +=
          parseFloat(latestByPid[pid].unrealized_pnl || "0") || 0;
      }
      unrealizedPnl = totalUnrealized;

      const seed = 20000;
      accountBalance = seed + realizedPnl + unrealizedPnl;
    }

    let signals: {
      sym: string;
      type: string;
      time: string;
      score: number;
    }[] = [];

    if (fs.existsSync(plansPath)) {
      const plans = JSON.parse(fs.readFileSync(plansPath, "utf8"));
      const results: any[] = plans.results || [];
      const active = results
        .filter((r: any) => r.status === "active")
        .sort(
          (a: any, b: any) =>
            (parseFloat(b.rank_score) || 0) - (parseFloat(a.rank_score) || 0)
        );

      signals = active.slice(0, 5).map((r: any) => ({
        sym: r.symbol || "?",
        type: (r.strategy || "HOLD").toUpperCase(),
        time: "Active",
        score: parseFloat(r.regime_confidence || r.confidence || "0.5") || 0.5,
      }));
    }

    const fmt = (v: number) =>
      v.toLocaleString("en-US", {
        style: "currency",
        currency: "USD",
        signDisplay: "always",
      });

    return NextResponse.json({
      accountBalance: accountBalance.toLocaleString("en-US", {
        style: "currency",
        currency: "USD",
      }),
      realizedPnl: fmt(realizedPnl),
      activeRisk:
        openCount > 0
          ? unrealizedPnl.toLocaleString("en-US", {
              style: "currency",
              currency: "USD",
              signDisplay: "always",
            })
          : "$0.00",
      winRate,
      activeSignalsCount: signals.length,
      signals:
        signals.length > 0
          ? signals
          : [{ sym: "N/A", type: "NONE", time: "Now", score: 0 }],
    });
  } catch (error) {
    console.error("Dashboard API Error:", error);
    return NextResponse.json(
      { error: "Failed to fetch metrics" },
      { status: 500 }
    );
  }
}
