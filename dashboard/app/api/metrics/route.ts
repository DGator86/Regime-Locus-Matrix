import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    // Paths to the live data on the VPS
    // We use environment variables or fallback to the standard deployment path
    const DATA_DIR = process.env.RLM_DATA_DIR || "/opt/regime-locus/data/processed";
    const positionsPath = path.join(DATA_DIR, "equity_positions_state.json");
    const monitorPath = path.join(DATA_DIR, "trade_monitor_state.json");

    let accountBalance = 0;
    let realizedPnl = 0;
    let activeSignalsCount = 0;
    let winRate = 0;

    // 1. Parse Positions for P/L and Stats
    if (fs.existsSync(positionsPath)) {
      const positions = JSON.parse(fs.readFileSync(positionsPath, "utf8"));
      const trades = Object.values(positions) as any[];
      
      const closedTrades = trades.filter(t => t.status === "closed");
      const winningTrades = closedTrades.filter(t => (t.exit_price - t.entry_price) * (t.side === "long" ? 1 : -1) > 0);
      
      realizedPnl = closedTrades.reduce((acc, t) => {
        const pnl = (t.exit_price - t.entry_price) * t.quantity * (t.side === "long" ? 1 : -1);
        return acc + pnl;
      }, 0);

      winRate = closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0;
    }

    // 2. Parse Monitor for Active Signals
    let signals: any[] = [];
    if (fs.existsSync(monitorPath)) {
      const monitor = JSON.parse(fs.readFileSync(monitorPath, "utf8"));
      activeSignalsCount = Object.keys(monitor).length;
      
      // Try to get more detailed signal info from universe_trade_plans if available
      const plansPath = path.join(DATA_DIR, "universe_trade_plans.json");
      if (fs.existsSync(plansPath)) {
        const plans = JSON.parse(fs.readFileSync(plansPath, "utf8"));
        signals = Object.entries(plans).slice(0, 5).map(([sym, plan]: [string, any]) => ({
          sym,
          type: plan.direction.toUpperCase(),
          time: "Active",
          score: plan.confidence || 0.5
        }));
      }
    }

    // TODO: Connect to IBKR for real-time NLV
    // For now, we calculate a baseline + realized P/L
    const baseline = 20000; 
    accountBalance = baseline + realizedPnl;

    return NextResponse.json({
      accountBalance: accountBalance.toLocaleString("en-US", { style: "currency", currency: "USD" }),
      realizedPnl: realizedPnl.toLocaleString("en-US", { style: "currency", currency: "USD", signDisplay: "always" }),
      activeRisk: (activeSignalsCount * 500).toLocaleString("en-US", { style: "currency", currency: "USD" }), 
      winRate: winRate.toFixed(1) + "%",
      activeSignalsCount,
      signals: signals.length > 0 ? signals : [
        { sym: "N/A", type: "NONE", time: "Now", score: 0 }
      ]
    });
  } catch (error) {
    console.error("Dashboard API Error:", error);
    return NextResponse.json({ error: "Failed to fetch metrics" }, { status: 500 });
  }
}
