"""
EOD PnL Report — calculates and formats daily trading performance.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

def calculate_daily_pnl(root: Path) -> str:
    trade_log = root / "data" / "processed" / "trade_log.csv"
    if not trade_log.is_file():
        return "No trade log found for today."

    today = datetime.now(timezone.utc).date()
    pnl_sum = 0.0
    wins = 0
    losses = 0
    total_trades = 0
    
    try:
        with trade_log.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row.get("timestamp_utc", "")
                if not ts_str:
                    continue
                
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).date()
                except ValueError:
                    continue
                    
                if ts == today:
                    # We check for closed trades or unrealized pnl from the last mark
                    # For EOD, we mostly care about realized PnL of closed positions today
                    # OR the change in unrealized PnL.
                    # Simplified: sum unrealized PnL of all active/closed today.
                    # Actually, let's just report the unrealized PnL from the last poll.
                    pass
            
            # Since the trade log is appended every poll, we can just look at the last poll's summary.
            # But wait, we want "PnL from the day".
            # Better approach: read the whole day's rows and calculate delta.
            
            f.seek(0)
            rows = list(reader)
            if not rows:
                return "Trade log is empty."
            
            # Get all rows for today
            today_rows = []
            for r in rows:
                try:
                    ts = datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00")).date()
                    if ts == today:
                        today_rows.append(r)
                except (ValueError, KeyError):
                    continue
            
            if not today_rows:
                return f"No trading activity recorded for {today}."

            # Group by plan_id to get the last known PnL for each trade active today
            latest_by_plan = {}
            for r in today_rows:
                latest_by_plan[r["plan_id"]] = r
                
            for pid, r in latest_by_plan.items():
                try:
                    pnl = float(r.get("unrealized_pnl", 0.0))
                    pnl_sum += pnl
                    total_trades += 1
                    if pnl > 0:
                        wins += 1
                    elif pnl < 0:
                        losses += 1
                except (ValueError, TypeError):
                    continue

            return (
                f"<b>[EOD Report] {today}</b>\n"
                f"Total PnL: ${pnl_sum:+.2f}\n"
                f"Active/Closed Trades: {total_trades}\n"
                f"Wins: {wins} | Losses: {losses}\n"
                f"Win Rate: {(wins/total_trades*100):.1f}%" if total_trades > 0 else "Win Rate: 0%"
            )
    except Exception as e:
        return f"Error generating PnL report: {e}"
