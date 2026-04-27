import type { Overview } from '../../lib/types'
import { fmtPct } from '../../lib/utils'

interface Props { data: Overview }

export default function QuickStatsCard({ data }: Props) {
  const { quick_stats } = data
  const stats = [
    { label: 'Avg Return', value: fmtPct(quick_stats.avg_return), color: quick_stats.avg_return >= 0 ? '#00ff9d' : '#ff3355' },
    { label: 'Win Rate', value: `${Math.round(quick_stats.win_rate * 100)}%`, color: '#00f5ff' },
    { label: 'Trades', value: quick_stats.trades, color: '#e2e8f0' },
    { label: 'Expectancy', value: `${quick_stats.expectancy.toFixed(2)}R`, color: quick_stats.expectancy >= 0 ? '#00ff9d' : '#ff3355' },
  ]

  return (
    <div className="card p-4">
      <div className="card-title mb-3">Quick Stats</div>
      <div className="grid grid-cols-2 gap-3 mb-3">
        {stats.map(({ label, value, color }) => (
          <div key={label}>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">{label}</div>
            <div className="font-mono text-sm font-bold" style={{ color }}>{String(value)}</div>
          </div>
        ))}
      </div>
      <div className="border-t border-white/5 pt-3">
        <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Best Strategy</div>
        <div className="text-xs font-semibold text-cyan">{quick_stats.best_strategy}</div>
      </div>
    </div>
  )
}
