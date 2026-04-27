import type { Overview } from '../../lib/types'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface Props { data: Overview }

export default function ActionCard({ data }: Props) {
  const { action } = data
  const isEnter = action.type === 'ENTER'
  const isExit = action.type === 'EXIT'
  const color = isEnter ? '#00ff9d' : (isExit ? '#ff3355' : '#ffaa00')
  const Icon = isEnter ? TrendingUp : (isExit ? TrendingDown : Minus)

  return (
    <div className="card p-4">
      <div className="card-title mb-3">Recommended Action</div>
      <div className="flex items-start gap-3 mb-3">
        <div
          className="flex items-center justify-center w-12 h-12 rounded-xl shrink-0"
          style={{ background: `${color}20`, border: `1px solid ${color}40` }}
        >
          <Icon size={22} style={{ color }} />
        </div>
        <div>
          <div className="font-mono text-2xl font-black leading-none mb-0.5" style={{ color }}>
            {action.type}
          </div>
          <div className="text-xs text-slate-400 font-medium">{action.strategy || '—'}</div>
        </div>
      </div>

      {action.size_pct > 0 && (
        <div className="flex items-center gap-2 mb-3">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">Size</span>
          <span className="font-mono text-sm font-bold text-white">{action.size_pct}%</span>
          <span className="text-[10px] text-slate-500">of capital</span>
        </div>
      )}

      <p className="text-[11px] text-slate-400 leading-relaxed border-t border-white/5 pt-3">
        {action.rationale}
      </p>
    </div>
  )
}
