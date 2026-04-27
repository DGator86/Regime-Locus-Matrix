import type { Overview } from '../../lib/types'
import { ArrowRight } from 'lucide-react'

interface Props { data: Overview }

export default function TransitionsPanel({ data }: Props) {
  const { recent_transitions: rt } = data

  const stabilityColor = rt.stability_score > 0.7 ? '#00ff9d' : rt.stability_score > 0.4 ? '#ffaa00' : '#ff3355'
  const typeColor = rt.transition_type.includes('UP') ? '#00ff9d' : rt.transition_type.includes('DOWN') ? '#ff3355' : '#ffaa00'

  return (
    <div className="card p-4">
      <div className="card-title mb-3">Recent Transitions</div>
      <div className="flex items-center gap-3 mb-4">
        {/* Prev state */}
        <div className="text-center">
          <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Previous State</div>
          <div className="font-mono text-2xl font-black text-slate-400">{rt.prev_code}</div>
        </div>

        <ArrowRight size={18} className="text-slate-600 shrink-0" />

        {/* Current state */}
        <div className="text-center">
          <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Current State</div>
          <div className="font-mono text-2xl font-black text-cyan">{rt.curr_code}</div>
        </div>

        <div className="flex-1" />

        {/* Type badge */}
        <div className="text-right">
          <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">Transition Type</div>
          <div className="text-xs font-bold" style={{ color: typeColor }}>{rt.transition_type}</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 border-t border-white/5 pt-3">
        <div>
          <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-0.5">Stability</div>
          <div className="font-mono text-sm font-bold" style={{ color: stabilityColor }}>{rt.stability_score.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-0.5">Bars in State</div>
          <div className="font-mono text-sm font-bold text-white">{rt.bars_in_state}</div>
        </div>
        <div>
          <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-0.5">Early Warning</div>
          <div className="text-xs font-semibold text-slate-400">{rt.early_warning}</div>
        </div>
      </div>
    </div>
  )
}
