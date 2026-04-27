import type { Overview } from '../../lib/types'
import { scoreColor } from '../../lib/utils'

interface Props { data: Overview }

function ScoreRow({ label, value }: { label: string; value: number }) {
  const pct = Math.round(((value + 1) / 2) * 100)
  const color = scoreColor(value)
  return (
    <div className="mb-2">
      <div className="flex justify-between items-center mb-1">
        <span className="text-[10px] text-slate-500">{label}</span>
        <span className="font-mono text-xs font-bold" style={{ color }}>{value.toFixed(3)}</span>
      </div>
      <div className="score-bar-track">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color, opacity: 0.8 }}
        />
        {/* Center line */}
        <div className="absolute inset-y-0 left-1/2 w-px bg-white/20" />
      </div>
    </div>
  )
}

export default function CurrentStateCard({ data }: Props) {
  const { state_code, confidence, scores, next_states } = data
  const vol = state_code[0]
  const dir = state_code.slice(1)

  const confRadius = 36
  const confCirc = 2 * Math.PI * confRadius
  const confOffset = confCirc * (1 - confidence / 100)

  return (
    <div className="card p-4 flex flex-col gap-4">
      <div className="card-title">Current State</div>

      <div className="flex items-center gap-4">
        {/* State code badge */}
        <div className="flex flex-col items-center justify-center w-20 h-20 rounded-xl bg-cyan/10 border border-cyan/30 shrink-0">
          <span className="font-mono text-3xl font-black text-cyan leading-none">{state_code}</span>
          <span className="text-[9px] text-slate-500 mt-1">Current</span>
        </div>

        {/* Confidence ring */}
        <div className="relative flex items-center justify-center">
          <svg width={90} height={90}>
            <circle cx={45} cy={45} r={confRadius} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth={6} />
            <circle
              cx={45} cy={45} r={confRadius}
              fill="none" stroke="#00f5ff" strokeWidth={6}
              strokeDasharray={confCirc}
              strokeDashoffset={confOffset}
              strokeLinecap="round"
              transform="rotate(-90 45 45)"
              style={{ filter: 'drop-shadow(0 0 6px rgba(0,245,255,0.5))' }}
            />
          </svg>
          <div className="absolute flex flex-col items-center">
            <span className="font-mono text-lg font-black text-cyan leading-none">{confidence}%</span>
            <span className="text-[9px] text-slate-500">Conf.</span>
          </div>
        </div>
      </div>

      {/* Factor scores */}
      <div>
        <ScoreRow label="S_D  Direction" value={scores.S_D} />
        <ScoreRow label="S_V  Volatility" value={scores.S_V} />
        <ScoreRow label="S_L  Liquidity" value={scores.S_L} />
        <ScoreRow label="S_G  Dealer Flow" value={scores.S_G} />
      </div>

      {/* Next best states */}
      {next_states.length > 0 && (
        <div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Next Best States</div>
          <div className="space-y-1">
            {next_states.slice(0, 4).map(ns => (
              <div key={ns.code} className="flex items-center justify-between">
                <span className="font-mono text-xs text-slate-300 bg-navy-800 px-2 py-0.5 rounded">{ns.code}</span>
                <div className="flex-1 mx-2 h-0.5 bg-navy-700 rounded overflow-hidden">
                  <div
                    className="h-full bg-cyan/50 rounded"
                    style={{ width: `${Math.round(ns.prob * 100)}%` }}
                  />
                </div>
                <span className="font-mono text-xs text-slate-400">{ns.prob.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
