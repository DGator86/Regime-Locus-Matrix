import type { Overview } from '../../lib/types'
import { directionLabel, volLabel, liqLabel, dealerLabel } from '../../lib/utils'

interface Props { data: Overview }

function Pill({ label, color }: { label: string; color: string }) {
  return (
    <span
      className="px-2 py-0.5 rounded text-xs font-bold tracking-wider border"
      style={{ color, borderColor: `${color}44`, background: `${color}15` }}
    >
      {label}
    </span>
  )
}

export default function MarketStateCard({ data }: Props) {
  const { S_D, S_V, S_L, S_G } = data.scores
  const dir = directionLabel(S_D)
  const vol = volLabel(S_V)
  const liq = liqLabel(S_L)
  const dealer = dealerLabel(S_G)

  const dirColor = S_D > 0.15 ? '#00ff9d' : (S_D < -0.15 ? '#ff3355' : '#ffaa00')
  const volColor = S_V > 0.3 ? '#ff3355' : (S_V < 0 ? '#00ff9d' : '#ffaa00')
  const liqColor = S_L > 0.1 ? '#00f5ff' : '#94a3b8'
  const dealerColor = S_G > 0.1 ? '#00ff9d' : (S_G < -0.1 ? '#ff3355' : '#94a3b8')

  return (
    <div className="card p-4">
      <div className="card-title mb-3">Market State</div>
      <div className="flex flex-wrap gap-2 mb-4">
        <Pill label={dir} color={dirColor} />
        <span className="text-slate-600 text-xs self-center">│</span>
        <Pill label={vol} color={volColor} />
        <span className="text-slate-600 text-xs self-center">│</span>
        <Pill label={liq} color={liqColor} />
        <span className="text-slate-600 text-xs self-center">│</span>
        <Pill label={dealer} color={dealerColor} />
      </div>
      <div className="grid grid-cols-2 gap-3">
        {[
          { label: 'Confidence', value: `${data.confidence}%`, color: '#00f5ff' },
          { label: 'HMM State', value: data.hmm_state, color: '#a855f7' },
          { label: 'Markov P(stay)', value: data.markov_prob, color: '#94a3b8' },
          { label: 'Trans. Risk', value: data.transition_risk,
            color: data.transition_risk === 'HIGH' ? '#ff3355' : data.transition_risk === 'MEDIUM' ? '#ffaa00' : '#00ff9d' },
        ].map(({ label, value, color }) => (
          <div key={label}>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">{label}</div>
            <div className="font-mono text-sm font-bold" style={{ color }}>{String(value)}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
