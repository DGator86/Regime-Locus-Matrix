import type { Overview } from '../../lib/types'

interface Props { data: Overview }

function RiskRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-white/5 last:border-0">
      <span className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</span>
      <span className="text-xs font-semibold font-mono" style={{ color: color ?? '#e2e8f0' }}>
        {value}
      </span>
    </div>
  )
}

export default function RiskCard({ data }: Props) {
  const { risk } = data
  const drColor = risk.drawdown_risk === 'HIGH' ? '#ff3355' : risk.drawdown_risk === 'MEDIUM' ? '#ffaa00' : '#00ff9d'
  const envColor = risk.environment === 'TRADEABLE' ? '#00ff9d' : '#ffaa00'
  const vpColor = risk.vp_gating === 'PASS' ? '#00ff9d' : '#ffaa00'

  return (
    <div className="card p-4">
      <div className="card-title mb-3">Risk Status</div>
      <RiskRow label="Uncertainty" value={`${risk.uncertainty_pct}%`} color="#ffaa00" />
      <RiskRow label="Vault" value={risk.vault_active ? 'ACTIVE' : 'OFF'} color={risk.vault_active ? '#ff3355' : '#00ff9d'} />
      <RiskRow label="VP Gating" value={risk.vp_gating} color={vpColor} />
      <RiskRow label="Environment" value={risk.environment} color={envColor} />
      <RiskRow label="Drawdown Risk" value={risk.drawdown_risk} color={drColor} />
    </div>
  )
}
