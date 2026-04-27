import type { Overview } from '../../lib/types'
import { AlertTriangle, Info, CheckCircle, XCircle, ChevronRight } from 'lucide-react'

interface Props { data: Overview }

const ICONS = {
  warning: AlertTriangle,
  info: Info,
  success: CheckCircle,
  error: XCircle,
}
const COLORS = {
  warning: '#ffaa00',
  info: '#00f5ff',
  success: '#00ff9d',
  error: '#ff3355',
}

export default function AlertsCard({ data }: Props) {
  const { alerts } = data

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="card-title">Alerts</div>
        {alerts.length > 0 && (
          <span className="text-[10px] font-mono text-slate-500">{alerts.length} active</span>
        )}
      </div>

      {alerts.length === 0 ? (
        <div className="text-xs text-slate-600 text-center py-4">No active alerts</div>
      ) : (
        <div className="space-y-2">
          {alerts.map((alert, i) => {
            const Icon = ICONS[alert.level]
            const color = COLORS[alert.level]
            return (
              <div
                key={i}
                className="flex items-start gap-2 p-2.5 rounded-lg border"
                style={{ borderColor: `${color}25`, background: `${color}08` }}
              >
                <Icon size={13} style={{ color, marginTop: 1 }} className="shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-semibold" style={{ color }}>{alert.title}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5 leading-tight">{alert.body}</div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      <button className="mt-3 w-full flex items-center justify-center gap-1 text-[10px] text-slate-500 hover:text-cyan transition-colors py-1">
        View All Alerts <ChevronRight size={10} />
      </button>
    </div>
  )
}
