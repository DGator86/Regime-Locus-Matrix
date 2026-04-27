import { useQuery } from '@tanstack/react-query'
import { CheckCircle, XCircle, Clock } from 'lucide-react'

async function fetchHealth() {
  const res = await fetch('/health')
  return res.json()
}

export default function Diagnostics() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 10000,
  })

  const systems = [
    { name: 'API Server', status: data ? 'ok' : error ? 'error' : 'loading' },
    { name: 'Data Lake (forecast_features)', status: 'ok' },
    { name: 'Trade Log', status: 'ok' },
    { name: 'Universe Plans', status: 'ok' },
    { name: 'IBKR TWS', status: 'warning' },
    { name: 'Massive API', status: 'warning' },
  ]

  return (
    <div className="flex-1 overflow-y-auto p-5 space-y-4">
      <h2 className="text-lg font-semibold text-white mb-0.5">Diagnostics</h2>
      <p className="text-xs text-slate-500">System health and data pipeline status.</p>

      <div className="grid grid-cols-2 gap-4">
        <div className="card p-4">
          <div className="card-title mb-3">System Status</div>
          <div className="space-y-2">
            {systems.map(({ name, status }) => {
              const Icon = status === 'ok' ? CheckCircle : status === 'loading' ? Clock : XCircle
              const color = status === 'ok' ? '#00ff9d' : status === 'loading' ? '#ffaa00' : '#ff3355'
              return (
                <div key={name} className="flex items-center gap-3 py-1.5 border-b border-white/5 last:border-0">
                  <Icon size={13} style={{ color }} />
                  <span className="text-xs text-slate-300 flex-1">{name}</span>
                  <span className="text-[10px] font-mono" style={{ color }}>{status.toUpperCase()}</span>
                </div>
              )
            })}
          </div>
        </div>

        <div className="card p-4">
          <div className="card-title mb-3">API Response</div>
          <pre className="text-xs font-mono text-slate-400 bg-navy-950 rounded-lg p-3 overflow-auto max-h-48">
            {isLoading ? 'Loading...' : error ? `Error: ${error}` : JSON.stringify(data, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  )
}
