import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { fmt } from '../lib/utils'

interface Props { symbol: string }

export default function BacktestLab({ symbol }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ['backtest', symbol],
    queryFn: () => api.backtest(symbol),
    staleTime: 300000,
  })

  const { data: wf } = useQuery({
    queryKey: ['backtest', symbol],
    queryFn: () => api.backtest(symbol),
    staleTime: 300000,
  })

  const stats = data?.stats ?? {}

  return (
    <div className="flex-1 overflow-y-auto p-5 space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-white mb-0.5">Backtest Lab — {symbol}</h2>
        <p className="text-xs text-slate-500">Historical equity curves and walk-forward performance.</p>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-5 gap-3">
        {[
          { label: 'Total Return', key: 'total_return', fmt_fn: (v: number) => `${(v * 100).toFixed(1)}%` },
          { label: 'Sharpe Ratio', key: 'sharpe', fmt_fn: (v: number) => v.toFixed(2) },
          { label: 'Max Drawdown', key: 'max_drawdown', fmt_fn: (v: number) => `${(v * 100).toFixed(1)}%` },
          { label: 'Win Rate', key: 'win_rate', fmt_fn: (v: number) => `${(v * 100).toFixed(0)}%` },
          { label: 'CAGR', key: 'cagr', fmt_fn: (v: number) => `${(v * 100).toFixed(1)}%` },
        ].map(({ label, key, fmt_fn }) => {
          const val = stats[key]
          return (
            <div key={key} className="card p-3 text-center">
              <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">{label}</div>
              <div className="font-mono text-base font-bold text-cyan">
                {val != null ? fmt_fn(val) : '—'}
              </div>
            </div>
          )
        })}
      </div>

      {/* Equity curve */}
      <div className="card p-4">
        <div className="card-title mb-3">Equity Curve (Backtest)</div>
        {isLoading ? (
          <div className="h-56 bg-navy-800 rounded-lg animate-pulse" />
        ) : data?.equity && data.equity.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={data.equity} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00f5ff" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#00f5ff" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
              <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#475569' }} tickFormatter={d => String(d).slice(0, 7)} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 9, fill: '#475569' }} width={50} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
              <Tooltip
                contentStyle={{ background: '#0f1930', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 11 }}
                formatter={(v: number) => [`$${v.toLocaleString()}`, 'Equity']}
              />
              <Area type="monotone" dataKey="equity" stroke="#00f5ff" fill="url(#equityGrad)" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-56 flex items-center justify-center text-slate-600">No backtest data — run <code className="mx-1 font-mono bg-navy-800 px-1 rounded">rlm backtest --symbol {symbol}</code></div>
        )}
      </div>

      {/* Walk-forward */}
      {data?.wf_equity && data.wf_equity.length > 0 && (
        <div className="card p-4">
          <div className="card-title mb-3">Walk-Forward Equity</div>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={data.wf_equity} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="wfGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#a855f7" stopOpacity={0.15} />
                  <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
              <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#475569' }} tickFormatter={d => String(d).slice(0, 7)} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 9, fill: '#475569' }} width={50} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
              <Tooltip
                contentStyle={{ background: '#0f1930', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 11 }}
              />
              <Area type="monotone" dataKey="equity" stroke="#a855f7" fill="url(#wfGrad)" strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Recent trades */}
      {data?.trades && data.trades.length > 0 && (
        <div className="card p-4">
          <div className="card-title mb-3">Recent Trades</div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-[9px] text-slate-500 uppercase tracking-wider border-b border-white/5">
                  {Object.keys(data.trades[0]).slice(0, 8).map(k => (
                    <th key={k} className="py-2 px-3 text-left font-semibold">{k.replace(/_/g, ' ')}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.trades.slice(-20).map((row, i) => (
                  <tr key={i} className="border-b border-white/3 hover:bg-white/3 transition-colors">
                    {Object.values(row).slice(0, 8).map((v, j) => (
                      <td key={j} className="py-1.5 px-3 font-mono text-slate-300">
                        {v == null ? '—' : String(v).slice(0, 16)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
