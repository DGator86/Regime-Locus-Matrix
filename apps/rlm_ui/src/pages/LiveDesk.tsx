import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { fmtPct } from '../lib/utils'
import { RefreshCw } from 'lucide-react'

export default function LiveDesk() {
  const { data: trades, isLoading, refetch } = useQuery({
    queryKey: ['trades'],
    queryFn: () => api.trades(50),
    refetchInterval: 15000,
  })

  const { data: challenge } = useQuery({
    queryKey: ['challenge'],
    queryFn: api.challenge,
    refetchInterval: 30000,
  })

  const openTrades = trades?.filter(t => !t.closed) ?? []
  const closedTrades = trades?.filter(t => t.closed) ?? []

  return (
    <div className="flex-1 overflow-y-auto p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white mb-0.5">Live Desk</h2>
          <p className="text-xs text-slate-500">Active positions and recent trade activity.</p>
        </div>
        <button onClick={() => refetch()} className="flex items-center gap-1.5 px-3 py-1.5 bg-navy-800 border border-white/10 rounded-lg text-xs text-slate-400 hover:text-white transition-colors">
          <RefreshCw size={12} /> Refresh
        </button>
      </div>

      {/* Challenge state */}
      {challenge && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Balance', value: `$${(challenge.balance ?? 0).toLocaleString()}`, color: '#00f5ff' },
            { label: 'Total Return', value: `${((challenge.total_return_pct ?? 0) * 100).toFixed(1)}%`, color: (challenge.total_return_pct ?? 0) >= 0 ? '#00ff9d' : '#ff3355' },
            { label: 'Win Rate', value: `${Math.round((challenge.win_rate ?? 0) * 100)}%`, color: '#00ff9d' },
            { label: 'Trades', value: challenge.trades ?? 0, color: '#e2e8f0' },
          ].map(({ label, value, color }) => (
            <div key={label} className="card p-3 text-center">
              <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">{label}</div>
              <div className="font-mono text-base font-bold" style={{ color }}>{String(value)}</div>
            </div>
          ))}
        </div>
      )}

      {/* Open positions */}
      <div className="card p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="card-title">Open Positions</div>
          <span className="text-xs font-mono text-slate-500">{openTrades.length} active</span>
        </div>
        {isLoading ? (
          <div className="h-24 bg-navy-800 rounded-lg animate-pulse" />
        ) : openTrades.length === 0 ? (
          <div className="text-xs text-slate-600 py-4 text-center">No open positions</div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-[9px] text-slate-500 uppercase tracking-wider border-b border-white/5">
                <th className="py-2 px-3 text-left">Symbol</th>
                <th className="py-2 px-3 text-left">Strategy</th>
                <th className="py-2 px-3 text-right">Entry</th>
                <th className="py-2 px-3 text-right">Mark</th>
                <th className="py-2 px-3 text-right">P&L %</th>
                <th className="py-2 px-3 text-right">DTE</th>
                <th className="py-2 px-3 text-left">Signal</th>
              </tr>
            </thead>
            <tbody>
              {openTrades.map((t, i) => {
                const pnl = Number(t.unrealized_pnl_pct ?? t.unrealized_pnl ?? 0)
                const color = pnl >= 0 ? '#00ff9d' : '#ff3355'
                return (
                  <tr key={i} className="border-b border-white/3 hover:bg-white/3 transition-colors">
                    <td className="py-2 px-3 font-semibold text-white">{String(t.symbol ?? '—')}</td>
                    <td className="py-2 px-3 text-slate-400">{String(t.strategy ?? '—')}</td>
                    <td className="py-2 px-3 text-right font-mono">{String(t.entry_debit ?? '—')}</td>
                    <td className="py-2 px-3 text-right font-mono">{String(t.current_mark ?? '—')}</td>
                    <td className="py-2 px-3 text-right font-mono font-bold" style={{ color }}>
                      {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-slate-400">{String(t.dte ?? '—')}</td>
                    <td className="py-2 px-3 text-slate-400">{String(t.signal ?? '—')}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Recent closed trades */}
      <div className="card p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="card-title">Recent Closed Trades</div>
          <span className="text-xs font-mono text-slate-500">{closedTrades.length} closed</span>
        </div>
        {closedTrades.length === 0 ? (
          <div className="text-xs text-slate-600 py-4 text-center">No closed trades</div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-[9px] text-slate-500 uppercase tracking-wider border-b border-white/5">
                <th className="py-2 px-3 text-left">Symbol</th>
                <th className="py-2 px-3 text-left">Strategy</th>
                <th className="py-2 px-3 text-right">P&L %</th>
                <th className="py-2 px-3 text-left">Signal</th>
              </tr>
            </thead>
            <tbody>
              {closedTrades.slice(-15).map((t, i) => {
                const pnl = Number(t.unrealized_pnl_pct ?? 0)
                return (
                  <tr key={i} className="border-b border-white/3 hover:bg-white/3 transition-colors opacity-70">
                    <td className="py-1.5 px-3 text-slate-300">{String(t.symbol ?? '—')}</td>
                    <td className="py-1.5 px-3 text-slate-500">{String(t.strategy ?? '—')}</td>
                    <td className={`py-1.5 px-3 text-right font-mono font-bold ${pnl >= 0 ? 'text-green-trade' : 'text-red-trade'}`}>
                      {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}%
                    </td>
                    <td className="py-1.5 px-3 text-slate-500">{String(t.signal ?? '—')}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
