import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { scoreColor } from '../lib/utils'

interface Props { symbol: string }

const CATEGORY_COLORS: Record<string, string> = {
  DIRECTION: '#00ff9d',
  VOLATILITY: '#ff3355',
  LIQUIDITY: '#00f5ff',
  DEALER_FLOW: '#a855f7',
}

export default function Factors({ symbol }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ['factors', symbol],
    queryFn: () => api.factors(symbol),
    refetchInterval: 30000,
  })

  return (
    <div className="flex-1 overflow-y-auto p-5">
      <h2 className="text-lg font-semibold text-white mb-1">Factor Dashboard</h2>
      <p className="text-xs text-slate-500 mb-5">
        Latest standardized factor values across all regime dimensions for {symbol}.
      </p>

      {isLoading && (
        <div className="grid grid-cols-2 gap-4">
          {Array.from({ length: 4 }, (_, i) => (
            <div key={i} className="h-48 bg-navy-800 rounded-xl animate-pulse" />
          ))}
        </div>
      )}

      {data && (
        <div className="grid grid-cols-2 gap-4">
          {Object.entries(data).map(([category, factors]) => {
            const color = CATEGORY_COLORS[category] ?? '#94a3b8'
            return (
              <div key={category} className="card p-4">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-2 h-2 rounded-full" style={{ background: color }} />
                  <div className="card-title" style={{ color }}>{category}</div>
                </div>
                <div className="space-y-3">
                  {factors.map(({ name, value }) => {
                    const pct = Math.round(((Math.tanh(value * 3) + 1) / 2) * 100)
                    const barColor = scoreColor(Math.tanh(value * 2))
                    return (
                      <div key={name}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-[10px] text-slate-400 font-mono truncate max-w-[65%]">
                            {name.replace('raw_', '').replace(/_/g, ' ')}
                          </span>
                          <span className="font-mono text-xs font-bold" style={{ color: barColor }}>
                            {value > 0 ? '+' : ''}{value.toFixed(4)}
                          </span>
                        </div>
                        <div className="score-bar-track">
                          <div
                            className="absolute inset-y-0 left-0 rounded-full"
                            style={{ width: `${pct}%`, background: barColor, opacity: 0.75 }}
                          />
                          <div className="absolute inset-y-0 left-1/2 w-px bg-white/20" />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
