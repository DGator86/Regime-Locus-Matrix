import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import PriceChart from '../components/charts/PriceChart'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid,
} from 'recharts'

interface Props { symbol: string }

const BARS_OPTIONS = [
  { label: '3M', bars: 63 },
  { label: '6M', bars: 126 },
  { label: '1Y', bars: 252 },
  { label: '2Y', bars: 504 },
]

export default function Forecast({ symbol }: Props) {
  const [bars, setBars] = useState(252)

  const { data: forecast, isLoading } = useQuery({
    queryKey: ['forecast', symbol, bars],
    queryFn: () => api.forecast(symbol, bars),
    refetchInterval: 60000,
    staleTime: 45000,
  })

  const volData = forecast?.map(b => ({
    date: b.timestamp.slice(0, 10),
    vol: b.realized_vol != null ? +(b.realized_vol * 100).toFixed(2) : null,
    iv: b.surface_atm_forward_iv != null ? +(b.surface_atm_forward_iv * 100).toFixed(2) : null,
  })).filter(d => d.vol != null) ?? []

  const regimeData = forecast?.map(b => ({
    date: b.timestamp.slice(0, 10),
    S_D: b.S_D != null ? +b.S_D.toFixed(4) : null,
    S_V: b.S_V != null ? +b.S_V.toFixed(4) : null,
    S_L: b.S_L != null ? +b.S_L.toFixed(4) : null,
    S_G: b.S_G != null ? +b.S_G.toFixed(4) : null,
  })).filter(d => d.S_D != null) ?? []

  return (
    <div className="flex-1 overflow-y-auto p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white mb-0.5">Forecast — {symbol}</h2>
          <p className="text-xs text-slate-500">Price, regime scores, and volatility over time.</p>
        </div>
        <div className="flex gap-1">
          {BARS_OPTIONS.map(({ label, bars: b }) => (
            <button
              key={b}
              onClick={() => setBars(b)}
              className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${
                bars === b
                  ? 'bg-cyan/20 text-cyan border border-cyan/30'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Price Chart */}
      <div className="card p-4">
        <div className="card-title mb-3">Price &amp; Regime Confidence</div>
        {isLoading ? (
          <div className="h-72 bg-navy-800 rounded-lg animate-pulse" />
        ) : forecast && forecast.length > 0 ? (
          <PriceChart data={forecast} height={300} />
        ) : (
          <div className="h-72 flex items-center justify-center text-slate-600">No data</div>
        )}
      </div>

      {/* Regime Scores over time */}
      <div className="card p-4">
        <div className="card-title mb-3">Regime Scores Over Time</div>
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={regimeData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 9, fill: '#475569' }}
              tickFormatter={d => d.slice(5)}
              interval={Math.floor(regimeData.length / 8)}
            />
            <YAxis domain={[-1, 1]} tick={{ fontSize: 9, fill: '#475569' }} width={28} />
            <Tooltip
              contentStyle={{ background: '#0f1930', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 11 }}
              labelStyle={{ color: '#94a3b8' }}
            />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
            <Area type="monotone" dataKey="S_D" stroke="#00ff9d" fill="#00ff9d22" strokeWidth={1.5} dot={false} name="Direction" />
            <Area type="monotone" dataKey="S_V" stroke="#ff3355" fill="#ff335522" strokeWidth={1.5} dot={false} name="Volatility" />
            <Area type="monotone" dataKey="S_L" stroke="#00f5ff" fill="#00f5ff22" strokeWidth={1.5} dot={false} name="Liquidity" />
            <Area type="monotone" dataKey="S_G" stroke="#a855f7" fill="#a855f722" strokeWidth={1.5} dot={false} name="Dealer Flow" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Realized Vol vs IV */}
      {volData.length > 0 && (
        <div className="card p-4">
          <div className="card-title mb-3">Realized Vol vs. IV Surface ATM</div>
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={volData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 9, fill: '#475569' }}
                tickFormatter={d => d.slice(5)}
                interval={Math.floor(volData.length / 8)}
              />
              <YAxis tick={{ fontSize: 9, fill: '#475569' }} width={32} tickFormatter={v => `${v}%`} />
              <Tooltip
                contentStyle={{ background: '#0f1930', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8, fontSize: 11 }}
                formatter={(v: number) => [`${v.toFixed(1)}%`]}
              />
              <Area type="monotone" dataKey="vol" stroke="#ffaa00" fill="#ffaa0022" strokeWidth={1.5} dot={false} name="Realized Vol (20d)" />
              <Area type="monotone" dataKey="iv" stroke="#a855f7" fill="#a855f722" strokeWidth={1.5} dot={false} name="ATM IV" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
