import { useQuery } from '@tanstack/react-query'
import { api } from '../../lib/api'
import type { TickerItem } from '../../lib/types'
import { TrendingUp, TrendingDown } from 'lucide-react'

const STATIC_TICKERS: TickerItem[] = [
  { symbol: 'SPY', price: null, change: null, change_pct: null },
  { symbol: 'QQQ', price: null, change: null, change_pct: null },
  { symbol: 'IWM', price: null, change: null, change_pct: null },
  { symbol: 'VIX', price: null, change: null, change_pct: null },
  { symbol: '10Y YIELD', price: null, change: null, change_pct: null },
  { symbol: 'DXY', price: null, change: null, change_pct: null },
]

function TickerChip({ item }: { item: TickerItem }) {
  const up = (item.change_pct ?? 0) >= 0
  const color = item.change === null ? 'text-slate-500' : (up ? 'text-green-trade' : 'text-red-trade')
  return (
    <span className="flex items-center gap-2 px-4 text-xs shrink-0">
      <span className="text-slate-400 font-semibold">{item.symbol}</span>
      {item.price != null && (
        <>
          <span className="text-white font-mono">{item.price.toFixed(2)}</span>
          <span className={`flex items-center gap-0.5 ${color} font-mono`}>
            {item.change != null && (
              <>
                {up ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
                {up ? '+' : ''}{item.change?.toFixed(2)} ({up ? '+' : ''}{item.change_pct?.toFixed(2)}%)
              </>
            )}
          </span>
        </>
      )}
      {item.price == null && <span className="text-slate-600">—</span>}
      <span className="text-white/10">│</span>
    </span>
  )
}

export default function TickerBar() {
  const { data } = useQuery({
    queryKey: ['ticker'],
    queryFn: api.ticker,
    refetchInterval: 15000,
    staleTime: 10000,
  })

  const tickers = data ?? STATIC_TICKERS
  const doubled = [...tickers, ...tickers]

  const now = new Date()
  const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  return (
    <div className="flex items-center h-7 bg-navy-950 border-t border-white/5 overflow-hidden shrink-0">
      <div className="px-3 text-[10px] font-semibold text-slate-600 shrink-0 border-r border-white/5">
        MARKET OVERVIEW
      </div>
      <div className="flex-1 overflow-hidden relative">
        <div className="ticker-inner flex whitespace-nowrap">
          {doubled.map((item, i) => <TickerChip key={`${item.symbol}-${i}`} item={item} />)}
        </div>
      </div>
      <div className="px-3 text-[10px] font-mono text-slate-600 shrink-0 border-l border-white/5">
        {timeStr} ET
      </div>
    </div>
  )
}
