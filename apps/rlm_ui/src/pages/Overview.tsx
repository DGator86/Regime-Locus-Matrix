import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import MarketStateCard from '../components/cards/MarketStateCard'
import ActionCard from '../components/cards/ActionCard'
import RiskCard from '../components/cards/RiskCard'
import CurrentStateCard from '../components/cards/CurrentStateCard'
import QuickStatsCard from '../components/cards/QuickStatsCard'
import AlertsCard from '../components/cards/AlertsCard'
import WhyRLMPanel from '../components/panels/WhyRLMPanel'
import TransitionsPanel from '../components/panels/TransitionsPanel'
import RegimeHeatmap from '../components/charts/RegimeHeatmap'
import PriceChart from '../components/charts/PriceChart'
import { RefreshCw } from 'lucide-react'

interface Props { symbol: string }

function Skeleton() {
  return <div className="h-32 rounded-xl bg-navy-800 animate-pulse" />
}

export default function Overview({ symbol }: Props) {
  const { data: overview, isLoading: ovLoading, refetch: refetchOv } = useQuery({
    queryKey: ['overview', symbol],
    queryFn: () => api.overview(symbol),
    refetchInterval: 30000,
    staleTime: 20000,
    retry: 1,
  })

  const { data: forecast, isLoading: fcLoading } = useQuery({
    queryKey: ['forecast', symbol],
    queryFn: () => api.forecast(symbol, 252),
    refetchInterval: 60000,
    staleTime: 45000,
    retry: 1,
  })

  const { data: grid } = useQuery({
    queryKey: ['regime-grid', symbol],
    queryFn: () => api.regimeGrid(symbol),
    refetchInterval: 60000,
    staleTime: 45000,
    retry: 1,
  })

  if (ovLoading) {
    return (
      <div className="p-5 grid grid-cols-3 gap-4">
        {Array.from({ length: 9 }, (_, i) => <Skeleton key={i} />)}
      </div>
    )
  }

  if (!overview) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-slate-500">
        <p className="text-sm">No data available for <span className="text-white font-semibold">{symbol}</span></p>
        <p className="text-xs">Run the pipeline first: <code className="bg-navy-800 px-2 py-1 rounded font-mono">rlm forecast --symbol {symbol}</code></p>
        <button
          onClick={() => refetchOv()}
          className="flex items-center gap-2 px-4 py-2 bg-cyan/10 border border-cyan/30 rounded-lg text-cyan text-sm hover:bg-cyan/20 transition-colors"
        >
          <RefreshCw size={14} /> Retry
        </button>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {/* Row 1: Market State | Action | Risk Status | Current State */}
      <div className="grid grid-cols-4 gap-4">
        <MarketStateCard data={overview} />
        <ActionCard data={overview} />
        <RiskCard data={overview} />
        <CurrentStateCard data={overview} />
      </div>

      {/* Row 2: Regime Heatmap | Price Chart */}
      <div className="grid grid-cols-5 gap-4">
        {/* Regime Locus Matrix heatmap */}
        <div className="col-span-2 card p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="card-title">Regime Locus Matrix</div>
            {grid && (
              <span className="font-mono text-xs text-cyan bg-cyan/10 px-2 py-0.5 rounded border border-cyan/20">
                {grid.current.state_code}
              </span>
            )}
          </div>
          {grid ? (
            <RegimeHeatmap grid={grid} />
          ) : (
            <div className="h-64 bg-navy-800 rounded-lg animate-pulse" />
          )}
        </div>

        {/* Price & State Chart */}
        <div className="col-span-3 card p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="card-title">Price &amp; State Chart</div>
            <div className="flex gap-2 text-xs text-slate-500">
              {['1D', '1W', '1M', '3M', 'YTD'].map(r => (
                <button
                  key={r}
                  className="px-2 py-0.5 rounded hover:bg-white/5 hover:text-white transition-colors"
                >{r}</button>
              ))}
            </div>
          </div>
          {fcLoading ? (
            <div className="h-72 bg-navy-800 rounded-lg animate-pulse" />
          ) : forecast && forecast.length > 0 ? (
            <PriceChart data={forecast} height={300} />
          ) : (
            <div className="h-72 flex items-center justify-center text-slate-600 text-sm">No forecast data</div>
          )}
        </div>
      </div>

      {/* Row 3: Why RLM | Transitions | Quick Stats | Alerts */}
      <div className="grid grid-cols-4 gap-4">
        <div className="col-span-2">
          <WhyRLMPanel data={overview} />
        </div>
        <TransitionsPanel data={overview} />
        <div className="flex flex-col gap-4">
          <QuickStatsCard data={overview} />
          <AlertsCard data={overview} />
        </div>
      </div>
    </div>
  )
}
