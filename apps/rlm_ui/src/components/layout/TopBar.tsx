import { Bell, Settings, Wifi, WifiOff, ChevronDown } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../../lib/api'

interface Props {
  symbol: string
  onSymbolChange: (s: string) => void
  timeframe: string
  onTimeframeChange: (t: string) => void
  profile: string
  onProfileChange: (p: string) => void
}

const SYMBOLS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'MSFT', 'NVDA']
const TIMEFRAMES = ['1D', '4H', '1H', '30m', '15m']
const PROFILES = ['Default Pro', 'Conservative', 'Aggressive', 'Research']

export default function TopBar({ symbol, onSymbolChange, timeframe, onTimeframeChange, profile, onProfileChange }: Props) {
  const { data: overview } = useQuery({
    queryKey: ['overview', symbol],
    queryFn: () => api.overview(symbol),
    refetchInterval: 30000,
    staleTime: 20000,
  })

  const isLive = !!overview

  return (
    <header className="flex items-center gap-4 px-5 py-2.5 bg-navy-900 border-b border-white/5 shrink-0">
      {/* Symbol */}
      <div className="relative">
        <label className="block text-[9px] text-slate-500 uppercase tracking-widest mb-0.5">Symbol</label>
        <div className="flex items-center gap-1">
          <select
            value={symbol}
            onChange={e => onSymbolChange(e.target.value)}
            className="bg-navy-800 border border-white/10 rounded px-2 py-1 text-sm font-semibold text-white appearance-none pr-6 focus:outline-none focus:border-cyan/40"
          >
            {SYMBOLS.map(s => <option key={s}>{s}</option>)}
          </select>
          <ChevronDown size={12} className="text-slate-500 absolute right-1.5 bottom-1.5 pointer-events-none" />
        </div>
      </div>

      {/* Timeframe */}
      <div className="relative">
        <label className="block text-[9px] text-slate-500 uppercase tracking-widest mb-0.5">Timeframe</label>
        <div className="flex items-center gap-1">
          <select
            value={timeframe}
            onChange={e => onTimeframeChange(e.target.value)}
            className="bg-navy-800 border border-white/10 rounded px-2 py-1 text-sm text-white appearance-none pr-6 focus:outline-none focus:border-cyan/40"
          >
            {TIMEFRAMES.map(t => <option key={t}>{t}</option>)}
          </select>
          <ChevronDown size={12} className="text-slate-500 absolute right-1.5 bottom-1.5 pointer-events-none" />
        </div>
      </div>

      {/* Profile */}
      <div className="relative">
        <label className="block text-[9px] text-slate-500 uppercase tracking-widest mb-0.5">Profile</label>
        <div className="flex items-center gap-1">
          <select
            value={profile}
            onChange={e => onProfileChange(e.target.value)}
            className="bg-navy-800 border border-white/10 rounded px-2 py-1 text-sm text-white appearance-none pr-6 focus:outline-none focus:border-cyan/40"
          >
            {PROFILES.map(p => <option key={p}>{p}</option>)}
          </select>
          <ChevronDown size={12} className="text-slate-500 absolute right-1.5 bottom-1.5 pointer-events-none" />
        </div>
      </div>

      {/* View mode pills */}
      <div className="flex gap-0 rounded-lg overflow-hidden border border-white/10 ml-2">
        {['BASIC', 'PRO', 'RESEARCH'].map(m => (
          <button
            key={m}
            className={`px-3 py-1.5 text-xs font-semibold transition-colors ${
              m === 'PRO'
                ? 'bg-cyan/20 text-cyan border-x border-cyan/30'
                : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
            }`}
          >
            {m}
          </button>
        ))}
      </div>

      <div className="flex-1" />

      {/* Data source indicator */}
      <div className="flex items-center gap-2 text-xs">
        {isLive ? (
          <span className="flex items-center gap-1.5 text-green-trade">
            <Wifi size={12} />
            <span className="font-semibold">IBKR</span>
            <span className="text-slate-500">●</span>
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-amber-trade">
            <WifiOff size={12} />
            <span>DEMO</span>
          </span>
        )}
      </div>

      {/* Alerts */}
      <button className="relative p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors">
        <Bell size={16} />
        {(overview?.alerts?.length ?? 0) > 0 && (
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-trade rounded-full" />
        )}
      </button>

      {/* Settings */}
      <button className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors">
        <Settings size={16} />
      </button>

      {/* Avatar */}
      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan/40 to-navy-600 border border-cyan/30 flex items-center justify-center text-xs font-bold text-cyan">
        D
      </div>
    </header>
  )
}
