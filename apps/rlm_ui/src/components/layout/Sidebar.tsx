import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Map, BarChart2, TrendingUp, Activity,
  Link2, FlaskConical, Radio, Stethoscope, Settings, Zap,
} from 'lucide-react'

const NAV = [
  { to: '/', icon: LayoutDashboard, label: 'Overview' },
  { to: '/state-map', icon: Map, label: 'State Map' },
  { to: '/factors', icon: BarChart2, label: 'Factors' },
  { to: '/forecast', icon: TrendingUp, label: 'Forecast' },
  { to: '/roee', icon: Activity, label: 'ROEE' },
  { to: '/chain', icon: Link2, label: 'Chain' },
  { to: '/backtest', icon: FlaskConical, label: 'Backtest Lab' },
  { to: '/live-desk', icon: Radio, label: 'Live Desk' },
  { to: '/diagnostics', icon: Stethoscope, label: 'Diagnostics' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

export default function Sidebar() {
  return (
    <aside className="flex flex-col w-52 shrink-0 bg-navy-900 border-r border-white/5 h-full">
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-4 border-b border-white/5">
        <div className="w-8 h-8 rounded-full bg-cyan/20 border border-cyan/40 flex items-center justify-center pulse-glow">
          <Zap size={14} className="text-cyan" />
        </div>
        <div>
          <div className="text-xs font-bold tracking-widest text-white uppercase">RLM</div>
          <div className="text-[10px] text-slate-500 leading-tight">Regime Locus Matrix</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              isActive
                ? 'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-cyan bg-cyan/10 border border-cyan/20'
                : 'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-colors'
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-white/5 text-[10px] text-slate-600">
        RLM v2.1 · Build 2026
      </div>
    </aside>
  )
}
