import type { Overview } from '../../lib/types'
import { TrendingUp, TrendingDown, GitMerge } from 'lucide-react'

interface Props { data: Overview }

function BulletList({ items, color, icon: Icon }: {
  items: string[]
  color: string
  icon: React.ElementType
}) {
  return (
    <ul className="space-y-1">
      {items.map((item, i) => (
        <li key={i} className="flex items-center gap-2 text-[11px] text-slate-300">
          <Icon size={9} style={{ color }} className="shrink-0" />
          {item}
        </li>
      ))}
    </ul>
  )
}

export default function WhyRLMPanel({ data }: Props) {
  const { why_rlm } = data
  return (
    <div className="card p-4">
      <div className="card-title mb-3">Why RLM Thinks This</div>
      <div className="grid grid-cols-3 gap-4">
        <div>
          <div className="text-[9px] font-bold uppercase tracking-widest text-green-trade mb-2">Top Drivers</div>
          <BulletList items={why_rlm.top_drivers} color="#00ff9d" icon={TrendingUp} />
        </div>
        <div>
          <div className="text-[9px] font-bold uppercase tracking-widest text-red-trade mb-2">Top Penalties</div>
          <BulletList items={why_rlm.top_penalties} color="#ff3355" icon={TrendingDown} />
        </div>
        <div>
          <div className="text-[9px] font-bold uppercase tracking-widest text-amber-trade mb-2">Key Confluences</div>
          <BulletList items={why_rlm.key_confluences} color="#ffaa00" icon={GitMerge} />
        </div>
      </div>
    </div>
  )
}
