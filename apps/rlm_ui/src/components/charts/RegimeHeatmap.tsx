import { useMemo } from 'react'
import type { RegimeGrid } from '../../lib/types'
import { regimeCellColor } from '../../lib/utils'

interface Props {
  grid: RegimeGrid
  width?: number
  height?: number
}

const VOL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
const DIR_LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

export default function RegimeHeatmap({ grid, width = 480, height = 380 }: Props) {
  const pad = { top: 12, right: 16, bottom: 32, left: 28 }
  const innerW = width - pad.left - pad.right
  const innerH = height - pad.top - pad.bottom
  const cellW = innerW / 10
  const cellH = innerH / 10

  const cellMap = useMemo(() => {
    const m: Record<string, typeof grid.cells[0]> = {}
    for (const c of grid.cells) m[`${c.dir_bin}-${c.vol_bin}`] = c
    return m
  }, [grid.cells])

  return (
    <svg width={width} height={height} className="w-full h-full">
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      <g transform={`translate(${pad.left},${pad.top})`}>
        {/* Grid cells */}
        {Array.from({ length: 10 }, (_, vi) =>
          Array.from({ length: 10 }, (_, di) => {
            const cell = cellMap[`${di + 1}-${vi}`]
            const fill = cell ? regimeCellColor(cell.avg_return) : 'rgba(15,25,50,0.6)'
            const x = di * cellW
            const y = (9 - vi) * cellH
            return (
              <g key={`${di}-${vi}`}>
                <rect
                  x={x + 1} y={y + 1}
                  width={cellW - 2} height={cellH - 2}
                  fill={fill}
                  rx={2}
                  className="regime-cell"
                >
                  <title>{cell ? `${cell.state_code}: avg ${(cell.avg_return * 100).toFixed(3)}% (n=${cell.count})` : 'no data'}</title>
                </rect>
              </g>
            )
          })
        )}

        {/* Trajectory path */}
        {grid.trajectory.length > 1 && (() => {
          const pts = grid.trajectory.map(p => ({
            x: (p.dir_bin - 1) * cellW + cellW / 2,
            y: (9 - p.vol_bin) * cellH + cellH / 2,
          }))
          const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
          return (
            <path
              d={d}
              fill="none"
              stroke="rgba(0,245,255,0.4)"
              strokeWidth={1.5}
              strokeDasharray="4 3"
            />
          )
        })()}

        {/* Trajectory dots */}
        {grid.trajectory.map((p, i) => {
          const x = (p.dir_bin - 1) * cellW + cellW / 2
          const y = (9 - p.vol_bin) * cellH + cellH / 2
          const isLast = i === grid.trajectory.length - 1
          return (
            <circle
              key={i}
              cx={x} cy={y}
              r={isLast ? 5 : 2.5}
              fill={isLast ? '#00f5ff' : 'rgba(0,245,255,0.45)'}
              stroke={isLast ? 'rgba(0,245,255,0.5)' : 'none'}
              strokeWidth={isLast ? 6 : 0}
              filter={isLast ? 'url(#glow)' : undefined}
            />
          )
        })}

        {/* X-axis labels — Direction */}
        {DIR_LABELS.map((l, i) => (
          <text
            key={l}
            x={i * cellW + cellW / 2}
            y={innerH + 16}
            textAnchor="middle"
            fontSize={9}
            fill="#64748b"
          >{l}</text>
        ))}

        {/* Y-axis labels — Volatility */}
        {VOL_LABELS.map((l, i) => (
          <text
            key={l}
            x={-6}
            y={(9 - i) * cellH + cellH / 2 + 3}
            textAnchor="end"
            fontSize={9}
            fill="#64748b"
          >{l}</text>
        ))}

        {/* Axis titles */}
        <text x={innerW / 2} y={innerH + 28} textAnchor="middle" fontSize={9} fill="#475569">
          DIRECTION (Bearish ← → Bullish)
        </text>
        <text
          x={-innerH / 2} y={-18}
          textAnchor="middle" fontSize={9} fill="#475569"
          transform="rotate(-90)"
        >
          VOLATILITY
        </text>
      </g>
    </svg>
  )
}
