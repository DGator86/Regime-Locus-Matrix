import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import RegimeHeatmap from '../components/charts/RegimeHeatmap'

interface Props { symbol: string }

export default function StateMap({ symbol }: Props) {
  const { data: grid, isLoading } = useQuery({
    queryKey: ['regime-grid', symbol],
    queryFn: () => api.regimeGrid(symbol),
    refetchInterval: 60000,
  })

  return (
    <div className="flex-1 overflow-y-auto p-5">
      <h2 className="text-lg font-semibold text-white mb-1">Regime State Map</h2>
      <p className="text-xs text-slate-500 mb-5">
        10×10 grid of (Direction × Volatility) regime cells. Color = avg 1-day forward return. Dot trail = recent trajectory.
      </p>

      {isLoading && <div className="h-96 bg-navy-800 rounded-xl animate-pulse" />}

      {grid && (
        <div className="grid grid-cols-3 gap-5">
          <div className="col-span-2 card p-5">
            <div className="card-title mb-4">Full State Map — {symbol}</div>
            <RegimeHeatmap grid={grid} width={640} height={520} />
          </div>

          <div className="space-y-4">
            {/* Current state info */}
            <div className="card p-4">
              <div className="card-title mb-3">Current Position</div>
              <div className="text-center py-4">
                <div className="font-mono text-5xl font-black text-cyan">{grid.current.state_code}</div>
                <div className="text-xs text-slate-500 mt-2">
                  Dir {grid.current.dir_bin} · Vol {String.fromCharCode(64 + grid.current.vol_bin + 1)}
                </div>
              </div>
            </div>

            {/* Recent trajectory */}
            <div className="card p-4">
              <div className="card-title mb-3">State Trajectory (last 25)</div>
              <div className="flex flex-wrap gap-1.5">
                {grid.trajectory.map((pt, i) => {
                  const isLast = i === grid.trajectory.length - 1
                  return (
                    <span
                      key={i}
                      className={`font-mono text-xs px-1.5 py-0.5 rounded ${
                        isLast
                          ? 'bg-cyan/20 text-cyan border border-cyan/40'
                          : 'bg-navy-800 text-slate-400'
                      }`}
                    >
                      {pt.state_code}
                    </span>
                  )
                })}
              </div>
            </div>

            {/* Cell stats */}
            <div className="card p-4">
              <div className="card-title mb-3">Cell Statistics (top 5 by count)</div>
              <div className="space-y-2">
                {grid.cells
                  .filter(c => c.count > 0)
                  .sort((a, b) => b.count - a.count)
                  .slice(0, 5)
                  .map(cell => (
                    <div key={cell.state_code} className="flex items-center gap-2">
                      <span className="font-mono text-xs bg-navy-800 px-1.5 py-0.5 rounded text-slate-300 w-8 text-center">
                        {cell.state_code}
                      </span>
                      <div className="flex-1 h-1.5 bg-navy-700 rounded overflow-hidden">
                        <div
                          className="h-full rounded"
                          style={{
                            width: `${Math.min(100, cell.count / 5)}%`,
                            background: cell.avg_return > 0 ? '#00ff9d' : '#ff3355',
                          }}
                        />
                      </div>
                      <span className="text-[10px] font-mono text-slate-500 w-16 text-right">
                        {(cell.avg_return * 100).toFixed(3)}%
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
