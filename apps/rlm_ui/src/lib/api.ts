import type { Overview, ForecastBar, RegimeGrid, TickerItem, UniverseRow } from './types'

const BASE = '/api/v1'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export const api = {
  overview: (symbol: string) => get<Overview>(`/overview/${symbol}`),
  forecast: (symbol: string, bars = 252) => get<ForecastBar[]>(`/forecast/${symbol}?bars=${bars}`),
  regimeGrid: (symbol: string) => get<RegimeGrid>(`/regime-grid/${symbol}`),
  factors: (symbol: string) => get<Record<string, { name: string; value: number }[]>>(`/factors/${symbol}`),
  backtest: (symbol: string) => get<{
    equity: { date: string; equity: number }[];
    wf_equity: { date: string; equity: number }[];
    trades: Record<string, unknown>[];
    stats: Record<string, number>;
  }>(`/backtest/${symbol}`),
  challenge: () => get<Record<string, number>>('/challenge'),
  universe: () => get<UniverseRow[]>('/universe'),
  trades: (limit = 50) => get<Record<string, unknown>[]>(`/trades?limit=${limit}`),
  ticker: () => get<TickerItem[]>('/ticker'),
}
