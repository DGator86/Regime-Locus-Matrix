import { useMarketFieldStore } from '../state/useMarketFieldStore';
export default function LeftMetricsPanel() {
  const s = useMarketFieldStore((st) => st.snapshot);
  if (!s) return <section className='panel'>Metrics loading…</section>;
  return <section className='panel'><h3>{s.symbol}</h3><div>Price: {s.current_price.toFixed(2)}</div><div>Change: {s.price_change.toFixed(2)} ({s.price_change_pct.toFixed(2)}%)</div><div>Regime: {s.regime.label}</div><div>Confidence: {s.regime.confidence.toFixed(2)}</div><div>Gamma Bias: {s.field_status.gamma_bias}</div><div>Volatility Pressure: {s.field_status.volatility_pressure.toFixed(2)}</div><div>Liquidity Pull: {s.field_status.liquidity_pull.toFixed(2)}</div></section>;
}
