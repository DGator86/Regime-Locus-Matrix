import { useMarketFieldStore } from '../state/useMarketFieldStore';

function Metric({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className='metric-row'>
      <div className='metric-label'>
        <span>{label}</span>
        <span>{value.toFixed(2)}</span>
      </div>
      <div className='bar'>
        <span style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function LeftMetricsPanel() {
  const s = useMarketFieldStore((st) => st.snapshot);
  if (!s) return <section className='panel'>Metrics loading…</section>;
  const regimeClass = s.regime.label.toLowerCase().includes('bull') ? '#38f29f' : s.regime.label.toLowerCase().includes('bear') ? '#ff6284' : '#ffd15f';

  return (
    <section className='panel'>
      <p className='title'>Field Metrics</p>
      <h2 style={{ margin: '0 0 4px' }}>{s.symbol}</h2>
      <div style={{ opacity: 0.85 }}>Price {s.current_price.toFixed(2)}</div>
      <div style={{ color: s.price_change >= 0 ? '#7dffbf' : '#ff889f', marginBottom: 8 }}>
        {s.price_change >= 0 ? '+' : ''}
        {s.price_change.toFixed(2)} ({(s.price_change_pct * 100).toFixed(2)}%)
      </div>
      <div className='decision-card' style={{ borderColor: regimeClass, boxShadow: `0 0 16px ${regimeClass}33` }}>
        Regime: <strong style={{ color: regimeClass }}>{s.regime.label}</strong>
      </div>
      <Metric label='Confidence' value={s.regime.confidence} />
      <Metric label='Force alignment' value={s.field_status.force_alignment} />
      <Metric label='Volatility pressure' value={s.field_status.volatility_pressure} />
      <Metric label='Liquidity pull' value={s.field_status.liquidity_pull} />
    </section>
  );
}
