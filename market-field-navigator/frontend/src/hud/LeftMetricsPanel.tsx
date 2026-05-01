import { useMarketFieldStore } from '../state/useMarketFieldStore';

export default function LeftMetricsPanel() {
  const s = useMarketFieldStore((st) => st.snapshot);
  if (!s) return <section className='panel'><span style={{ color: '#3a5a7a' }}>Awaiting feed…</span></section>;

  const changePos = s.price_change >= 0;
  const regime = s.regime.label.toLowerCase().includes('bull') ? 'bull'
    : s.regime.label.toLowerCase().includes('bear') ? 'bear' : 'chop';

  return (
    <section className='panel'>
      <div className='symbol'>{s.symbol}</div>
      <div className='price-big'>{s.current_price.toFixed(2)}</div>
      <div className={changePos ? 'change-positive' : 'change-negative'}>
        {changePos ? '+' : ''}{s.price_change.toFixed(2)} ({changePos ? '+' : ''}{s.price_change_pct.toFixed(2)}%)
      </div>

      <div className='section-label'>Regime</div>
      <span className={`regime-badge regime-${regime}`}>
        {s.regime.label}
      </span>
      <div className='bar-track' style={{ marginTop: 6 }}>
        <div className='bar-fill bar-green' style={{ width: `${s.regime.confidence * 100}%` }} />
      </div>
      <div className='metric-row'>
        <span className='m-label'>Confidence</span>
        <span className='m-val val-green'>{(s.regime.confidence * 100).toFixed(0)}%</span>
      </div>
      <div className='metric-row'>
        <span className='m-label'>Bull prob</span>
        <span className='m-val val-green'>{(s.regime.bull_probability * 100).toFixed(0)}%</span>
      </div>
      <div className='metric-row'>
        <span className='m-label'>Bear prob</span>
        <span className='m-val val-red'>{(s.regime.bear_probability * 100).toFixed(0)}%</span>
      </div>

      <div className='section-label'>Field Status</div>
      <div className='metric-row'>
        <span className='m-label'>Force Align</span>
        <span className='m-val val-cyan'>{s.field_status.force_alignment.toFixed(2)}</span>
      </div>
      <div className='bar-track'>
        <div className='bar-fill bar-cyan' style={{ width: `${s.field_status.force_alignment * 100}%` }} />
      </div>
      <div className='metric-row'>
        <span className='m-label'>Vol Pressure</span>
        <span className='m-val val-yellow'>{s.field_status.volatility_pressure.toFixed(2)}</span>
      </div>
      <div className='bar-track'>
        <div className='bar-fill bar-yellow' style={{ width: `${s.field_status.volatility_pressure * 100}%` }} />
      </div>
      <div className='metric-row'>
        <span className='m-label'>Liquidity Pull</span>
        <span className='m-val val-cyan'>{s.field_status.liquidity_pull.toFixed(2)}</span>
      </div>
      <div className='bar-track'>
        <div className='bar-fill bar-cyan' style={{ width: `${s.field_status.liquidity_pull * 100}%` }} />
      </div>
      <div className='metric-row'>
        <span className='m-label'>Gamma Bias</span>
        <span className='m-val val-green'>{s.field_status.gamma_bias}</span>
      </div>
      <div className='metric-row'>
        <span className='m-label'>Risk State</span>
        <span className='m-val val-white' style={{ fontSize: 10 }}>{s.field_status.risk_state.replace(/_/g, ' ')}</span>
      </div>
    </section>
  );
}
