import { useMarketFieldStore } from '../state/useMarketFieldStore';

export default function LeftMetricsPanel() {
  const s = useMarketFieldStore((st) => st.snapshot);
  if (!s) return <section className='panel'><span className='v-dim'>AWAITING FEED…</span></section>;

  const regime = s.regime.label.toLowerCase();
  const regimeBadgeCls = regime.includes('bull') ? 'badge-bull' : regime.includes('bear') ? 'badge-bear' : 'badge-chop';

  return (
    <section className='panel'>
      <div className='sec'>Regime</div>
      <span className={`header-badge ${regimeBadgeCls}`}>{s.regime.label.toUpperCase()}</span>

      <div className='mrow' style={{ marginTop: 6 }}>
        <span className='mlabel'>CONFIDENCE</span>
        <span className='mval v-green'>{(s.regime.confidence * 100).toFixed(0)}%</span>
      </div>
      <div className='bar-track'><div className='bar-fill bar-g' style={{ width: `${s.regime.confidence * 100}%` }} /></div>

      <div className='mrow'>
        <span className='mlabel'>BULL PROB</span>
        <span className='mval v-green'>{(s.regime.bull_probability * 100).toFixed(0)}%</span>
      </div>
      <div className='mrow'>
        <span className='mlabel'>BEAR PROB</span>
        <span className='mval v-red'>{(s.regime.bear_probability * 100).toFixed(0)}%</span>
      </div>
      <div className='mrow'>
        <span className='mlabel'>CHOP PROB</span>
        <span className='mval v-yellow'>{(s.regime.chop_probability * 100).toFixed(0)}%</span>
      </div>

      <div className='sec'>Field Status</div>
      <div className='mrow'>
        <span className='mlabel'>FORCE ALIGN</span>
        <span className='mval v-cyan'>{s.field_status.force_alignment.toFixed(2)}</span>
      </div>
      <div className='bar-track'><div className='bar-fill bar-c' style={{ width: `${s.field_status.force_alignment * 100}%` }} /></div>

      <div className='mrow'>
        <span className='mlabel'>VOL PRESSURE</span>
        <span className='mval v-yellow'>{s.field_status.volatility_pressure.toFixed(2)}</span>
      </div>
      <div className='bar-track'><div className='bar-fill bar-y' style={{ width: `${s.field_status.volatility_pressure * 100}%` }} /></div>

      <div className='mrow'>
        <span className='mlabel'>LIQ PULL</span>
        <span className='mval v-cyan'>{s.field_status.liquidity_pull.toFixed(2)}</span>
      </div>
      <div className='bar-track'><div className='bar-fill bar-c' style={{ width: `${s.field_status.liquidity_pull * 100}%` }} /></div>

      <div className='mrow'>
        <span className='mlabel'>GAMMA BIAS</span>
        <span className='mval v-green'>{s.field_status.gamma_bias.toUpperCase()}</span>
      </div>
      <div className='mrow'>
        <span className='mlabel'>RISK STATE</span>
        <span className='mval v-dim' style={{ fontSize: 9 }}>{s.field_status.risk_state.replace(/_/g, ' ').toUpperCase()}</span>
      </div>
    </section>
  );
}
