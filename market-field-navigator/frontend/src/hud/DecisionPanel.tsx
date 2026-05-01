import { useMarketFieldStore } from '../state/useMarketFieldStore';

const TYPE_COLORS: Record<string, string> = {
  particle: '#ff2255',
  wall: '#00d4ff',
  liquidity: '#27c7ff',
  gamma: '#ffc845',
  regime: '#00ff88',
};

export default function DecisionPanel() {
  const snapshot = useMarketFieldStore((s) => s.snapshot);
  const selected = useMarketFieldStore((s) => s.selected);
  if (!snapshot) return null;

  const support = snapshot.sr_walls
    .filter((w: any) => w.type === 'support')
    .sort((a: any, b: any) => b.price - a.price)[0];
  const resistance = snapshot.sr_walls
    .filter((w: any) => w.type === 'resistance')
    .sort((a: any, b: any) => a.price - b.price)[0];

  return (
    <section className='panel' style={{ flex: '1 1 0', minHeight: 0 }}>
      <div className='section-label'>Current Read</div>
      <div className='decision-headline'>{snapshot.decision_summary.headline}</div>

      <div className='metric-row'>
        <span className='m-label'>Posture</span>
        <span className='m-val val-green'>{snapshot.recommended_action_label}</span>
      </div>
      <div className='metric-row'>
        <span className='m-label'>Force Align</span>
        <span className='m-val val-cyan'>{snapshot.field_status.force_alignment.toFixed(2)}</span>
      </div>
      <div className='metric-row'>
        <span className='m-label'>Support</span>
        <span className='m-val val-cyan'>{support?.price ?? '—'}</span>
      </div>
      <div className='metric-row'>
        <span className='m-label'>Resistance</span>
        <span className='m-val val-red'>{resistance?.price ?? '—'}</span>
      </div>

      <div className='section-label'>Drivers</div>
      {snapshot.decision_summary.details.map((d: string, i: number) => (
        <div className='decision-detail' key={i}>{d}</div>
      ))}

      <div className='risk-warning'>{snapshot.decision_summary.risk_warning}</div>

      {selected && (
        <div className='sel-card'>
          <div className='sel-title' style={{ color: TYPE_COLORS[selected.type] ?? '#3a5a7a' }}>
            {selected.type} selected
          </div>
          {selected.label && (
            <div className='metric-row'>
              <span className='m-label'>Label</span>
              <span className='m-val val-white'>{selected.label}</span>
            </div>
          )}
          {selected.price != null && (
            <div className='metric-row'>
              <span className='m-label'>Price</span>
              <span className='m-val val-cyan'>{Number(selected.price).toFixed(2)}</span>
            </div>
          )}
          {selected.strength != null && (
            <div className='metric-row'>
              <span className='m-label'>Strength</span>
              <span className='m-val val-yellow'>{(Number(selected.strength) * 100).toFixed(0)}%</span>
            </div>
          )}
          {selected.magnitude != null && (
            <div className='metric-row'>
              <span className='m-label'>Magnitude</span>
              <span className='m-val val-yellow'>{Number(selected.magnitude).toFixed(2)}</span>
            </div>
          )}
          {selected.probability != null && (
            <div className='metric-row'>
              <span className='m-label'>Probability</span>
              <span className='m-val val-green'>{(Number(selected.probability) * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
