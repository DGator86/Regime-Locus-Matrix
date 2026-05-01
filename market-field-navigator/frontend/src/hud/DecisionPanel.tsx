import { useMarketFieldStore } from '../state/useMarketFieldStore';

const TYPE_COLOR: Record<string, string> = {
  particle: '#FF2255', wall: '#00CCFF',
  liquidity: '#AA44FF', gamma: '#FFCC00', regime: '#00CC66',
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
    <section className='panel grow'>
      <div className='sec'>Decision</div>
      <div className='dec-head'>{snapshot.decision_summary.headline}</div>

      <div className='mrow'>
        <span className='mlabel'>POSTURE</span>
        <span className='mval v-green'>{snapshot.recommended_action_label.toUpperCase()}</span>
      </div>
      <div className='mrow'>
        <span className='mlabel'>FORCE ALIGN</span>
        <span className='mval v-cyan'>{snapshot.field_status.force_alignment.toFixed(2)}</span>
      </div>
      <div className='bar-track'><div className='bar-fill bar-c' style={{ width: `${snapshot.field_status.force_alignment * 100}%` }} /></div>
      <div className='mrow'>
        <span className='mlabel'>SUPPORT</span>
        <span className='mval v-cyan'>{support?.price ?? '—'}</span>
      </div>
      <div className='mrow'>
        <span className='mlabel'>RESISTANCE</span>
        <span className='mval v-red'>{resistance?.price ?? '—'}</span>
      </div>

      <div className='sec'>Drivers</div>
      {snapshot.decision_summary.details.map((d: string, i: number) => (
        <div className='dec-detail' key={i}>{d}</div>
      ))}

      <div className='risk-warn'>{snapshot.decision_summary.risk_warning}</div>

      {selected && (
        <div className='sel-card'>
          <div className='sel-head' style={{ color: TYPE_COLOR[selected.type] ?? '#445566' }}>
            {selected.type.toUpperCase()} SELECTED
          </div>
          {selected.label != null && (
            <div className='mrow'><span className='mlabel'>LABEL</span><span className='mval v-white'>{selected.label}</span></div>
          )}
          {selected.price != null && (
            <div className='mrow'><span className='mlabel'>PRICE</span><span className='mval v-cyan'>{Number(selected.price).toFixed(2)}</span></div>
          )}
          {selected.strength != null && (
            <div className='mrow'><span className='mlabel'>STRENGTH</span><span className='mval v-yellow'>{(Number(selected.strength) * 100).toFixed(0)}%</span></div>
          )}
          {selected.magnitude != null && (
            <div className='mrow'><span className='mlabel'>MAG</span><span className='mval v-yellow'>{Number(selected.magnitude).toFixed(2)}</span></div>
          )}
          {selected.probability != null && (
            <div className='mrow'><span className='mlabel'>PROB</span><span className='mval v-green'>{(Number(selected.probability) * 100).toFixed(0)}%</span></div>
          )}
        </div>
      )}
    </section>
  );
}
