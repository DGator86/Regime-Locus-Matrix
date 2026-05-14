import { useMarketFieldStore } from '../state/useMarketFieldStore';

export default function DecisionPanel() {
  const snapshot = useMarketFieldStore((s) => s.snapshot);
  const selected = useMarketFieldStore((s) => s.selected);
  if (!snapshot) return null;

  const support = snapshot.sr_walls.filter((w: any) => w.type === 'support').sort((a: any, b: any) => b.price - a.price)[0];
  const resistance = snapshot.sr_walls.filter((w: any) => w.type === 'resistance').sort((a: any, b: any) => a.price - b.price)[0];

  return (
    <section className='panel'>
      <p className='title'>Decision Guidance</p>
      <h3>{snapshot.decision_summary.headline}</h3>
      <div className='decision-card'>Posture: <strong>{snapshot.recommended_action_label}</strong></div>
      <div>Force Alignment: {snapshot.field_status.force_alignment.toFixed(2)}</div>
      <div>Risk State: {snapshot.field_status.risk_state}</div>
      <div>Nearest Support: {support?.price ?? 'n/a'}</div>
      <div>Nearest Resistance: {resistance?.price ?? 'n/a'}</div>
      <ul>{snapshot.decision_summary.details.map((d: string) => <li key={d}>{d}</li>)}</ul>
      <p>{snapshot.decision_summary.risk_warning}</p>

      {selected && (
        <div className='decision-card'>
          <strong>Selection</strong>
          <div>Type: {selected.type}</div>
          {selected.label && <div>Label: {selected.label}</div>}
          {selected.price && <div>Price: {selected.price}</div>}
          {selected.strength && <div>Strength: {selected.strength}</div>}
        </div>
      )}
    </section>
  );
}
