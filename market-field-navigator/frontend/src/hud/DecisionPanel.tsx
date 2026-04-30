import { useMarketFieldStore } from '../state/useMarketFieldStore';
export default function DecisionPanel() {
  const snapshot = useMarketFieldStore((s) => s.snapshot);
  const selected = useMarketFieldStore((s) => s.selected);
  if (!snapshot) return null;
  return <section className='panel'><div>Current Read</div><h3>{snapshot.decision_summary.headline}</h3><div>Posture: {snapshot.recommended_action_label}</div><div>Force Alignment: {snapshot.field_status.force_alignment.toFixed(2)}</div><ul>{snapshot.decision_summary.details.map((d:string)=><li key={d}>{d}</li>)}</ul><p>{snapshot.decision_summary.risk_warning}</p>{selected && <pre>{JSON.stringify(selected,null,2)}</pre>}</section>;
}
