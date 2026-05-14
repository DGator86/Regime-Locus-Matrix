import { useMarketFieldStore } from '../state/useMarketFieldStore';
export default function RightPriceScale() {
  const s = useMarketFieldStore((st) => st.snapshot);
  if (!s) return <section className='panel'>Price scale loading…</section>;
  const supports = s.sr_walls.filter((w: any) => w.type === 'support');
  const resistances = s.sr_walls.filter((w: any) => w.type === 'resistance');
  return (
    <section className='panel'>
      <p className='title'>Price Scale</p>
      <div>Current: <strong>{s.current_price.toFixed(2)}</strong></div>
      <div>Anchor: {s.anchor_price.toFixed(2)}</div>
      <div style={{ marginTop: 8, color: '#78e8ff' }}>Supports</div>
      {supports.map((w: any) => (
        <div key={w.id} style={{ fontSize: '0.88rem' }}>
          {w.price.toFixed(2)} <span style={{ opacity: 0.65 }}>({Math.round(w.strength * 100)}%)</span>
        </div>
      ))}
      <div style={{ marginTop: 8, color: '#ff8ca8' }}>Resistance</div>
      {resistances.map((w: any) => (
        <div key={w.id} style={{ fontSize: '0.88rem' }}>
          {w.price.toFixed(2)} <span style={{ opacity: 0.65 }}>({Math.round(w.strength * 100)}%)</span>
        </div>
      ))}
    </section>
  );
}
