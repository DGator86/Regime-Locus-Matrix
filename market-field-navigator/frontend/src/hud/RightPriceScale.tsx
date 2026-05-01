import { useMarketFieldStore } from '../state/useMarketFieldStore';

export default function RightPriceScale() {
  const s = useMarketFieldStore((st) => st.snapshot);
  if (!s) return null;

  const supports = s.sr_walls
    .filter((w: any) => w.type === 'support')
    .sort((a: any, b: any) => b.price - a.price);
  const resistances = s.sr_walls
    .filter((w: any) => w.type === 'resistance')
    .sort((a: any, b: any) => a.price - b.price);

  return (
    <section className='panel' style={{ flex: '0 0 auto' }}>
      <div className='section-label'>Price Scale</div>
      <div className='metric-row'>
        <span className='m-label'>Current</span>
        <span className='m-val val-white'>{s.current_price.toFixed(2)}</span>
      </div>
      <div className='metric-row'>
        <span className='m-label'>Anchor</span>
        <span className='m-val val-cyan'>{s.anchor_price.toFixed(2)}</span>
      </div>

      <div className='section-label'>Resistance</div>
      {resistances.map((w: any) => (
        <div className='sr-row' key={w.id}>
          <span className='sr-resistance'>{w.price.toFixed(0)}</span>
          <span style={{ color: '#3a5a7a', fontSize: 10 }}>{(w.strength * 100).toFixed(0)}%</span>
        </div>
      ))}

      <div className='section-label'>Support</div>
      {supports.map((w: any) => (
        <div className='sr-row' key={w.id}>
          <span className='sr-support'>{w.price.toFixed(0)}</span>
          <span style={{ color: '#3a5a7a', fontSize: 10 }}>{(w.strength * 100).toFixed(0)}%</span>
        </div>
      ))}
    </section>
  );
}
