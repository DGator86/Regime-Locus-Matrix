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
    <section className='panel'>
      <div className='sec'>Price Scale</div>
      <div className='mrow'>
        <span className='mlabel'>CURRENT</span>
        <span className='mval v-white'>{s.current_price.toFixed(2)}</span>
      </div>
      <div className='mrow'>
        <span className='mlabel'>ANCHOR</span>
        <span className='mval v-orange'>{s.anchor_price.toFixed(2)}</span>
      </div>
      <div className='mrow'>
        <span className='mlabel'>CHANGE</span>
        <span className={`mval ${s.price_change >= 0 ? 'v-green' : 'v-red'}`}>
          {s.price_change >= 0 ? '+' : ''}{s.price_change.toFixed(2)}
        </span>
      </div>

      <div className='sec'>Resistance</div>
      {resistances.map((w: any) => (
        <div className='srrow' key={w.id}>
          <span className='sr-res'>{w.price.toFixed(0)}</span>
          <span className='sr-pct'>{(w.strength * 100).toFixed(0)}% STR</span>
        </div>
      ))}

      <div className='sec'>Support</div>
      {supports.map((w: any) => (
        <div className='srrow' key={w.id}>
          <span className='sr-sup'>{w.price.toFixed(0)}</span>
          <span className='sr-pct'>{(w.strength * 100).toFixed(0)}% STR</span>
        </div>
      ))}
    </section>
  );
}
