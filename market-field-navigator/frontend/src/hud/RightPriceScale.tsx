import { useMarketFieldStore } from '../state/useMarketFieldStore';
export default function RightPriceScale() {
  const s = useMarketFieldStore((st) => st.snapshot);
  if (!s) return <section className='panel'>Price scale loading…</section>;
  const supports = s.sr_walls.filter((w:any)=>w.type==='support').map((w:any)=>w.price);
  const resistances = s.sr_walls.filter((w:any)=>w.type==='resistance').map((w:any)=>w.price);
  return <section className='panel'><h4>Price Scale</h4><div>Current: {s.current_price.toFixed(2)}</div><div>Anchor: {s.anchor_price.toFixed(2)}</div><div>Supports: {supports.join(', ')}</div><div>Resistances: {resistances.join(', ')}</div></section>;
}
