import { useMarketFieldStore } from '../state/useMarketFieldStore';
export default function BottomControls() {
  const toggles = useMarketFieldStore((s) => s.toggles);
  const setToggle = useMarketFieldStore((s) => s.setToggle);
  return <section className='panel row'>{Object.entries(toggles).map(([k,v]) => <label key={k}><input type='checkbox' checked={v} onChange={(e)=>setToggle(k,e.target.checked)} /> {k}</label>)}</section>;
}
