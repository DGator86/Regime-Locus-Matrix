import { useMarketFieldStore } from '../state/useMarketFieldStore';
export default function BottomControls() {
  const toggles = useMarketFieldStore((s) => s.toggles);
  const setToggle = useMarketFieldStore((s) => s.setToggle);
  return (
    <section className='panel chip-row'>
      {Object.entries(toggles).map(([k, v]) => (
        <button key={k} className={`toggle-chip ${v ? 'active' : ''}`} onClick={() => setToggle(k, !v)}>
          {k}
        </button>
      ))}
    </section>
  );
}
