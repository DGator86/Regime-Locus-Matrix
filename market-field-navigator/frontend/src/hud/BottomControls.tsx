import { useMarketFieldStore } from '../state/useMarketFieldStore';

const LABELS: Record<string, string> = {
  regime: 'Regime Zones',
  gamma: 'Gamma Force',
  iv: 'IV Surface',
  liquidity: 'Liquidity',
  sr: 'S/R Walls',
  path: 'Price Path',
};

export default function BottomControls() {
  const toggles = useMarketFieldStore((s) => s.toggles);
  const setToggle = useMarketFieldStore((s) => s.setToggle);

  return (
    <section className='panel row'>
      <span style={{ color: '#3a5a7a', fontSize: 10, letterSpacing: '.08em', textTransform: 'uppercase', marginRight: 4 }}>
        Layers
      </span>
      {Object.entries(toggles).map(([k, v]) => (
        <label key={k} className={`toggle-chip${v ? ' active' : ''}`}>
          <input type='checkbox' checked={v} onChange={(e) => setToggle(k, e.target.checked)} />
          {LABELS[k] ?? k}
        </label>
      ))}
    </section>
  );
}
