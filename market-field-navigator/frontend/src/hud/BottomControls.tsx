import { useMarketFieldStore } from '../state/useMarketFieldStore';

const LABELS: Record<string, string> = {
  regime: 'Regime', gamma: 'Gamma', iv: 'IV Surface',
  liquidity: 'Liquidity', sr: 'S/R Walls', path: 'Price Path',
};

export default function BottomControls() {
  const toggles = useMarketFieldStore((s) => s.toggles);
  const setToggle = useMarketFieldStore((s) => s.setToggle);

  return (
    <section className='row'>
      <span className='row-label'>Active Layers</span>
      {Object.entries(toggles).map(([k, v]) => (
        <label key={k} className={`chip${v ? ' on' : ''}`}>
          <input type='checkbox' checked={v} onChange={(e) => setToggle(k, e.target.checked)} />
          {LABELS[k] ?? k}
        </label>
      ))}
    </section>
  );
}
