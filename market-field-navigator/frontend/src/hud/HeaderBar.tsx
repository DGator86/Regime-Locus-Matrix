import { useMarketFieldStore } from '../state/useMarketFieldStore';

export default function HeaderBar() {
  const s = useMarketFieldStore((st) => st.snapshot);

  const regime = s?.regime.label ?? '—';
  const regimeCls = regime.toLowerCase().includes('bull') ? 'badge-bull'
    : regime.toLowerCase().includes('bear') ? 'badge-bear' : 'badge-chop';

  const changePos = (s?.price_change ?? 0) >= 0;

  return (
    <header className='lcars-header'>
      <div className='header-title'>MARKET FIELD NAVIGATOR</div>

      {s ? (
        <>
          <div className='header-sym'>{s.symbol}</div>
          <div className='header-price'>{s.current_price.toFixed(2)}</div>
          <div className={changePos ? 'header-change pos' : 'header-change neg'}>
            {changePos ? '+' : ''}{s.price_change.toFixed(2)}
            &nbsp;({changePos ? '+' : ''}{s.price_change_pct.toFixed(2)}%)
          </div>
          <span className={`header-badge ${regimeCls}`}>{regime.toUpperCase()}</span>
          <div className='header-conf'>
            CONF&nbsp;{(s.regime.confidence * 100).toFixed(0)}%
          </div>
        </>
      ) : (
        <div className='header-sym' style={{ color: '#555' }}>—</div>
      )}

      <div className='header-spacer' />
      <div className='header-status'>
        <span className='status-dot'>
          <animate attributeName="opacity" values="1;0.3;1" dur="1.6s" repeatCount="indefinite" />
        </span>
        LIVE
      </div>
    </header>
  );
}
