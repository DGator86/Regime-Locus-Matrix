import { useEffect, useState } from 'react';
import { fetchSnapshot } from './api/marketFieldApi';
import MarketFieldScene from './scene/MarketFieldScene';
import LeftMetricsPanel from './hud/LeftMetricsPanel';
import RightPriceScale from './hud/RightPriceScale';
import BottomControls from './hud/BottomControls';
import DecisionPanel from './hud/DecisionPanel';
import Legend from './hud/Legend';
import { useMarketFieldStore } from './state/useMarketFieldStore';

export default function App() {
  const setSnapshot = useMarketFieldStore((s) => s.setSnapshot);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSnapshot()
      .then(setSnapshot)
      .catch(() => setError('Backend offline — start FastAPI on localhost:8000 first.'))
      .finally(() => setLoading(false));
  }, [setSnapshot]);

  return (
    <div className='app'>
      <LeftMetricsPanel />

      <div className='scene-wrap'>
        {loading ? (
          <div className='panel' style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#3a5a7a' }}>
            Loading market field…
          </div>
        ) : error ? (
          <div className='panel' style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ff2255' }}>
            {error}
          </div>
        ) : (
          <MarketFieldScene />
        )}
      </div>

      <div className='right-col'>
        <RightPriceScale />
        <Legend />
        <DecisionPanel />
      </div>

      <BottomControls />
    </div>
  );
}
