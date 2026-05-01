import { useEffect, useState } from 'react';
import { fetchSnapshot } from './api/marketFieldApi';
import FieldView from './scene/FieldView';
import HeaderBar from './hud/HeaderBar';
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
      .catch(() => setError('BACKEND OFFLINE — START FASTAPI ON LOCALHOST:8000'))
      .finally(() => setLoading(false));
  }, [setSnapshot]);

  return (
    <div className='app'>
      <HeaderBar />

      <LeftMetricsPanel />

      <div className='scene-wrap'>
        {loading ? (
          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#FF9900', fontFamily: 'Courier New', letterSpacing: '.14em', fontSize: 13 }}>
            INITIALIZING FIELD…
          </div>
        ) : error ? (
          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#FF2255', fontFamily: 'Courier New', letterSpacing: '.1em', fontSize: 12 }}>
            {error}
          </div>
        ) : (
          <FieldView />
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
