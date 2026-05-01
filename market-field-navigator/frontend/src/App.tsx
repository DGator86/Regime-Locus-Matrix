import { useEffect, useState } from 'react';
import { fetchSnapshot } from './api/marketFieldApi';
import MarketFieldScene from './scene/MarketFieldScene';
import LeftMetricsPanel from './hud/LeftMetricsPanel';
import RightPriceScale from './hud/RightPriceScale';
import BottomControls from './hud/BottomControls';
import DecisionPanel from './hud/DecisionPanel';
import Legend from './hud/Legend';
import { useMarketFieldStore } from './state/useMarketFieldStore';

export default function App(){
  const setSnapshot = useMarketFieldStore((s)=>s.setSnapshot);
  const [loading,setLoading]=useState(true);
  const [error,setError]=useState<string | null>(null);
  useEffect(()=>{ fetchSnapshot().then(setSnapshot).catch(()=>setError('Backend offline. Start FastAPI on localhost:8000 and retry.')).finally(()=>setLoading(false)); }, [setSnapshot]);
  return <div className='app'><LeftMetricsPanel /><div className='scene-wrap panel'>{loading ? <div>Loading market field snapshot...</div> : error ? <div>{error}</div> : <MarketFieldScene />}</div><div><RightPriceScale /><Legend /><DecisionPanel /></div><BottomControls /></div>;
}
