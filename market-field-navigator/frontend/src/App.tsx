import { useEffect } from 'react';
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
  useEffect(()=>{ fetchSnapshot().then(setSnapshot); }, [setSnapshot]);
  return <div className='app'><LeftMetricsPanel /><div className='scene-wrap panel'><MarketFieldScene /></div><div><RightPriceScale /><Legend /><DecisionPanel /></div><BottomControls /></div>;
}
