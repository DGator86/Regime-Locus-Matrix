import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useMarketFieldStore } from '../state/useMarketFieldStore';
import PriceParticle from './PriceParticle';
import SRWalls from './SRWalls';
import LiquidityWells from './LiquidityWells';
import GammaVectors from './GammaVectors';
import RegimeZones from './RegimeZones';
import PricePath from './PricePath';
import IVSurface from './IVSurface';

export default function MarketFieldScene() {
  const snapshot = useMarketFieldStore((s) => s.snapshot);
  const toggles = useMarketFieldStore((s) => s.toggles);
  const setSelected = useMarketFieldStore((s) => s.setSelected);
  if (!snapshot) return <div className='panel'>Loading scene...</div>;

  return (
    <Canvas camera={{ position: [30, 25, 40], fov: 55 }}>
      <ambientLight intensity={0.8} />
      <pointLight position={[10, 20, 10]} />
      <PriceParticle x={snapshot.particle.x} onClick={() => setSelected({ type: 'particle', ...snapshot.particle })} />
      {toggles.regime && <RegimeZones zones={snapshot.regime_zones} onSelect={(z) => setSelected({ type: 'regime', ...z })} />}
      {toggles.sr && <SRWalls walls={snapshot.sr_walls} onSelect={(w) => setSelected({ type: 'wall', ...w })} />}
      {toggles.liquidity && <LiquidityWells wells={snapshot.liquidity_wells} onSelect={(l) => setSelected({ type: 'liquidity', ...l })} />}
      {toggles.gamma && <GammaVectors vectors={snapshot.gamma_vectors} onSelect={(g) => setSelected({ type: 'gamma', ...g })} />}
      {toggles.path && <PricePath points={snapshot.price_path} />}
      {toggles.iv && <IVSurface points={snapshot.iv_surface.points} />}
      <gridHelper args={[120, 40, '#1f4a7a', '#0d223b']} />
      <OrbitControls />
    </Canvas>
  );
}
