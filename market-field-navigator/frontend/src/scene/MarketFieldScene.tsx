import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
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
    <Canvas camera={{ position: [24, 22, 46], fov: 48 }}>
      <color attach='background' args={['#020810']} />
      <fog attach='fog' args={['#020810', 50, 130]} />
      <ambientLight intensity={0.26} />
      <pointLight position={[-24, 16, 14]} color='#2489ff' intensity={1050} />
      <pointLight position={[22, 18, 16]} color='#ff315e' intensity={850} />
      <pointLight position={[0, -6, 26]} color='#70d6ff' intensity={420} />
      <Stars radius={140} depth={45} count={1200} factor={4} saturation={0} fade speed={0.22} />
      <PriceParticle x={snapshot.particle.x} onClick={() => setSelected({ type: 'particle', ...snapshot.particle })} />
      {toggles.regime && <RegimeZones zones={snapshot.regime_zones} onSelect={(z) => setSelected({ type: 'regime', ...z })} />}
      {toggles.sr && <SRWalls walls={snapshot.sr_walls} onSelect={(w) => setSelected({ type: 'wall', ...w })} />}
      {toggles.liquidity && <LiquidityWells wells={snapshot.liquidity_wells} onSelect={(l) => setSelected({ type: 'liquidity', ...l })} />}
      {toggles.gamma && <GammaVectors vectors={snapshot.gamma_vectors} onSelect={(g) => setSelected({ type: 'gamma', ...g })} />}
      {toggles.path && <PricePath points={snapshot.price_path} />}
      {toggles.iv && <IVSurface points={snapshot.iv_surface.points} />}
      <gridHelper args={[120, 48, '#11406b', '#0c2136']} position={[0, -4.1, 0]} />
      <OrbitControls maxDistance={80} minDistance={26} maxPolarAngle={1.48} />
      <EffectComposer>
        <Bloom intensity={1.15} luminanceThreshold={0.13} luminanceSmoothing={0.22} mipmapBlur />
        <Vignette eskil={false} offset={0.22} darkness={0.6} />
      </EffectComposer>
    </Canvas>
  );
}
