import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
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
    <Canvas
      camera={{ position: [0, 38, 65], fov: 52 }}
      gl={{ antialias: true, alpha: false }}
    >
      <color attach="background" args={['#020810']} />
      <fog attach="fog" args={['#020810', 80, 160]} />

      <ambientLight intensity={0.15} />
      <pointLight position={[-40, 30, 20]} color="#00aaff" intensity={6} distance={120} />
      <pointLight position={[40, 30, 20]} color="#ff2255" intensity={4} distance={120} />
      <pointLight position={[0, 10, 0]} color="#ffffff" intensity={1.5} distance={60} />

      <PriceParticle
        x={snapshot.particle.x}
        onClick={() => setSelected({ type: 'particle', ...snapshot.particle })}
      />
      {toggles.regime && (
        <RegimeZones
          zones={snapshot.regime_zones}
          onSelect={(z) => setSelected({ type: 'regime', ...z })}
        />
      )}
      {toggles.sr && (
        <SRWalls
          walls={snapshot.sr_walls}
          onSelect={(w) => setSelected({ type: 'wall', ...w })}
        />
      )}
      {toggles.liquidity && (
        <LiquidityWells
          wells={snapshot.liquidity_wells}
          onSelect={(l) => setSelected({ type: 'liquidity', ...l })}
        />
      )}
      {toggles.gamma && (
        <GammaVectors
          vectors={snapshot.gamma_vectors}
          onSelect={(g) => setSelected({ type: 'gamma', ...g })}
        />
      )}
      {toggles.path && <PricePath points={snapshot.price_path} />}
      {toggles.iv && <IVSurface points={snapshot.iv_surface.points} />}

      <gridHelper args={[140, 48, '#0a2040', '#060e1e']} />

      <EffectComposer>
        <Bloom
          intensity={1.8}
          luminanceThreshold={0.08}
          luminanceSmoothing={0.8}
          mipmapBlur
        />
      </EffectComposer>

      <OrbitControls
        enablePan
        enableZoom
        minDistance={15}
        maxDistance={130}
        maxPolarAngle={Math.PI * 0.48}
      />
    </Canvas>
  );
}
