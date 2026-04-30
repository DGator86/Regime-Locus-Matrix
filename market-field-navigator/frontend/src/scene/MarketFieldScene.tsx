import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useMarketFieldStore } from '../state/useMarketFieldStore';

export default function MarketFieldScene() {
  const snapshot = useMarketFieldStore((s) => s.snapshot);
  const setSelected = useMarketFieldStore((s) => s.setSelected);
  if (!snapshot) return <div className='panel'>Loading scene...</div>;
  return <Canvas camera={{ position: [30, 25, 40], fov: 55 }}><ambientLight intensity={0.8} /><pointLight position={[10, 20, 10]} />
    <mesh position={[0,0,0]} onClick={() => setSelected({type:'particle', data:snapshot.particle})}><sphereGeometry args={[1.2, 32, 32]} /><meshStandardMaterial emissive={'#ff3b7f'} color={'#ff3b7f'} /></mesh>
    <gridHelper args={[120, 40, '#1f4a7a', '#0d223b']} />
    <OrbitControls />
  </Canvas>;
}
