import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useMarketFieldStore } from '../state/useMarketFieldStore';

export default function MarketFieldScene() {
  const snapshot = useMarketFieldStore((s) => s.snapshot);
  const toggles = useMarketFieldStore((s) => s.toggles);
  const setSelected = useMarketFieldStore((s) => s.setSelected);
  if (!snapshot) return <div className='panel'>Loading scene...</div>;
  return <Canvas camera={{ position: [30, 25, 40], fov: 55 }}><ambientLight intensity={0.8} /><pointLight position={[10, 20, 10]} />
    <mesh position={[snapshot.particle.x,0,0]} onClick={() => setSelected({type:'particle', ...snapshot.particle})}><sphereGeometry args={[1.2, 32, 32]} /><meshStandardMaterial emissive={'#ff3b7f'} color={'#ff3b7f'} /></mesh>
    {toggles.sr && snapshot.sr_walls.map((w:any)=><mesh key={w.id} position={[w.x,8,0]} onClick={()=>setSelected({type:'wall',...w})}><boxGeometry args={[1, w.height/4, 8]} /><meshStandardMaterial color={w.type==='support'?'#27c7ff':'#ff355e'} transparent opacity={0.35} /></mesh>)}
    {toggles.liquidity && snapshot.liquidity_wells.map((l:any)=><mesh key={l.id} position={[l.x,-2,0]} onClick={()=>setSelected({type:'liquidity',...l})}><torusGeometry args={[1.2,0.3,16,60]} /><meshStandardMaterial color={'#27c7ff'} emissive={'#27c7ff'} /></mesh>)}
    {toggles.gamma && snapshot.gamma_vectors.map((g:any)=><mesh key={g.id} position={[g.origin.x,g.origin.y,g.origin.z]} onClick={()=>setSelected({type:'gamma',...g})}><coneGeometry args={[0.4, 2 + g.magnitude*2, 8]} /><meshStandardMaterial color={'#ffc845'} /></mesh>)}
    <gridHelper args={[120, 40, '#1f4a7a', '#0d223b']} />
    <OrbitControls />
  </Canvas>;
}
