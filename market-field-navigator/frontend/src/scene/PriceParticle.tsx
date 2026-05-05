import { useFrame } from '@react-three/fiber';
import { useRef } from 'react';
import type { Mesh } from 'three';

export default function PriceParticle({ x, onClick }: { x: number; onClick: () => void }) {
  const orbRef = useRef<Mesh>(null);
  const ringRef = useRef<Mesh>(null);
  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (orbRef.current) orbRef.current.scale.setScalar(1 + Math.sin(t * 2.2) * 0.05);
    if (ringRef.current) ringRef.current.rotation.z = t * 0.8;
  });

  return (
    <group position={[x, 0, 0]} onClick={onClick}>
      <pointLight color='#ff5e9e' intensity={320} distance={16} />
      <mesh ref={orbRef}>
        <sphereGeometry args={[1.1, 42, 42]} />
        <meshStandardMaterial color='#ff89be' emissive='#ff4a8d' emissiveIntensity={1.9} roughness={0.12} metalness={0.2} />
      </mesh>
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[1.9, 0.06, 12, 72]} />
        <meshStandardMaterial color='#ffd2e6' emissive='#ff88b8' emissiveIntensity={1.1} />
      </mesh>
    </group>
  );
}
