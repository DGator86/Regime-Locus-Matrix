import { useFrame } from '@react-three/fiber';
import { useRef } from 'react';
import type { Mesh } from 'three';

function Well({ l, onSelect }: { l: any; onSelect: (v: any) => void }) {
  const ringA = useRef<Mesh>(null);
  const ringB = useRef<Mesh>(null);
  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (ringA.current) ringA.current.rotation.z = t * 0.9;
    if (ringB.current) ringB.current.rotation.z = -t * 0.6;
  });

  return (
    <group position={[l.x, -2, 0]} onClick={() => onSelect(l)}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <circleGeometry args={[1.35 + l.strength * 1.1, 48]} />
        <meshStandardMaterial color='#4dd9ff' emissive='#1baeff' emissiveIntensity={0.75} transparent opacity={0.26} />
      </mesh>
      <mesh ref={ringA} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[1.3 + l.strength, 0.14, 18, 90]} />
        <meshStandardMaterial color='#6fe4ff' emissive='#3ccfff' emissiveIntensity={1.05} />
      </mesh>
      <mesh ref={ringB} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[1.8 + l.strength * 1.2, 0.06, 14, 60]} />
        <meshStandardMaterial color='#d6f8ff' emissive='#7fe9ff' emissiveIntensity={0.85} />
      </mesh>
    </group>
  );
}

export default function LiquidityWells({ wells, onSelect }: { wells: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {wells.map((l: any) => <Well key={l.id} l={l} onSelect={onSelect} />)}
    </>
  );
}
