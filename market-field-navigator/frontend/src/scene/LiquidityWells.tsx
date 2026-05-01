import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

function Well({ well, onSelect }: { well: any; onSelect: (v: any) => void }) {
  const outerRef = useRef<THREE.Mesh>(null);
  const innerRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (outerRef.current) outerRef.current.rotation.z = t * 0.6;
    if (innerRef.current) innerRef.current.rotation.z = -t * 1.1;
  });

  const color = well.type === 'supportive' ? '#27c7ff' : '#aa44ff';
  const r = 1.0 + well.strength * 1.4;

  return (
    <group
      position={[well.x, -1, 0]}
      onClick={(e) => { e.stopPropagation(); onSelect(well); }}
    >
      {/* outer ring */}
      <mesh ref={outerRef}>
        <torusGeometry args={[r, 0.14, 12, 48]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={3}
          transparent
          opacity={0.75}
        />
      </mesh>
      {/* inner ring */}
      <mesh ref={innerRef}>
        <torusGeometry args={[r * 0.55, 0.08, 10, 36]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={4}
          transparent
          opacity={0.9}
        />
      </mesh>
      {/* floor glow disc */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]}>
        <circleGeometry args={[r * 1.2, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.8}
          transparent
          opacity={0.08}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

export default function LiquidityWells({ wells, onSelect }: { wells: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {wells.map((l: any) => (
        <Well key={l.id} well={l} onSelect={onSelect} />
      ))}
    </>
  );
}
