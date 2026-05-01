import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function PriceParticle({ x, onClick }: { x: number; onClick: () => void }) {
  const coreRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    if (coreRef.current) {
      const s = 1 + 0.18 * Math.sin(clock.getElapsedTime() * 2.8);
      coreRef.current.scale.setScalar(s);
    }
  });

  return (
    <group position={[x, 0.5, 0]} onClick={(e) => { e.stopPropagation(); onClick(); }}>
      {/* wide outer halo */}
      <mesh>
        <sphereGeometry args={[5, 16, 16]} />
        <meshStandardMaterial
          color="#ff1a55"
          emissive="#ff1a55"
          emissiveIntensity={0.6}
          transparent
          opacity={0.06}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </mesh>
      {/* mid glow ring */}
      <mesh>
        <sphereGeometry args={[2.4, 16, 16]} />
        <meshStandardMaterial
          color="#ff1a55"
          emissive="#ff1a55"
          emissiveIntensity={1}
          transparent
          opacity={0.12}
          depthWrite={false}
        />
      </mesh>
      {/* bright core */}
      <mesh ref={coreRef}>
        <sphereGeometry args={[1.1, 32, 32]} />
        <meshStandardMaterial
          color="#ff3366"
          emissive="#ff3366"
          emissiveIntensity={5}
        />
      </mesh>
      <pointLight color="#ff2255" intensity={10} distance={35} decay={2} />
    </group>
  );
}
