import * as THREE from 'three';

const WALL_DEPTH = 12;

function Wall({ wall, onSelect }: { wall: any; onSelect: (w: any) => void }) {
  const isSup = wall.type === 'support';
  const color = isSup ? '#00d4ff' : '#ff2255';
  const h = Math.max(wall.height * 0.7, 8);

  return (
    <group
      position={[wall.x, h / 2, 0]}
      onClick={(e) => { e.stopPropagation(); onSelect(wall); }}
    >
      {/* translucent slab */}
      <mesh>
        <boxGeometry args={[0.6, h, WALL_DEPTH]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={1.5}
          transparent
          opacity={0.22 + wall.strength * 0.18}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      {/* bright edge strip */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[0.12, h, WALL_DEPTH + 0.2]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={4}
          transparent
          opacity={0.9}
        />
      </mesh>
    </group>
  );
}

export default function SRWalls({ walls, onSelect }: { walls: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {walls.map((w: any) => (
        <Wall key={w.id} wall={w} onSelect={onSelect} />
      ))}
    </>
  );
}
