import * as THREE from 'three';

const UP = new THREE.Vector3(0, 1, 0);

function Arrow({ vector, onSelect }: { vector: any; onSelect: (v: any) => void }) {
  const dir = new THREE.Vector3(vector.direction.x, vector.direction.y, vector.direction.z).normalize();
  const q = new THREE.Quaternion().setFromUnitVectors(UP, dir);
  const len = 2.5 + vector.magnitude * 5;
  const color = '#ffc845';

  return (
    <group
      position={[vector.origin.x, vector.origin.y, vector.origin.z]}
      quaternion={q}
      onClick={(e) => { e.stopPropagation(); onSelect(vector); }}
    >
      {/* shaft */}
      <mesh position={[0, len * 0.42, 0]}>
        <cylinderGeometry args={[0.1, 0.18, len * 0.85, 7]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={3}
        />
      </mesh>
      {/* arrowhead */}
      <mesh position={[0, len, 0]}>
        <coneGeometry args={[0.45, 1.4, 7]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={4}
        />
      </mesh>
    </group>
  );
}

export default function GammaVectors({ vectors, onSelect }: { vectors: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {vectors.map((g: any) => (
        <Arrow key={g.id} vector={g} onSelect={onSelect} />
      ))}
    </>
  );
}
