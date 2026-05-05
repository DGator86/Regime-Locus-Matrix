export default function GammaVectors({ vectors, onSelect }: { vectors: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {vectors.map((g: any) => (
        <group key={g.id} position={[g.origin.x, g.origin.y, g.origin.z]} onClick={() => onSelect(g)}>
          <mesh rotation={[0, 0, Math.atan2(g.direction.y, g.direction.x)]} position={[0.8 + g.magnitude * 2.2, 0, 0]}>
            <cylinderGeometry args={[0.08, 0.08, 1.8 + g.magnitude * 4.2, 12]} />
            <meshStandardMaterial color='#ffc957' emissive='#ffbf3b' emissiveIntensity={1.2} />
          </mesh>
          <mesh rotation={[0, 0, Math.atan2(g.direction.y, g.direction.x) - Math.PI / 2]} position={[1.8 + g.magnitude * 3.9, 0, 0]}>
            <coneGeometry args={[0.25, 0.95, 14]} />
            <meshStandardMaterial color='#ffd788' emissive='#ffc74a' emissiveIntensity={1.45} />
          </mesh>
        </group>
      ))}
    </>
  );
}
