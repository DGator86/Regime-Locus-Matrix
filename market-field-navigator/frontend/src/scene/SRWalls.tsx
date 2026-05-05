export default function SRWalls({ walls, onSelect }: { walls: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {walls.map((w: any) => (
        <group key={w.id} position={[w.x, 8, 0]} onClick={() => onSelect(w)}>
          <mesh>
            <boxGeometry args={[1.4, w.height / 3.5, 11]} />
            <meshStandardMaterial
              color={w.type === 'support' ? '#28d2ff' : '#ff446f'}
              transparent
              opacity={0.22}
              emissive={w.type === 'support' ? '#28d2ff' : '#ff446f'}
              emissiveIntensity={0.52}
            />
          </mesh>
          <mesh position={[0.72, 0, 0]}>
            <boxGeometry args={[0.11, w.height / 3.3, 11.1]} />
            <meshStandardMaterial color='#f2fbff' emissive='#ffffff' emissiveIntensity={1.35} />
          </mesh>
        </group>
      ))}
    </>
  );
}
