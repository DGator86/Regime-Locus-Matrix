export default function SRWalls({ walls, onSelect }: { walls: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {walls.map((w: any) => (
        <mesh key={w.id} position={[w.x, 8, 0]} onClick={() => onSelect(w)}>
          <boxGeometry args={[1, w.height / 4, 8]} />
          <meshStandardMaterial color={w.type === 'support' ? '#27c7ff' : '#ff355e'} transparent opacity={0.35} />
        </mesh>
      ))}
    </>
  );
}
