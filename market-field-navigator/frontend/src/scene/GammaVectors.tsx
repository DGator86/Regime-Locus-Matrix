export default function GammaVectors({ vectors, onSelect }: { vectors: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {vectors.map((g: any) => (
        <mesh key={g.id} position={[g.origin.x, g.origin.y, g.origin.z]} onClick={() => onSelect(g)}>
          <coneGeometry args={[0.4, 2 + g.magnitude * 2, 8]} />
          <meshStandardMaterial color={'#ffc845'} />
        </mesh>
      ))}
    </>
  );
}
