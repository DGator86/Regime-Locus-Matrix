export default function IVSurface({ points }: { points: any[] }) {
  const sampled = points.filter((_: any, i: number) => i % 16 === 0).slice(0, 120);
  return (
    <>
      {sampled.map((p: any, i: number) => (
        <mesh key={`iv-${i}`} position={[p.x, p.y, p.z - 2]}>
          <sphereGeometry args={[0.1, 8, 8]} />
          <meshStandardMaterial color={'#2f8cff'} transparent opacity={0.65} />
        </mesh>
      ))}
    </>
  );
}
