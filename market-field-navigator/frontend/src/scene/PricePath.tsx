import { Line } from '@react-three/drei';

export default function PricePath({ points }: { points: any[] }) {
  const linePoints = points.map((p: any) => [p.x, p.y, p.z] as [number, number, number]);
  return (
    <>
      <Line points={linePoints} color='#ff93c0' lineWidth={2.8} transparent opacity={0.95} />
      {points.map((p: any, i: number) => (
        <mesh key={`${p.price}-${i}`} position={[p.x, p.y, p.z]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial color='#ffc3db' emissive='#ff75aa' emissiveIntensity={1.1} />
        </mesh>
      ))}
    </>
  );
}
