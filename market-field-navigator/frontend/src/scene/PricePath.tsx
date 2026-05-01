import { Line } from '@react-three/drei';

export default function PricePath({ points }: { points: any[] }) {
  if (points.length < 2) return null;
  const pts = points.map((p: any) => [p.x, p.y + 0.5, p.z] as [number, number, number]);
  return (
    <>
      <Line points={pts} color="#ff4488" lineWidth={3} />
      {points.map((p: any, i: number) => (
        <mesh key={`path-dot-${i}`} position={[p.x, p.y + 0.5, p.z]}>
          <sphereGeometry args={[0.35, 14, 14]} />
          <meshStandardMaterial color="#ff4488" emissive="#ff4488" emissiveIntensity={4} />
        </mesh>
      ))}
    </>
  );
}
