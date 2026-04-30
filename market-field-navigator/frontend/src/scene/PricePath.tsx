export default function PricePath({ points }: { points: any[] }) {
  return (
    <>
      {points.map((p: any, i: number) => (
        <mesh key={`${p.price}-${i}`} position={[p.x, p.y, p.z]}>
          <sphereGeometry args={[0.25, 10, 10]} />
          <meshStandardMaterial color={'#ff83ad'} />
        </mesh>
      ))}
    </>
  );
}
