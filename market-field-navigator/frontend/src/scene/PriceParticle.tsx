export default function PriceParticle({ x, onClick }: { x: number; onClick: () => void }) {
  return (
    <mesh position={[x, 0, 0]} onClick={onClick}>
      <sphereGeometry args={[1.2, 32, 32]} />
      <meshStandardMaterial emissive={'#ff3b7f'} color={'#ff3b7f'} />
    </mesh>
  );
}
