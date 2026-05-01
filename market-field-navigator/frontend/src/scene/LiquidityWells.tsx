export default function LiquidityWells({ wells, onSelect }: { wells: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {wells.map((l: any) => (
        <mesh key={l.id} position={[l.x, -2, 0]} onClick={() => onSelect(l)}>
          <torusGeometry args={[1.2, 0.3, 16, 60]} />
          <meshStandardMaterial color={'#27c7ff'} emissive={'#27c7ff'} />
        </mesh>
      ))}
    </>
  );
}
