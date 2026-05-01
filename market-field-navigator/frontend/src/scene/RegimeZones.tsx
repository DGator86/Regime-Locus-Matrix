export default function RegimeZones({ zones, onSelect }: { zones: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {zones.map((z: any) => (
        <mesh key={z.id} position={[z.center.x, z.center.y, z.center.z]} onClick={() => onSelect(z)}>
          <boxGeometry args={[z.size.x, z.size.y, z.size.z]} />
          <meshStandardMaterial color={z.type === 'bull' ? '#28f56d' : z.type === 'bear' ? '#ff355e' : '#ffc845'} transparent opacity={z.opacity} wireframe />
        </mesh>
      ))}
    </>
  );
}
