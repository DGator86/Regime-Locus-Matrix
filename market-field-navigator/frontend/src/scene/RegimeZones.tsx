export default function RegimeZones({ zones, onSelect }: { zones: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {zones.map((z: any) => (
        <group key={z.id} position={[z.center.x, z.center.y, z.center.z]} onClick={() => onSelect(z)}>
          <mesh>
            <sphereGeometry args={[z.size.x * 0.4, 28, 22]} />
            <meshStandardMaterial
              color={z.type === 'bull' ? '#20f58d' : z.type === 'bear' ? '#ff305a' : '#ffc845'}
              emissive={z.type === 'bull' ? '#20f58d' : z.type === 'bear' ? '#ff305a' : '#ffc845'}
              emissiveIntensity={0.58}
              transparent
              opacity={0.06 + z.opacity * 0.35}
            />
          </mesh>
          <mesh scale={[1.3, 0.9, 1.1]}>
            <sphereGeometry args={[z.size.x * 0.34, 18, 16]} />
            <meshStandardMaterial
              color={z.type === 'bull' ? '#48ffd0' : z.type === 'bear' ? '#ff7f9f' : '#ffe17a'}
              transparent
              opacity={0.1 + z.opacity * 0.4}
              emissive='#ffffff'
              emissiveIntensity={0.08}
            />
          </mesh>
        </group>
      ))}
    </>
  );
}
