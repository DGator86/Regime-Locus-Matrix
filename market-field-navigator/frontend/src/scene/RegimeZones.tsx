import * as THREE from 'three';

const ZONE_COLORS: Record<string, { main: string; emissive: string }> = {
  bull: { main: '#00ff88', emissive: '#00ff88' },
  bear: { main: '#ff2255', emissive: '#ff2255' },
  chop: { main: '#ffc845', emissive: '#ffc845' },
};

function RegimeCloud({ zone, onSelect }: { zone: any; onSelect: (z: any) => void }) {
  const { main, emissive } = ZONE_COLORS[zone.type] ?? ZONE_COLORS.chop;
  const rx = zone.size.x * 0.55;
  const ry = zone.size.y * 0.5;
  const rz = zone.size.z * 0.55;

  return (
    <group
      position={[zone.center.x, zone.center.y, zone.center.z]}
      onClick={(e) => { e.stopPropagation(); onSelect(zone); }}
    >
      {/* outer volume — backside for interior glow */}
      <mesh>
        <sphereGeometry args={[Math.max(rx, ry, rz) * 1.05, 18, 12]} />
        <meshStandardMaterial
          color={main}
          emissive={emissive}
          emissiveIntensity={0.4}
          transparent
          opacity={zone.opacity * 0.35}
          side={THREE.BackSide}
          depthWrite={false}
        />
      </mesh>
      {/* mid cloud */}
      <mesh scale={[rx / ry, 1, rz / ry]}>
        <sphereGeometry args={[ry * 0.8, 14, 10]} />
        <meshStandardMaterial
          color={main}
          emissive={emissive}
          emissiveIntensity={0.6}
          transparent
          opacity={zone.opacity * 0.55}
          depthWrite={false}
        />
      </mesh>
      {/* bright inner core */}
      <mesh scale={[rx / ry, 1, rz / ry]}>
        <sphereGeometry args={[ry * 0.35, 10, 8]} />
        <meshStandardMaterial
          color={main}
          emissive={emissive}
          emissiveIntensity={2.5}
          transparent
          opacity={zone.opacity * 0.8}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

export default function RegimeZones({ zones, onSelect }: { zones: any[]; onSelect: (v: any) => void }) {
  return (
    <>
      {zones.map((z: any) => (
        <RegimeCloud key={z.id} zone={z} onSelect={onSelect} />
      ))}
    </>
  );
}
