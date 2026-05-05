import { useMemo } from 'react';
import { BufferGeometry, Float32BufferAttribute } from 'three';
import { Line } from '@react-three/drei';

export default function IVSurface({ points }: { points: any[] }) {
  const { geometry, gridX, gridY } = useMemo(() => {
    const xs = [...new Set(points.map((p: any) => p.x))].sort((a, b) => a - b);
    const ys = [...new Set(points.map((p: any) => p.y))].sort((a, b) => a - b);
    const lookup = new Map(points.map((p: any) => [`${p.x}|${p.y}`, p]));
    const vertices: number[] = [];
    const indices: number[] = [];

    for (let yi = 0; yi < ys.length; yi += 1) {
      for (let xi = 0; xi < xs.length; xi += 1) {
        const p = lookup.get(`${xs[xi]}|${ys[yi]}`);
        if (!p) continue;
        vertices.push(p.x, p.y, p.z - 2.4);
      }
    }

    for (let yi = 0; yi < ys.length - 1; yi += 1) {
      for (let xi = 0; xi < xs.length - 1; xi += 1) {
        const a = yi * xs.length + xi;
        const b = a + 1;
        const c = a + xs.length;
        const d = c + 1;
        indices.push(a, b, d, a, d, c);
      }
    }

    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(vertices, 3));
    g.setIndex(indices);
    g.computeVertexNormals();
    return { geometry: g, gridX: xs.length, gridY: ys.length };
  }, [points]);

  const wireRows = useMemo(() => {
    const rows: [number, number, number][][] = [];
    const sampleStep = Math.max(1, Math.floor(gridY / 6));
    for (let y = 0; y < gridY; y += sampleStep) {
      const row: [number, number, number][] = [];
      for (let x = 0; x < gridX; x += 1) {
        const idx = (y * gridX + x) * 3;
        const pos = geometry.attributes.position.array as ArrayLike<number>;
        row.push([pos[idx], pos[idx + 1], pos[idx + 2]]);
      }
      rows.push(row);
    }
    return rows;
  }, [geometry, gridX, gridY]);

  return (
    <>
      <mesh geometry={geometry}>
        <meshStandardMaterial color='#2f79e0' emissive='#1e4ea1' emissiveIntensity={0.62} transparent opacity={0.55} side={2} />
      </mesh>
      {wireRows.map((row, idx) => (
        <Line key={`iv-row-${idx}`} points={row} color='#84c7ff' lineWidth={1.8} transparent opacity={0.72} />
      ))}
    </>
  );
}
