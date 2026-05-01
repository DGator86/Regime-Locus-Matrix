import { useMemo } from 'react';
import * as THREE from 'three';

const GRID_X = 32;
const GRID_Y = 32;

export default function IVSurface({ points }: { points: any[] }) {
  const geometry = useMemo(() => {
    if (points.length < GRID_X * GRID_Y) return null;

    const positions = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
      positions[i * 3]     = p.x * 0.6;           // price axis
      positions[i * 3 + 1] = p.z * 0.28 - 10;    // iv height → y, sits below scene
      positions[i * 3 + 2] = p.y * 0.55;          // time axis → depth
    });

    const indices: number[] = [];
    for (let ix = 0; ix < GRID_X - 1; ix++) {
      for (let iy = 0; iy < GRID_Y - 1; iy++) {
        const a = ix * GRID_Y + iy;
        const b = ix * GRID_Y + iy + 1;
        const c = (ix + 1) * GRID_Y + iy;
        const d = (ix + 1) * GRID_Y + iy + 1;
        indices.push(a, c, b);
        indices.push(b, c, d);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }, [points]);

  if (!geometry) return null;

  return (
    <>
      {/* solid surface */}
      <mesh geometry={geometry}>
        <meshStandardMaterial
          color="#0044cc"
          emissive="#003399"
          emissiveIntensity={1.2}
          transparent
          opacity={0.55}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      {/* wireframe overlay for the grid lines */}
      <mesh geometry={geometry}>
        <meshStandardMaterial
          color="#00aaff"
          emissive="#00aaff"
          emissiveIntensity={2}
          transparent
          opacity={0.25}
          wireframe
          depthWrite={false}
        />
      </mesh>
    </>
  );
}
