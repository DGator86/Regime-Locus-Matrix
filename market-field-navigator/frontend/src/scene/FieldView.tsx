import { useMarketFieldStore } from '../state/useMarketFieldStore';

/* ── LCARS palette ─────────────────────────────────────────── */
const C = {
  orange:    '#FF9900',
  orangeDim: '#8B4400',
  red:       '#FF2255',
  cyan:      '#00CCFF',
  cyanDim:   '#004466',
  green:     '#00CC66',
  yellow:    '#FFCC00',
  purple:    '#AA44FF',
  white:     '#FFFFFF',
  textDim:   '#334455',
} as const;

/* ── layout constants ─────────────────────────────────────── */
const VW = 900, VH = 560;
const PL = 74, PR = 118, PT = 38, PB = 82;
const FW = VW - PL - PR;
const FH = VH - PT - PB;
const IV_H = 48;
const MAX_BAR = FW * 0.72;

function py(price: number, minP: number, maxP: number): number {
  return PT + FH * (1 - (price - minP) / (maxP - minP));
}

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v;
}

export default function FieldView() {
  const snapshot = useMarketFieldStore((s) => s.snapshot);
  const toggles  = useMarketFieldStore((s) => s.toggles);
  const setSelected = useMarketFieldStore((s) => s.setSelected);

  if (!snapshot) {
    return (
      <div style={{
        background: '#000', height: '100%',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: C.orange, fontFamily: 'Courier New, monospace',
        letterSpacing: '0.14em', fontSize: 13,
      }}>
        AWAITING FEED…
      </div>
    );
  }

  const walls = snapshot.sr_walls;
  const allPrices = walls.map((w: any) => w.price);
  const priceSpan = Math.max(...allPrices) - Math.min(...allPrices);
  const pad = priceSpan * 0.28;
  const minP = Math.min(...allPrices) - pad;
  const maxP = Math.max(...allPrices) + pad;

  const curY = py(snapshot.current_price, minP, maxP);

  /* map price-relative x coord (−60…+60) to SVG field x */
  const xRelToFX = (xRel: number) =>
    clamp(PL + FW * ((xRel + 60) / 120), PL + 4, PL + FW - 4);

  return (
    <svg
      viewBox={`0 0 ${VW} ${VH}`}
      width="100%" height="100%"
      style={{ background: '#000', display: 'block' }}
    >
      <defs>
        <marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
          <polygon points="0,0 7,3.5 0,7" fill={C.yellow} />
        </marker>
        <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="2.5" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="glow-sm" x="-15%" y="-15%" width="130%" height="130%">
          <feGaussianBlur stdDeviation="1.4" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {/* ── scanner grid ─────────────────────────────────────── */}
      {Array.from({ length: 12 }, (_, i) => (
        <line key={`hg${i}`}
          x1={PL} y1={PT + FH * i / 12}
          x2={PL + FW} y2={PT + FH * i / 12}
          stroke="#090d09" strokeWidth={1} />
      ))}
      {Array.from({ length: 20 }, (_, i) => (
        <line key={`vg${i}`}
          x1={PL + FW * i / 20} y1={PT}
          x2={PL + FW * i / 20} y2={PT + FH}
          stroke="#090d09" strokeWidth={1} />
      ))}

      {/* ── regime zone tints ─────────────────────────────────── */}
      <rect x={PL} y={curY} width={FW} height={PT + FH - curY}
        fill="rgba(0,180,80,0.055)" />
      <rect x={PL} y={PT} width={FW} height={Math.max(0, curY - PT)}
        fill="rgba(255,30,60,0.055)" />

      {/* ── S/R walls ─────────────────────────────────────────── */}
      {toggles.sr && walls.map((w: any) => {
        const wy = py(w.price, minP, maxP);
        const isSup = w.type === 'support';
        const color = isSup ? C.cyan : C.red;
        const barW = MAX_BAR * w.strength;
        /* support fills from left, resistance from right */
        const barX = isSup ? PL : PL + FW - barW;

        return (
          <g key={w.id} style={{ cursor: 'pointer' }}
            onClick={() => setSelected({ type: 'wall', ...w })}>
            {/* full-width dashed hairline */}
            <line x1={PL} y1={wy} x2={PL + FW} y2={wy}
              stroke={color} strokeWidth={0.5} strokeDasharray="5 5" opacity={0.22} />
            {/* filled strength slab */}
            <rect x={barX} y={wy - 4} width={barW} height={8}
              fill={color} opacity={0.5} rx={1} />
            {/* bright leading edge */}
            <rect
              x={isSup ? PL : PL + FW - 3}
              y={wy - 6} width={3} height={12}
              fill={color} rx={1} filter="url(#glow-sm)" />
            {/* price label – left side */}
            <text x={PL - 6} y={wy + 4} textAnchor="end"
              fill={color} fontSize={11}
              fontFamily="'Courier New', monospace" fontWeight="bold">
              {w.price.toFixed(0)}
            </text>
            {/* wall label – right side */}
            <text x={PL + FW + 8} y={wy + 4} textAnchor="start"
              fill={color} fontSize={9}
              fontFamily="'Courier New', monospace" letterSpacing="1" opacity={0.9}>
              {w.label.toUpperCase()}
            </text>
          </g>
        );
      })}

      {/* ── liquidity wells ───────────────────────────────────── */}
      {toggles.liquidity && snapshot.liquidity_wells.slice(0, 8).map((l: any) => {
        const ly = py(l.price, minP, maxP);
        if (ly < PT + 4 || ly > PT + FH - 4) return null;
        const lx = xRelToFX(l.x * 0.9);
        const r = 4 + l.strength * 9;
        const color = l.type === 'supportive' ? C.cyan : C.purple;
        return (
          <g key={l.id} opacity={0.8} style={{ cursor: 'pointer' }}
            onClick={() => setSelected({ type: 'liquidity', ...l })}>
            <circle cx={lx} cy={ly} r={r * 1.9}
              fill="none" stroke={color} strokeWidth={0.5} opacity={0.22} />
            <circle cx={lx} cy={ly} r={r}
              fill="none" stroke={color} strokeWidth={1.5} />
            <circle cx={lx} cy={ly} r={2.5} fill={color} />
          </g>
        );
      })}

      {/* ── gamma force vectors ───────────────────────────────── */}
      {toggles.gamma && snapshot.gamma_vectors.map((g: any) => {
        const gy = clamp(
          curY + g.origin.y * 2.2 - 8,
          PT + 6, PT + FH - 6
        );
        const gx = xRelToFX(g.origin.x * 0.9);
        const len = 16 + g.magnitude * 52;
        return (
          <line key={g.id}
            x1={gx} y1={gy}
            x2={Math.min(gx + len, PL + FW - 4)} y2={gy}
            stroke={C.yellow} strokeWidth={2.2}
            markerEnd="url(#arr)"
            opacity={0.82}
          />
        );
      })}

      {/* ── current price cursor ──────────────────────────────── */}
      <g filter="url(#glow-sm)">
        <line x1={PL} y1={curY} x2={PL + FW} y2={curY}
          stroke={C.white} strokeWidth={1.8} />
        {/* blinking LCARS triangle */}
        <polygon
          points={`${PL - 2},${curY - 7} ${PL - 17},${curY} ${PL - 2},${curY + 7}`}
          fill={C.orange}
        >
          <animate attributeName="opacity" values="1;0.3;1" dur="1.4s" repeatCount="indefinite" />
        </polygon>
        <text x={PL - 20} y={curY + 4} textAnchor="end"
          fill={C.orange} fontSize={12.5}
          fontFamily="'Courier New', monospace" fontWeight="bold">
          {snapshot.current_price.toFixed(2)}
        </text>
      </g>

      {/* ── IV heatmap strip ─────────────────────────────────── */}
      {toggles.iv && (() => {
        const sy = PT + FH + 14;
        const sh = IV_H - 16;
        const pts: any[] = snapshot.iv_surface?.points ?? [];
        const GRDX = 32, GRDY = 32;
        const cellW = FW / GRDX;

        return (
          <g>
            <text x={PL} y={sy - 4} fill={C.textDim} fontSize={9}
              fontFamily="'Courier New', monospace" letterSpacing="2">
              IMPLIED VOLATILITY SURFACE
            </text>
            {Array.from({ length: GRDX }, (_, ix) => {
              let tot = 0, cnt = 0;
              for (let iy = 0; iy < GRDY; iy++) {
                const p = pts[ix * GRDY + iy];
                if (p) { tot += p.iv; cnt++; }
              }
              const av = cnt ? tot / cnt : 0.3;
              const t = clamp((av - 0.22) / 0.28, 0, 1);
              /* blue → cyan → green → yellow → red */
              const r = Math.round(20 + t * 220);
              const g2 = Math.round(100 - t * 50);
              const b = Math.round(220 - t * 200);
              return (
                <rect key={ix}
                  x={PL + ix * cellW} y={sy}
                  width={cellW + 0.5} height={sh}
                  fill={`rgb(${r},${g2},${b})`}
                />
              );
            })}
            <rect x={PL} y={sy} width={FW} height={sh}
              fill="none" stroke="#182030" strokeWidth={0.5} />
            <text x={PL} y={sy + sh + 11} fill={C.textDim} fontSize={8}
              fontFamily="'Courier New', monospace">LOW VOL</text>
            <text x={PL + FW} y={sy + sh + 11} textAnchor="end"
              fill={C.textDim} fontSize={8}
              fontFamily="'Courier New', monospace">HIGH VOL</text>
          </g>
        );
      })()}

      {/* ── LCARS frame decorations ───────────────────────────── */}
      {/* top rule + label */}
      <rect x={PL} y={PT - 10} width={FW} height={2}
        fill={C.orange} opacity={0.65} />
      <text x={PL} y={PT - 13} fill={C.orangeDim} fontSize={9}
        fontFamily="'Courier New', monospace" letterSpacing="3">
        MARKET FIELD · FIELD VIEW
      </text>
      <text x={PL + FW} y={PT - 13} textAnchor="end" fill={C.orangeDim} fontSize={9}
        fontFamily="'Courier New', monospace" letterSpacing="3">
        {snapshot.symbol} · {snapshot.regime.label.toUpperCase()}
      </text>

      {/* left LCARS bumper stripe */}
      <rect x={0} y={0} width={PL - 14} height={VH} fill="#000" />
      <rect x={PL - 14} y={PT} width={2} height={FH}
        fill={C.orangeDim} opacity={0.7} />

      {/* right LCARS bumper stripe */}
      <rect x={PL + FW + 12} y={0} width={VW - PL - FW - 12} height={VH} fill="#000" />
      <rect x={PL + FW + 12} y={PT} width={2} height={FH}
        fill={C.orangeDim} opacity={0.7} />

      {/* bottom divider */}
      <rect x={PL} y={PT + FH + 6} width={FW} height={2}
        fill={C.orangeDim} opacity={0.45} />

      {/* field border */}
      <rect x={PL} y={PT} width={FW} height={FH}
        fill="none" stroke="#141e14" strokeWidth={0.5} />
    </svg>
  );
}
