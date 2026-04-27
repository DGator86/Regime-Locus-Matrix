export function cn(...classes: (string | undefined | false | null)[]) {
  return classes.filter(Boolean).join(' ')
}

export function fmt(n: number | null | undefined, decimals = 2): string {
  if (n == null || isNaN(n)) return '—'
  return n.toFixed(decimals)
}

export function fmtPct(n: number | null | undefined, decimals = 2): string {
  if (n == null || isNaN(n)) return '—'
  const sign = n > 0 ? '+' : ''
  return `${sign}${(n * 100).toFixed(decimals)}%`
}

export function fmtChange(n: number | null | undefined): string {
  if (n == null || isNaN(n)) return '—'
  const sign = n > 0 ? '+' : ''
  return `${sign}${n.toFixed(2)}`
}

export function scoreColor(v: number): string {
  if (v > 0.3) return '#00ff9d'
  if (v > 0.1) return '#7fffbf'
  if (v < -0.3) return '#ff3355'
  if (v < -0.1) return '#ff8888'
  return '#94a3b8'
}

export function regimeCellColor(avgReturn: number): string {
  const clamped = Math.max(-0.015, Math.min(0.015, avgReturn))
  const t = (clamped + 0.015) / 0.03
  if (t > 0.65) {
    const g = Math.round(80 + (t - 0.65) / 0.35 * 175)
    return `rgba(0,${g},80,0.7)`
  }
  if (t < 0.35) {
    const r = Math.round(100 + (0.35 - t) / 0.35 * 155)
    return `rgba(${r},20,40,0.7)`
  }
  return 'rgba(60,70,100,0.5)'
}

export function transitionRiskColor(risk: string): string {
  if (risk === 'HIGH') return '#ff3355'
  if (risk === 'MEDIUM') return '#ffaa00'
  return '#00ff9d'
}

export function directionLabel(s_d: number): string {
  if (s_d > 0.4) return 'BULLISH'
  if (s_d > 0.15) return 'MOD BULL'
  if (s_d < -0.4) return 'BEARISH'
  if (s_d < -0.15) return 'MOD BEAR'
  return 'NEUTRAL'
}

export function volLabel(s_v: number): string {
  if (s_v > 0.4) return 'HI VOL'
  if (s_v > 0.15) return 'MOD VOL'
  if (s_v < -0.15) return 'LO VOL'
  return 'MOD VOL'
}

export function liqLabel(s_l: number): string {
  return s_l > 0.1 ? 'LIQUID' : (s_l < -0.1 ? 'ILLIQUID' : 'MOD LIQ')
}

export function dealerLabel(s_g: number): string {
  return s_g > 0.1 ? 'SUPPORTIVE' : (s_g < -0.1 ? 'HOSTILE' : 'NEUTRAL')
}
