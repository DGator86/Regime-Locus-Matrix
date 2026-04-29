/**
 * Normalize HMM fields from forecast CSV / JSON where columns may be misaligned,
 * numeric junk exported into label fields, or state 0 dropped by truthy checks.
 */

export function optionalNum(raw: unknown): number | null {
  if (raw == null || raw === "") return null;
  const n = typeof raw === "number" ? raw : parseFloat(String(raw).trim());
  return Number.isFinite(n) ? n : null;
}

/** Integer HMM state index, or null if the value looks like epsilon noise. */
export function parseHmmStateIndex(raw: unknown): number | null {
  const n = optionalNum(raw);
  if (n == null) return null;
  const ir = Math.round(n);
  if (ir >= 0 && ir <= 128 && Math.abs(n - ir) < 0.501) return ir;
  if (Math.abs(n) < 1e-9) return null;
  return null;
}

/**
 * Prefer human-readable labels; reject pure numeric / scientific strings
 * (often mis-exported probabilities) and fall back to S{n}.
 */
export function sanitizeHmmStateLabel(
  raw: unknown,
  stateIdx: number | null,
): string | null {
  const s = raw != null ? String(raw).trim() : "";
  if (!s) return stateIdx != null ? `S${stateIdx}` : null;
  if (/^[-+]?\d*\.?\d+([eE][-+]?\d+)?$/.test(s)) {
    return stateIdx != null ? `S${stateIdx}` : null;
  }
  return s;
}

export function displayHmmState(line: {
  hmm_state?: number | null;
  hmm_state_label?: string | null;
}): string {
  const idx =
    line.hmm_state != null && Number.isFinite(line.hmm_state)
      ? Math.round(line.hmm_state)
      : null;
  const label = sanitizeHmmStateLabel(line.hmm_state_label, idx);
  return label ?? (idx != null ? `S${idx}` : "—");
}
