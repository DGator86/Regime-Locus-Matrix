export interface Scores {
  S_D: number; S_V: number; S_L: number; S_G: number;
}

export interface MarketState {
  direction: string; volatility: string; liquidity: string; dealer_flow: string;
}

export interface Action {
  type: 'ENTER' | 'HOLD' | 'EXIT';
  strategy: string;
  size_pct: number;
  rationale: string;
}

export interface Risk {
  uncertainty_pct: number;
  vault_active: boolean;
  vp_gating: string;
  environment: string;
  drawdown_risk: string;
}

export interface QuickStats {
  avg_return: number; win_rate: number; trades: number;
  expectancy: number; best_strategy: string;
}

export interface NextState { code: string; prob: number; }

export interface Alert { level: 'warning' | 'info' | 'success' | 'error'; title: string; body: string; }

export interface WhyRLM {
  top_drivers: string[]; top_penalties: string[]; key_confluences: string[];
}

export interface RecentTransition {
  prev_code: string; curr_code: string; transition_type: string;
  stability_score: number; bars_in_state: number; early_warning: string;
}

export interface Overview {
  symbol: string; timestamp: string; close: number;
  scores: Scores; market_state: MarketState;
  state_code: string; hmm_state: number; hmm_confidence: number;
  confidence: number; markov_prob: number; transition_risk: string;
  action: Action; risk: Risk; quick_stats: QuickStats;
  next_states: NextState[]; alerts: Alert[]; why_rlm: WhyRLM;
  recent_transitions: RecentTransition;
}

export interface ForecastBar {
  timestamp: string; open?: number; high?: number; low?: number;
  close: number; volume?: number;
  S_D?: number; S_V?: number; S_L?: number; S_G?: number;
  hmm_state?: number; hmm_confidence?: number;
  forecast_return_lower?: number; forecast_return_median?: number; forecast_return_upper?: number;
  realized_vol?: number; state_code: string;
  surface_atm_forward_iv?: number;
}

export interface GridCell {
  dir_bin: number; vol_bin: number; state_code: string;
  avg_return: number; count: number; win_rate: number;
}

export interface GridPoint {
  dir_bin: number; vol_bin: number; state_code: string; timestamp: string;
}

export interface RegimeGrid {
  cells: GridCell[];
  trajectory: GridPoint[];
  current: { dir_bin: number; vol_bin: number; state_code: string };
}

export interface TickerItem {
  symbol: string; price: number | null;
  change: number | null; change_pct: number | null;
}

export interface UniverseRow {
  symbol: string; status: string; regime_key: string; state_code: string;
  S_D: number; S_V: number; S_L: number; S_G: number;
  action: string; strategy: string; confidence: number;
}
