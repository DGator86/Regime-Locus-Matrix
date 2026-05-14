import { create } from 'zustand';

type State = { snapshot: any | null; selected: any | null; setSnapshot: (s: any) => void; setSelected: (s: any) => void; toggles: Record<string, boolean>; setToggle: (k: string, v: boolean) => void };

export const useMarketFieldStore = create<State>((set) => ({
  snapshot: null,
  selected: null,
  setSnapshot: (snapshot) => set({ snapshot }),
  setSelected: (selected) => set({ selected }),
  toggles: { regime: true, gamma: true, iv: true, liquidity: true, sr: true, path: true },
  setToggle: (k, v) => set((st) => ({ toggles: { ...st.toggles, [k]: v } })),
}));
