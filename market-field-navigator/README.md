# Market Field Navigator

Interactive 3D market field interface showing price as a particle moving through regime, gamma, liquidity, volatility, and support/resistance fields.

## Run backend
```bash
cd market-field-navigator/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload
```

## Run frontend
```bash
cd market-field-navigator/frontend
npm install
npm run dev
```

Open http://localhost:5173

## Current status
Synthetic data only via `RLMAdapter` stub.

## Next step
Connect `RLMAdapter` to real Regime-Locus-Matrix outputs while keeping backend as translation layer.
