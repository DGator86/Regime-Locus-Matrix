# Market Field Navigator

Interactive 3D market field interface showing price as a particle moving through regime, gamma, liquidity, volatility, and support/resistance fields.

## Run backend
```bash
cd market-field-navigator/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload --port 8000
```

## Run frontend
```bash
cd market-field-navigator/frontend
npm install
npm run dev
```

Open http://localhost:5173

## Troubleshooting (CORS / API)
- Backend enables CORS for `http://localhost:5173`.
- If UI shows "Backend offline", verify backend is running on port `8000`.
- Verify the snapshot endpoint works directly:
  `curl http://localhost:8000/api/market-field/snapshot?symbol=SPY`

## What you should see
- Dark cockpit layout with left metrics, center 3D scene, and right decision/scale panels.
- Clickable particle, gamma cones, liquidity rings, and support/resistance wall objects.

## Synthetic-only disclaimer
This module currently uses synthetic data from `RLMAdapter` and is decision-support visualization only.
It does not place trades and makes no predictive accuracy claims.
