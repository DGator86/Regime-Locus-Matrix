export async function fetchSnapshot(symbol = 'SPY') {
  const res = await fetch(`http://localhost:8000/api/market-field/snapshot?symbol=${symbol}`);
  if (!res.ok) throw new Error('snapshot fetch failed');
  return res.json();
}
