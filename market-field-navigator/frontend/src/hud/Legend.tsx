const ITEMS = [
  { color: '#ff2255', label: 'Price Particle' },
  { color: '#ffc845', label: 'Gamma Vectors' },
  { color: '#0088ff', label: 'IV Surface' },
  { color: '#27c7ff', label: 'Liquidity Wells' },
  { color: '#ff2255', label: 'S/R Resistance' },
  { color: '#00d4ff', label: 'S/R Support' },
  { color: '#00ff88', label: 'Bull Zone' },
  { color: '#ff2255', label: 'Bear Zone' },
  { color: '#ffc845', label: 'Chop Zone' },
];

export default function Legend() {
  return (
    <section className='panel' style={{ flex: '0 0 auto' }}>
      <div className='section-label'>Legend</div>
      {ITEMS.map(({ color, label }) => (
        <div className='legend-row' key={label}>
          <span className='legend-dot' style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
          {label}
        </div>
      ))}
    </section>
  );
}
