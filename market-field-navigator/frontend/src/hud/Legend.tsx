const ITEMS = [
  { color: '#FF9900', label: 'Current Price' },
  { color: '#FFCC00', label: 'Gamma Vectors' },
  { color: '#3399FF', label: 'IV Surface' },
  { color: '#00CCFF', label: 'Support / Liquidity' },
  { color: '#FF2255', label: 'Resistance' },
  { color: '#AA44FF', label: 'Overhead Liquidity' },
  { color: '#00CC66', label: 'Bull Zone' },
  { color: '#FF2255', label: 'Bear Zone' },
];

export default function Legend() {
  return (
    <section className='panel'>
      <div className='sec'>Legend</div>
      {ITEMS.map(({ color, label }) => (
        <div className='leg-row' key={label}>
          <span className='leg-dot' style={{ background: color }} />
          {label}
        </div>
      ))}
    </section>
  );
}
