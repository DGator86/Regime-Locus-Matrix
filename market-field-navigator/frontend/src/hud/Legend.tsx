const items = [
  ['#ff78b0', 'Price particle'],
  ['#ffd061', 'Gamma vectors'],
  ['#72b5ff', 'IV surface'],
  ['#66e8ff', 'Liquidity wells'],
  ['#73dbff', 'Support / resistance walls'],
  ['#5dffb8', 'Regime zones'],
] as const;

export default function Legend() {
  return (
    <section className='panel'>
      <p className='title'>Legend</p>
      {items.map(([color, label]) => (
        <div className='legend-item' key={label}>
          <span className='dot' style={{ color }} />
          <span>{label}</span>
        </div>
      ))}
    </section>
  );
}
