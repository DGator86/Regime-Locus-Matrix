import { useEffect, useRef } from 'react'
import {
  createChart, ColorType, CrosshairMode,
  type IChartApi, type ISeriesApi, type CandlestickData,
  type HistogramData, type LineData,
} from 'lightweight-charts'
import type { ForecastBar } from '../../lib/types'

interface Props {
  data: ForecastBar[]
  height?: number
}

function stateToColor(code: string): string {
  const dir = parseInt(code.slice(1)) || 5
  const vol = code.charCodeAt(0) - 65 // A=0 .. J=9
  if (dir >= 7 && vol <= 3) return 'rgba(0,255,157,0.12)'
  if (dir <= 3 && vol >= 6) return 'rgba(255,51,85,0.12)'
  if (dir >= 6 && vol <= 5) return 'rgba(0,200,120,0.07)'
  if (dir <= 4) return 'rgba(255,80,80,0.07)'
  return 'rgba(255,170,0,0.05)'
}

export default function PriceChart({ data, height = 320 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const confRef = useRef<ISeriesApi<'Line'> | null>(null)

  useEffect(() => {
    if (!containerRef.current) return
    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#64748b',
        fontSize: 10,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.03)' },
        horzLines: { color: 'rgba(255,255,255,0.03)' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: 'rgba(255,255,255,0.08)' },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.08)',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    const candle = chart.addCandlestickSeries({
      upColor: '#00ff9d',
      downColor: '#ff3355',
      borderUpColor: '#00ff9d',
      borderDownColor: '#ff3355',
      wickUpColor: '#00cc7a',
      wickDownColor: '#cc2244',
      priceScaleId: 'right',
    })

    const conf = chart.addLineSeries({
      color: 'rgba(0,245,255,0.7)',
      lineWidth: 1,
      priceScaleId: 'left',
      title: 'Confidence',
    })

    chart.priceScale('left').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
      visible: false,
    })

    chartRef.current = chart
    candleRef.current = candle
    confRef.current = conf

    const ro = new ResizeObserver(() => {
      if (containerRef.current) chart.resize(containerRef.current.clientWidth, height)
    })
    ro.observe(containerRef.current)
    return () => { ro.disconnect(); chart.remove() }
  }, [height])

  useEffect(() => {
    if (!candleRef.current || !confRef.current || !data.length) return

    const candles: CandlestickData[] = []
    const confData: LineData[] = []

    for (const b of data) {
      if (!b.close) continue
      const time = b.timestamp.split('T')[0] as unknown as CandlestickData['time']
      candles.push({
        time,
        open: b.open ?? b.close,
        high: b.high ?? b.close,
        low: b.low ?? b.close,
        close: b.close,
      })
      if (b.hmm_confidence != null) {
        confData.push({ time, value: b.hmm_confidence })
      }
    }

    candleRef.current.setData(candles)
    if (confData.length) confRef.current.setData(confData)
    chartRef.current?.timeScale().fitContent()
  }, [data])

  // Background regime bands via markers approach — just render colored overlay divs
  return (
    <div className="relative w-full" style={{ height }}>
      <div ref={containerRef} className="absolute inset-0" />
    </div>
  )
}
