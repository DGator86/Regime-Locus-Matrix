import { useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/layout/Sidebar'
import TopBar from './components/layout/TopBar'
import TickerBar from './components/layout/TickerBar'
import Overview from './pages/Overview'
import StateMap from './pages/StateMap'
import Factors from './pages/Factors'
import Forecast from './pages/Forecast'
import BacktestLab from './pages/BacktestLab'
import LiveDesk from './pages/LiveDesk'
import Diagnostics from './pages/Diagnostics'
import Placeholder from './pages/Placeholder'

export default function App() {
  const [symbol, setSymbol] = useState('SPY')
  const [timeframe, setTimeframe] = useState('1D')
  const [profile, setProfile] = useState('Default Pro')

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-navy-950">
      {/* Top bar */}
      <TopBar
        symbol={symbol} onSymbolChange={setSymbol}
        timeframe={timeframe} onTimeframeChange={setTimeframe}
        profile={profile} onProfileChange={setProfile}
      />

      {/* Main area: sidebar + content */}
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />

        <main className="flex flex-col flex-1 overflow-hidden">
          <Routes>
            <Route path="/" element={<Overview symbol={symbol} />} />
            <Route path="/state-map" element={<StateMap symbol={symbol} />} />
            <Route path="/factors" element={<Factors symbol={symbol} />} />
            <Route path="/forecast" element={<Forecast symbol={symbol} />} />
            <Route path="/roee" element={<Placeholder title="ROEE" description="Risk-On / Risk-Off Engine analysis coming soon." />} />
            <Route path="/chain" element={<Placeholder title="Options Chain" description="Live options chain and Greeks viewer coming soon." />} />
            <Route path="/backtest" element={<BacktestLab symbol={symbol} />} />
            <Route path="/live-desk" element={<LiveDesk />} />
            <Route path="/diagnostics" element={<Diagnostics />} />
            <Route path="/settings" element={<Placeholder title="Settings" description="Configuration and preferences coming soon." />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>

      {/* Bottom ticker */}
      <TickerBar />
    </div>
  )
}
