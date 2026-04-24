"use client";

import React, { useEffect, useState } from "react";
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Percent, 
  Target,
  ArrowUpRight,
  RefreshCcw
} from "lucide-react";
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from "recharts";
import { motion } from "framer-motion";

const data = [
  { name: "09:00", pnl: 0 },
  { name: "10:00", pnl: 1200 },
  { name: "11:00", pnl: 800 },
  { name: "12:00", pnl: 2400 },
  { name: "13:00", pnl: 1800 },
  { name: "14:00", pnl: 3200 },
  { name: "15:00", pnl: 4500 },
];

const MetricCard = ({ title, value, delta, icon: Icon, color }: any) => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="glass rounded-2xl p-6 relative overflow-hidden"
  >
    <div className={`absolute top-0 right-0 w-32 h-32 bg-${color}-500/10 rounded-full -mr-16 -mt-16 blur-2xl`} />
    <div className="flex justify-between items-start mb-4">
      <div className={`p-3 rounded-xl bg-secondary border border-border shadow-lg`}>
        <Icon className={`w-6 h-6 text-${color}-400`} />
      </div>
      {delta && (
        <span className={`text-xs font-bold px-2 py-1 rounded-lg ${delta > 0 ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}`}>
          {delta > 0 ? "+" : ""}{delta}%
        </span>
      )}
    </div>
    <p className="text-muted-foreground text-sm font-medium">{title}</p>
    <h3 className="text-2xl font-bold mt-1 tracking-tight">{value}</h3>
  </motion.div>
);

export default function Overview() {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) return null;

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      <header className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Strategy Overview</h1>
          <p className="text-muted-foreground mt-1">Real-time performance monitoring across all regimes.</p>
        </div>
        <button className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-2 rounded-xl font-semibold hover:opacity-90 transition-opacity">
          <RefreshCcw className="w-4 h-4" />
          Refresh
        </button>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard title="Account Balance" value="$21,450.80" delta={4.2} icon={DollarSign} color="blue" />
        <MetricCard title="Realized P/L" value="+$3,240.15" delta={12.5} icon={TrendingUp} color="green" />
        <MetricCard title="Active Risk" value="$1,120.00" icon={Target} color="purple" />
        <MetricCard title="Win Rate" value="64.2%" delta={-1.4} icon={Percent} color="amber" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 glass rounded-2xl p-8">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-xl font-bold">Equity Performance</h2>
            <div className="flex gap-2">
              <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded border border-border">Live</span>
            </div>
          </div>
          <div className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00f5ff" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#00f5ff" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#1e1e24" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} dy={10} />
                <YAxis axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 12}} tickFormatter={(v) => `$${v}`} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#111114', border: '1px solid #1e1e24', borderRadius: '12px' }}
                  itemStyle={{ color: '#00f5ff' }}
                />
                <Area type="monotone" dataKey="pnl" stroke="#00f5ff" strokeWidth={3} fillOpacity={1} fill="url(#colorPnl)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass rounded-2xl p-8 flex flex-col">
          <h2 className="text-xl font-bold mb-6">Active Signals</h2>
          <div className="flex-1 space-y-6">
            {[
              { sym: "SPY", type: "LONG", time: "10m ago", score: 0.82 },
              { sym: "QQQ", type: "HOLD", time: "25m ago", score: 0.12 },
              { sym: "TSLA", type: "EXIT", time: "1h ago", score: -0.45 },
            ].map((item, i) => (
              <div key={i} className="flex items-center justify-between p-4 bg-secondary/30 rounded-xl border border-border">
                <div className="flex items-center gap-4">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center font-bold ${
                    item.type === "LONG" ? "bg-green-500/20 text-green-400" : 
                    item.type === "EXIT" ? "bg-red-500/20 text-red-400" : "bg-blue-500/20 text-blue-400"
                  }`}>
                    {item.sym[0]}
                  </div>
                  <div>
                    <p className="font-bold">{item.sym}</p>
                    <p className="text-xs text-muted-foreground">{item.time}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`font-bold ${item.score > 0 ? "text-green-400" : "text-red-400"}`}>{item.type}</p>
                  <p className="text-xs text-muted-foreground">Conf: {(item.score * 100).toFixed(0)}%</p>
                </div>
              </div>
            ))}
          </div>
          <button className="mt-8 text-sm text-primary font-semibold flex items-center justify-center gap-2 hover:underline">
            View all signals <ArrowUpRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
