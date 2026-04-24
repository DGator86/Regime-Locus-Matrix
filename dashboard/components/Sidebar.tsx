"use client";

import React from "react";
import { 
  LayoutDashboard, 
  Map, 
  Activity, 
  BarChart3, 
  Settings, 
  ShieldCheck,
  ChevronRight
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const menuItems = [
  { icon: LayoutDashboard, label: "Overview", href: "/" },
  { icon: Map, label: "State Map", href: "/state-map" },
  { icon: Activity, label: "Locus Matrix", href: "/matrix" },
  { icon: BarChart3, label: "Analysis", href: "/analysis" },
  { icon: ShieldCheck, label: "Risk Center", href: "/risk" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 glass border-r border-border flex flex-col h-full z-10">
      <div className="p-6 flex items-center gap-3">
        <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center neon-border">
          <ShieldCheck className="text-primary-foreground w-5 h-5" />
        </div>
        <span className="font-bold text-xl tracking-tight neon-text">RLM <span className="text-primary opacity-80">v2</span></span>
      </div>

      <nav className="flex-1 px-4 py-6 space-y-2">
        {menuItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex items-center justify-between px-4 py-3 rounded-xl transition-all duration-200 group",
              pathname === item.href 
                ? "bg-primary/10 text-primary border border-primary/20" 
                : "text-muted-foreground hover:text-foreground hover:bg-secondary"
            )}
          >
            <div className="flex items-center gap-3">
              <item.icon className={cn("w-5 h-5", pathname === item.href && "neon-text")} />
              <span className="font-medium">{item.label}</span>
            </div>
            {pathname === item.href && (
              <ChevronRight className="w-4 h-4" />
            )}
          </Link>
        ))}
      </nav>

      <div className="p-4 border-t border-border mt-auto">
        <div className="bg-secondary/50 rounded-xl p-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-accent/20 flex items-center justify-center">
            <Activity className="text-accent w-5 h-5" />
          </div>
          <div>
            <p className="text-xs text-muted-foreground">System Status</p>
            <p className="text-sm font-semibold text-green-400 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              Operational
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}
