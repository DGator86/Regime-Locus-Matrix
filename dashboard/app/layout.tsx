import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import StatusBar from "@/components/StatusBar";
import AppHeader from "@/components/AppHeader";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "RLM · Regime Locus Matrix",
  description: "Market regime analysis, locus positioning, and signal quality",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.className} ${jetbrainsMono.variable} antialiased flex h-screen overflow-hidden bg-background text-foreground`}
      >
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden bg-[radial-gradient(ellipse_120%_80%_at_50%_-20%,rgba(34,211,238,0.08),transparent_55%)]">
          <AppHeader />
          <main className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 pb-16">
            {children}
          </main>
          <StatusBar />
        </div>
      </body>
    </html>
  );
}
