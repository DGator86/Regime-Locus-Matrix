import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin Turbopack root when multiple lockfiles exist (e.g. user home + this app)
  turbopack: {
    root: process.cwd(),
  },
};

export default nextConfig;
