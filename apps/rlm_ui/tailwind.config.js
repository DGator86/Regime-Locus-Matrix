/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#060b14',
          900: '#0a1020',
          800: '#0f1930',
          700: '#162040',
          600: '#1e2d52',
          500: '#263760',
        },
        cyan: {
          DEFAULT: '#00f5ff',
          dim: '#00b8c4',
          glow: 'rgba(0,245,255,0.15)',
        },
        green: {
          trade: '#00ff9d',
          dim: '#00cc7a',
        },
        red: {
          trade: '#ff3355',
          dim: '#cc2244',
        },
        amber: {
          trade: '#ffaa00',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        card: '0 0 0 1px rgba(0,245,255,0.08), 0 4px 24px rgba(0,0,0,0.4)',
        glow: '0 0 20px rgba(0,245,255,0.2)',
        'glow-green': '0 0 20px rgba(0,255,157,0.25)',
        'glow-red': '0 0 20px rgba(255,51,85,0.25)',
      },
    },
  },
  plugins: [],
}
