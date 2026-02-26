import type { Config } from 'tailwindcss'
import typography from '@tailwindcss/typography'

const config: Config = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        apple: {
          bg: '#F5F5F7',
          text: '#1D1D1F',
          secondary: '#86868B',
          blue: '#007AFF',
        },
      },
      boxShadow: {
        'apple-sm': '0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04)',
        'apple': '0 2px 12px rgba(0, 0, 0, 0.08)',
        'apple-hover': '0 8px 30px rgba(0, 0, 0, 0.12)',
      },
    },
  },
  plugins: [typography],
}
export default config
