/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0A0A0F',
        surface: '#111115',
        ink: '#F5F5F7',
        accent: '#00FF9C',
      },
      fontFamily: {
        mono: ['SF Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};
