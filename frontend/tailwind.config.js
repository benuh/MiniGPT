/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        // Oxford-inspired color palette
        oxford: {
          blue: '#002147',
          'blue-light': '#1A3A5C',
          'blue-dark': '#001834',
          'blue-darker': '#001020',
          navy: '#172554',
          'navy-light': '#334B73',
          'navy-dark': '#0F1B3C',
          gold: '#B8860B',
          'gold-light': '#DAA520',
          'gold-dark': '#9A7209',
          charcoal: '#1F2937',
          'gray-warm': '#F8FAFC',
          'gray-cool': '#F1F5F9',
          ivory: '#FFFFFF'
        },
        primary: {
          50: '#F0F4FF',
          100: '#E0E7FF',
          200: '#C7D2FE',
          300: '#A5B4FC',
          400: '#818CF8',
          500: '#002147',
          600: '#001B3A',
          700: '#001630',
          800: '#001020',
          900: '#000A15',
        },
        gray: {
          50: '#FAFAFA',
          100: '#F5F5F5',
          200: '#E5E5E5',
          300: '#D4D4D4',
          400: '#A3A3A3',
          500: '#737373',
          600: '#525252',
          700: '#404040',
          800: '#262626',
          900: '#171717',
        },
        neutral: {
          50: '#FFFEF7',
          100: '#F7F7F7',
          200: '#F2F2F2',
          300: '#E8E8E8',
          400: '#D1D1D1',
          500: '#9A9A9A',
          600: '#6B6B6B',
          700: '#4A4A4A',
          800: '#2C2C2C',
          900: '#1C1C1C',
        }
      },
      fontFamily: {
        sans: ['Lato', 'Inter', 'system-ui', '-apple-system', 'sans-serif'],
        serif: ['Georgia', 'Times New Roman', 'serif'],
        mono: ['Menlo', 'Monaco', 'Courier New', 'monospace'],
      },
      letterSpacing: {
        'extra-wide': '0.1em',
        'ultra-wide': '0.2em',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}