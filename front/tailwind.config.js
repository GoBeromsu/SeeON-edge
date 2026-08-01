/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // front(front/tailwind.config.js) 정렬 토큰 — CSS 변수 기반
        bg: 'var(--c-bg)',
        surface: 'var(--c-surface)',
        surface2: 'var(--c-surface-2)',
        border: 'var(--c-border)',
        ink: {
          DEFAULT: 'var(--c-ink)',
          soft: 'var(--c-ink-soft)',
          faint: 'var(--c-ink-faint)',
        },
        brand: {
          DEFAULT: 'var(--c-brand)',
          soft: 'var(--c-brand-soft)',
        },
        teal: 'var(--c-teal)',
        status: {
          stable: 'var(--c-stable)',
          stableBg: 'var(--c-stable-bg)',
          caution: 'var(--c-caution)',
          cautionBg: 'var(--c-caution-bg)',
          danger: 'var(--c-danger)',
          dangerBg: 'var(--c-danger-bg)',
          check: 'var(--c-check)',
          checkBg: 'var(--c-check-bg)',
        },
      },
      borderRadius: {
        '4xl': '2rem',
      },
    },
  },
  plugins: [],
};
