/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // front(front/tailwind.config.js) 정렬 토큰 — CSS 변수 기반. 이름 체계는
        // front/design-handoff/README.md의 디자인 토큰 스펙을 따른다.
        background: 'var(--background)',
        foreground: 'var(--foreground)',
        card: {
          DEFAULT: 'var(--card)',
          foreground: 'var(--card-foreground)',
        },
        border: 'var(--border)',
        input: 'var(--input)',
        muted: {
          DEFAULT: 'var(--muted)',
          foreground: 'var(--muted-foreground)',
        },
        primary: {
          DEFAULT: 'var(--primary)',
          foreground: 'var(--primary-foreground)',
        },
        destructive: {
          DEFAULT: 'var(--destructive)',
          foreground: 'var(--destructive-foreground)',
        },
        teal: 'var(--overlay-teal)',
        status: {
          approvedBg: 'var(--status-approved-bg)',
          approvedFg: 'var(--status-approved-fg)',
          rejectedBg: 'var(--status-rejected-bg)',
          rejectedFg: 'var(--status-rejected-fg)',
          pendingBg: 'var(--status-pending-bg)',
          pendingFg: 'var(--status-pending-fg)',
          closedBg: 'var(--status-closed-bg)',
          closedFg: 'var(--status-closed-fg)',
        },
      },
      borderRadius: {
        card: 'var(--radius-card)',
        control: 'var(--radius-control)',
      },
      boxShadow: {
        modal: 'var(--modal-shadow)',
      },
    },
  },
  plugins: [],
};
