import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    // Vite >=5.4.12 rejects any request whose Host header is not an IP literal,
    // localhost, or listed here (DNS-rebinding protection). Reviewing the dev
    // server from another network goes over the tailnet, and
    // `scripts/ml-front-tailscale-serve.sh serve` forwards the MagicDNS name
    // through unchanged, so without this every such request gets
    // "Blocked request. This host is not allowed." The tailnet IP already
    // worked (IP literals are exempt); this is what makes the name work too.
    // Dev-server only — production serves the built SPA from FastAPI
    // StaticFiles (`backend/app/main.py` `_mount_front_dist`), which never
    // reads this config.
    allowedHosts: ['.ts.net'],
    proxy: {
      '/api/v1': {
        target: process.env.ML_API_PROXY_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    css: true,
    setupFiles: './src/test/setup.ts',
    exclude: ['node_modules/**', 'e2e/**', 'test-results/**'],
  },
});
