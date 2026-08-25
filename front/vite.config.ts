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
    // Run test files one at a time.
    //
    // Under the default parallel forks pool this suite fails at roughly 1% with a partially
    // materialized module namespace: a `vi.mock` factory's `vi.importActual('@/shared/api/client')`
    // returns a namespace that is missing real exports, surfacing as
    // `No "<export>" export is defined on the "@/shared/api/client" mock` or
    // `TypeError: <export> is not a function`. It reproduces on unmodified base source, so it is a
    // vite-node/Vitest 2.1.9 defect rather than anything this repo's tests spell wrongly, and every
    // attempt to fix it at the test seam only relocated the victim.
    //
    // `fileParallelism: false` is the single semantic setting that contains it: Vitest still uses
    // the forks pool with per-file isolation, it just stops running files concurrently. Do not add
    // redundant `pool`/`maxWorkers`/`minWorkers` bounds -- bounded-but-still-parallel worker counts
    // were measured and are not deterministic. Cost is roughly 2.4s -> 19-24s for the full suite.
    fileParallelism: false,
  },
});
