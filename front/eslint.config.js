// Boundary rules only. Encodes the import contract in src/AGENTS.md and
// src/features/AGENTS.md: shared never depends upward, features never reach
// into sibling slices except the two frozen seams, and nothing in the SPA
// imports backend/ or worker/. No stylistic rules live here on purpose.
import tsParser from '@typescript-eslint/parser';

const FEATURES = ['account-settings', 'cameras', 'connection', 'events', 'operations', 'settings'];

// The two documented one-way seams. Do not add a third; lift to shared/ instead.
const ALLOWED_SEAMS = {
  connection: ['cameras'],
  operations: ['settings'],
};

const NO_SERVICE_CODE = {
  group: ['backend', 'backend/**', '**/backend/**', 'worker', 'worker/**', '**/worker/**'],
  message: 'front/src must not import backend/ or worker/ code. Talk to them over HTTP via @/shared/api.',
};

function featureRules(slice) {
  const allowed = new Set([slice, ...(ALLOWED_SEAMS[slice] ?? [])]);
  const forbidden = FEATURES.filter((s) => !allowed.has(s));
  return {
    files: [`src/features/${slice}/**`],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            NO_SERVICE_CODE,
            {
              group: forbidden.flatMap((s) => [`@/features/${s}`, `@/features/${s}/**`]),
              message:
                `features/${slice} may only import its own slice` +
                (ALLOWED_SEAMS[slice] ? ` and the frozen seam(s) ${ALLOWED_SEAMS[slice].join(', ')}` : '') +
                '. See src/features/AGENTS.md "Frozen crossings"; lift shared widgets to @/shared.',
            },
            {
              group: ['../**'],
              message: 'Feature slices are flat: `../` leaves the slice. Use `@/` imports (src/AGENTS.md).',
            },
            {
              group: ['@/app/App', '@/app/App.tsx'],
              message: 'Features must not import App. Use @/app/dashboardLocation via your use*Location hook.',
            },
          ],
        },
      ],
    },
  };
}

export default [
  {
    files: ['src/**/*.{ts,tsx}'],
    languageOptions: {
      parser: tsParser,
      parserOptions: { ecmaVersion: 'latest', sourceType: 'module', ecmaFeatures: { jsx: true } },
    },
    rules: {
      'no-restricted-imports': ['error', { patterns: [NO_SERVICE_CODE] }],
    },
  },
  {
    // shared/ is the bottom layer: never @/features, never @/app.
    files: ['src/shared/**'],
    ignores: ['src/shared/ui/NavBar.tsx'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            NO_SERVICE_CODE,
            {
              group: ['@/features/**', '@/app/**'],
              message: 'shared/ must not depend upward on @/features or @/app (src/AGENTS.md "Imports").',
            },
          ],
        },
      ],
    },
  },
  {
    // Documented exception: NavBar types against @/app/dashboardLocation. Don't spread it.
    files: ['src/shared/ui/NavBar.tsx'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            NO_SERVICE_CODE,
            { group: ['@/features/**'], message: 'shared/ must not import @/features (src/AGENTS.md).' },
          ],
        },
      ],
    },
  },
  ...FEATURES.map(featureRules),
];
