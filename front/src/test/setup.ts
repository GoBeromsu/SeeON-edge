import { withOverrides } from '@/test/moduleMock';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

HTMLMediaElement.prototype.play = () => Promise.resolve();

/**
 * `vi.mock` factories are hoisted above imports, so they cannot use a top-level import of the
 * shared partial-mock seam. Resolving it inside each factory with `vi.importActual` was the
 * alternative, but that turns the helper into a fan-in point with the very hazard it exists to
 * prevent: 18 factories racing to observe one namespace produced `TypeError: withOverrides is not
 * a function` at ~1%. Vitest fully evaluates `setupFiles` before any test module is collected, so
 * publishing the seam here gives every factory a synchronous, already-materialized reference with
 * no module request at all.
 */
declare global {
  // eslint-disable-next-line no-var
  var withOverrides: <T extends object>(actual: T, overrides: Record<string, unknown>) => T;
}

globalThis.withOverrides = withOverrides;
