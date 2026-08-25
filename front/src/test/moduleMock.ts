/**
 * Deterministic seam for `vi.mock` factories that need "the real module, with a few overrides".
 *
 * The naive spelling is `{ ...(await vi.importActual(path)), ...overrides }`, and it is racy.
 * Vite's SSR transform does not hoist export definitions: a module body opens with top-level
 * `await __vite_ssr_import__(...)` calls and then installs each export with a separate
 * `Object.defineProperty(__vite_ssr_exports__, ...)` scattered through the body (for
 * `shared/api/client.ts`: line 11, 99, 254 of 305). A factory that observes that namespace while it
 * is still materializing spreads a *partial* object, and the spread freezes the gap permanently --
 * the mock is then missing real exports for the rest of the file, surfacing as
 * `No "<name>" export is defined on the "<path>" mock`.
 *
 * `withOverrides` never copies the namespace. It returns a Proxy that forwards every read to the
 * live namespace at access time, by which point the module has finished evaluating, so a late-bound
 * export resolves normally. Overrides shadow the real export and are the only eagerly-held values.
 *
 * This changes no test's behaviour: reads that used to hit a copied value now hit the same live
 * binding, and `Object.keys`/`in` still report the union of real exports and overrides.
 */
export function withOverrides<T extends object>(actual: T, overrides: Record<string, unknown>): T {
  const owns = (prop: PropertyKey): boolean => Object.prototype.hasOwnProperty.call(overrides, prop);
  return new Proxy(actual, {
    get(target, prop, receiver) {
      return owns(prop) ? overrides[prop as string] : Reflect.get(target, prop, receiver);
    },
    has(target, prop) {
      return owns(prop) || Reflect.has(target, prop);
    },
    ownKeys(target) {
      return [...new Set([...Reflect.ownKeys(target), ...Reflect.ownKeys(overrides)])];
    },
    getOwnPropertyDescriptor(target, prop) {
      if (owns(prop)) {
        return { value: overrides[prop as string], enumerable: true, configurable: true, writable: true };
      }
      const descriptor = Reflect.getOwnPropertyDescriptor(target, prop);
      // Proxy invariants forbid reporting a non-configurable descriptor for a target whose shape
      // can still change while the module finishes evaluating.
      return descriptor === undefined ? undefined : { ...descriptor, configurable: true };
    },
  }) as T;
}
