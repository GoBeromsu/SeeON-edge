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
 * `withOverrides` never copies the namespace. It forwards reads to the live namespace at access
 * time, by which point the module has finished evaluating, so a late-bound export resolves
 * normally. Overrides shadow the real export and are the only eagerly-held values.
 *
 * ## Why the Proxy target is a facade, not `actual`
 *
 * Proxy invariants are checked against the *target*. A real Vite SSR namespace is extensible with a
 * null prototype and carries one non-configurable own symbol:
 * `Symbol.toStringTag = "Module"` (`writable:false, enumerable:false, configurable:false`).
 * Using `actual` as the target therefore makes two operations illegal:
 *
 * 1. reporting that symbol as `configurable: true` -- `Object.getOwnPropertyDescriptor(proxy,
 *    Symbol.toStringTag)` and `Object.getOwnPropertyDescriptors(proxy)` throw
 *    `TypeError: ... incompatible with the existing property in the proxy target`;
 * 2. returning an override-only key from `ownKeys` when the target is non-extensible -- throws
 *    `TypeError: 'ownKeys' on proxy: trap returned extra keys but proxy target is non-extensible`.
 *
 * The target here is instead a fresh, extensible, null-prototype facade that starts empty, with
 * `actual` and the override layer captured in the closure. Invariants then bind to the facade, so
 * neither the namespace's non-configurable symbol nor a sealed `actual` can make reflection throw.
 * A non-configurable descriptor is mirrored onto the facade on first observation so the proxy can
 * report it *faithfully* rather than downgrading it; that is safe precisely because a
 * non-configurable, non-writable property can never subsequently change.
 *
 * Scope: this implements the mock semantics these factories need -- live reads, override
 * shadowing, and consistent enumeration/reflection. It is not a general module-namespace emulator;
 * `isExtensible` reports the facade (always extensible), which is what lets an override-only key
 * enumerate over a sealed `actual`.
 */
export function withOverrides<T extends object>(actual: T, overrides: Record<string, unknown>): T {
  // Private copy so nothing here mutates the caller's override object.
  const layer: Record<PropertyKey, unknown> = Object.create(null);
  for (const key of Reflect.ownKeys(overrides)) {
    layer[key as string] = (overrides as Record<PropertyKey, unknown>)[key as string];
  }
  const facade: Record<PropertyKey, unknown> = Object.create(null);
  const isOverride = (prop: PropertyKey): boolean => Object.prototype.hasOwnProperty.call(layer, prop);

  return new Proxy(facade, {
    get(_facade, prop) {
      // Reflect.get(actual, prop) without a receiver keeps live bindings resolving on the namespace.
      return isOverride(prop) ? layer[prop as string] : Reflect.get(actual, prop);
    },
    has(_facade, prop) {
      return isOverride(prop) || Reflect.has(actual, prop);
    },
    ownKeys() {
      return [...new Set([...Reflect.ownKeys(actual), ...Reflect.ownKeys(layer)])];
    },
    getOwnPropertyDescriptor(facadeTarget, prop) {
      if (isOverride(prop)) {
        const descriptor = Reflect.getOwnPropertyDescriptor(layer, prop);
        // The facade never holds override keys, so reporting them configurable is always legal.
        return descriptor === undefined ? undefined : { ...descriptor, configurable: true };
      }
      const descriptor = Reflect.getOwnPropertyDescriptor(actual, prop);
      if (descriptor === undefined) return undefined;
      if (descriptor.configurable === false) {
        // Mirror it so the target genuinely owns it and the faithful descriptor is reportable.
        if (!Object.prototype.hasOwnProperty.call(facadeTarget, prop)) {
          Object.defineProperty(facadeTarget, prop, descriptor);
        }
        return descriptor;
      }
      return { ...descriptor, configurable: true };
    },
    set(_facade, prop, value) {
      layer[prop as string] = value;
      return true;
    },
    defineProperty(_facade, prop, descriptor) {
      Object.defineProperty(layer, prop, descriptor);
      return true;
    },
    deleteProperty(facadeTarget, prop) {
      // A mirrored non-configurable property cannot be removed without breaking the invariant.
      if (Object.prototype.hasOwnProperty.call(facadeTarget, prop)) return false;
      delete layer[prop as string];
      return true;
    },
  }) as T;
}
