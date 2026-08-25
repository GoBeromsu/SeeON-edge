import { describe, expect, it, vi } from 'vitest';
import { withOverrides } from '@/test/moduleMock';

/**
 * Models the exact Vitest failure mechanism without depending on scheduling luck.
 *
 * Vite's SSR transform installs exports with `Object.defineProperty` calls spread through a module
 * body that begins with top-level `await`s. A factory that observes the namespace mid-evaluation
 * sees only the exports defined so far. `lateBindingNamespace` reproduces that deterministically:
 * `earlyExport` is present immediately, `lateExport` appears one microtask later.
 */
function lateBindingNamespace(): Record<string, unknown> {
  const namespace: Record<string, unknown> = {};
  Object.defineProperty(namespace, 'earlyExport', {
    get: () => () => 'early', enumerable: true, configurable: true,
  });
  void Promise.resolve().then(() => {
    Object.defineProperty(namespace, 'lateExport', {
      get: () => () => 'late', enumerable: true, configurable: true,
    });
  });
  return namespace;
}

async function settleModuleEvaluation(): Promise<void> {
  await Promise.resolve();
}

describe('partial-mock seam against a still-materializing module namespace', () => {
  it('REGRESSION GUARD: the eager spread freezes the gap and loses a late-bound export', async () => {
    const namespace = lateBindingNamespace();

    // This is the base-owned `{ ...actual, ...overrides }` spelling.
    const eager = { ...namespace, overridden: vi.fn() };
    await settleModuleEvaluation();

    // The real namespace completed, but the copy did not follow: this is the shipped failure.
    expect('lateExport' in namespace).toBe(true);
    expect('lateExport' in eager).toBe(false);
    expect(Object.keys(eager)).not.toContain('lateExport');
  });

  it('withOverrides resolves a late-bound export because it never copies the namespace', async () => {
    const namespace = lateBindingNamespace();

    const lazy = withOverrides(namespace, { overridden: vi.fn() });
    await settleModuleEvaluation();

    expect('lateExport' in lazy).toBe(true);
    expect(Object.keys(lazy)).toContain('lateExport');
    expect((lazy.lateExport as () => string)()).toBe('late');
  });

  it('exposes real exports, applies overrides, and reports the union of keys', () => {
    const namespace = lateBindingNamespace();
    const overridden = vi.fn(() => 'mocked');

    const lazy = withOverrides(namespace, { overridden, earlyExport: overridden });

    expect((lazy.earlyExport as () => string)()).toBe('mocked');
    expect((lazy.overridden as () => string)()).toBe('mocked');
    expect(Object.keys(lazy).sort()).toEqual(['earlyExport', 'overridden']);
    expect('earlyExport' in lazy).toBe(true);
    expect('missingExport' in lazy).toBe(false);
  });

  it('keeps a non-overridden real export live rather than snapshotting its value', () => {
    let current = 'first';
    const namespace: Record<string, unknown> = {};
    Object.defineProperty(namespace, 'live', { get: () => current, enumerable: true, configurable: true });

    const lazy = withOverrides(namespace, {});
    expect(lazy.live).toBe('first');

    current = 'second';
    expect(lazy.live).toBe('second');
  });
});

/**
 * Proxy-invariant contract. A real Vite SSR namespace is extensible with a null prototype and one
 * non-configurable own symbol (`Symbol.toStringTag` = "Module", writable:false, enumerable:false).
 * A trap may not report such a property as configurable, and may not report an override-only key
 * that is absent from a non-extensible target. Both must hold for reflection to work at all.
 */
describe('withOverrides preserves Proxy invariants against a real module namespace', () => {
  it('reflects Symbol.toStringTag from an actual vi.importActual namespace without throwing', async () => {
    const actual = await vi.importActual<Record<string, unknown>>('@/shared/api/client');
    const proxy = withOverrides(actual, { deleteClip: vi.fn() });

    const descriptor = Object.getOwnPropertyDescriptor(proxy, Symbol.toStringTag);
    expect(descriptor).toEqual({
      value: 'Module', writable: false, enumerable: false, configurable: false,
    });
    expect(proxy[Symbol.toStringTag as unknown as string]).toBe('Module');
    expect(Object.prototype.toString.call(proxy)).toBe('[object Module]');
  });

  it('reflects every descriptor of an actual namespace via getOwnPropertyDescriptors', async () => {
    const actual = await vi.importActual<Record<string, unknown>>('@/shared/api/client');
    const override = vi.fn();
    const proxy = withOverrides(actual, { deleteClip: override, overrideOnly: override });

    const descriptors = Object.getOwnPropertyDescriptors(proxy);

    expect(Object.keys(descriptors).length).toBeGreaterThan(0);
    expect(descriptors[Symbol.toStringTag as unknown as string]).toEqual({
      value: 'Module', writable: false, enumerable: false, configurable: false,
    });
    // Real export still reflected, override shadows it, override-only key is present.
    expect(Reflect.ownKeys(proxy)).toContain('fetchClipArtifacts');
    expect((descriptors as Record<string, PropertyDescriptor>).overrideOnly?.value).toBe(override);
    expect(proxy.deleteClip).toBe(override);
    expect(typeof proxy.fetchClipArtifacts).toBe('function');
  });

  it('enumerates an actual namespace consistently across keys, ownKeys and `in`', async () => {
    const actual = await vi.importActual<Record<string, unknown>>('@/shared/api/client');
    const proxy = withOverrides(actual, { overrideOnly: vi.fn() });

    expect(Object.keys(proxy)).toContain('fetchClipArtifacts');
    expect(Object.keys(proxy)).toContain('overrideOnly');
    // Symbol.toStringTag is non-enumerable, so it must appear in ownKeys but not in Object.keys.
    expect(Reflect.ownKeys(proxy)).toContain(Symbol.toStringTag);
    expect(Object.keys(proxy)).not.toContain('Symbol(Symbol.toStringTag)');
    expect('fetchClipArtifacts' in proxy).toBe(true);
    expect('overrideOnly' in proxy).toBe(true);
    expect('definitelyMissingExport' in proxy).toBe(false);
    expect(Object.getOwnPropertyDescriptor(proxy, 'definitelyMissingExport')).toBeUndefined();
  });

  it('handles a non-extensible, null-prototype actual carrying a non-configurable symbol', () => {
    const marker = Symbol('marker');
    const actual: Record<PropertyKey, unknown> = Object.create(null);
    Object.defineProperty(actual, Symbol.toStringTag, {
      value: 'Module', writable: false, enumerable: false, configurable: false,
    });
    Object.defineProperty(actual, marker, {
      value: 'sealed', writable: false, enumerable: true, configurable: false,
    });
    Object.defineProperty(actual, 'realExport', {
      value: () => 'real', writable: false, enumerable: true, configurable: false,
    });
    Object.preventExtensions(actual);
    expect(Object.isExtensible(actual)).toBe(false);

    const overrideOnly = vi.fn();
    const proxy = withOverrides(actual, { overrideOnly });

    // Override-only key must still enumerate even though the actual object is sealed.
    expect(Reflect.ownKeys(proxy)).toContain('overrideOnly');
    expect(Reflect.ownKeys(proxy)).toContain(marker);
    expect(Object.getOwnPropertyDescriptor(proxy, Symbol.toStringTag)).toEqual({
      value: 'Module', writable: false, enumerable: false, configurable: false,
    });
    expect(Object.getOwnPropertyDescriptor(proxy, marker)).toEqual({
      value: 'sealed', writable: false, enumerable: true, configurable: false,
    });
    expect(() => Object.getOwnPropertyDescriptors(proxy)).not.toThrow();
    expect(proxy.overrideOnly).toBe(overrideOnly);
    expect((proxy.realExport as () => string)()).toBe('real');
  });

  it('does not mutate the actual namespace or the caller overrides', async () => {
    const actual = await vi.importActual<Record<string, unknown>>('@/shared/api/client');
    const before = Reflect.ownKeys(actual).map(String).sort();
    const overrides = { deleteClip: vi.fn() };
    const overrideKeysBefore = Reflect.ownKeys(overrides).map(String).sort();

    const proxy = withOverrides(actual, overrides);
    void Object.getOwnPropertyDescriptors(proxy);
    void Reflect.ownKeys(proxy);

    expect(Reflect.ownKeys(actual).map(String).sort()).toEqual(before);
    expect(Reflect.ownKeys(overrides).map(String).sort()).toEqual(overrideKeysBefore);
  });
});

describe('withOverrides property semantics', () => {
  it('shadows a real export with an override and leaves the namespace value reachable underneath', async () => {
    const actual = await vi.importActual<Record<string, unknown>>('@/shared/api/client');
    const real = actual.fetchClipArtifacts;
    const override = vi.fn();

    const proxy = withOverrides(actual, { fetchClipArtifacts: override });

    expect(proxy.fetchClipArtifacts).toBe(override);
    expect(actual.fetchClipArtifacts).toBe(real);
    expect(Reflect.ownKeys(proxy).filter((k) => k === 'fetchClipArtifacts')).toHaveLength(1);
  });

  it('supports a symbol override alongside the namespace symbol', () => {
    const custom = Symbol('custom');
    const actual: Record<PropertyKey, unknown> = Object.create(null);
    Object.defineProperty(actual, Symbol.toStringTag, {
      value: 'Module', writable: false, enumerable: false, configurable: false,
    });
    const value = vi.fn();

    const proxy = withOverrides(actual, { [custom]: value } as Record<string, unknown>);

    expect(proxy[custom as unknown as string]).toBe(value);
    expect(Reflect.ownKeys(proxy)).toContain(custom);
    expect(Reflect.ownKeys(proxy)).toContain(Symbol.toStringTag);
    expect(Object.getOwnPropertyDescriptor(proxy, Symbol.toStringTag)?.configurable).toBe(false);
  });

  it('keeps a getter-backed export live and does not snapshot it into a descriptor read', () => {
    let current = 'first';
    const actual: Record<string, unknown> = Object.create(null);
    Object.defineProperty(actual, 'live', { get: () => current, enumerable: true, configurable: true });

    const proxy = withOverrides(actual, {});
    expect(proxy.live).toBe('first');
    void Object.getOwnPropertyDescriptor(proxy, 'live');

    current = 'second';
    expect(proxy.live).toBe('second');
  });

  it('reports an absent key as absent through every reflection path', () => {
    const actual: Record<string, unknown> = Object.create(null);
    const proxy = withOverrides(actual, { present: vi.fn() });

    expect('absent' in proxy).toBe(false);
    expect(proxy.absent).toBeUndefined();
    expect(Object.getOwnPropertyDescriptor(proxy, 'absent')).toBeUndefined();
    expect(Reflect.ownKeys(proxy)).not.toContain('absent');
    expect(Reflect.ownKeys(proxy)).toContain('present');
  });
});
