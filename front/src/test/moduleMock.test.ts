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
