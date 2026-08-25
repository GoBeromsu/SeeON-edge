import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

/**
 * Machine-consumed guard for the one setting that makes `pnpm --dir front test --run` deterministic.
 *
 * Under the default parallel forks pool this suite fails at roughly 1% because a `vi.mock` factory's
 * `vi.importActual('@/shared/api/client')` can observe a partially materialized namespace. It
 * reproduces on unmodified base source. `fileParallelism: false` is the containment; if it is
 * removed or weakened the suite silently becomes flaky again, so the setting is asserted here from
 * the config AST (not from source prose) and from the observed runtime worker identity.
 */
describe('vitest determinism configuration', () => {
  it('declares fileParallelism: false in vite.config.ts', () => {
    const config = resolve(__dirname, '../../vite.config.ts');
    const source = ts.createSourceFile(config, readFileSync(config, 'utf8'), ts.ScriptTarget.Latest, true);

    let fileParallelism: boolean | undefined;
    const visit = (node: ts.Node): void => {
      if (ts.isPropertyAssignment(node)
        && ts.isIdentifier(node.name)
        && node.name.text === 'fileParallelism') {
        if (node.initializer.kind === ts.SyntaxKind.FalseKeyword) fileParallelism = false;
        if (node.initializer.kind === ts.SyntaxKind.TrueKeyword) fileParallelism = true;
      }
      ts.forEachChild(node, visit);
    };
    visit(source);

    expect(fileParallelism).toBe(false);
  });

  it('does not pin redundant pool or worker bounds alongside it', () => {
    const config = resolve(__dirname, '../../vite.config.ts');
    const source = ts.createSourceFile(config, readFileSync(config, 'utf8'), ts.ScriptTarget.Latest, true);

    const pinned: string[] = [];
    const visit = (node: ts.Node): void => {
      if (ts.isPropertyAssignment(node) && ts.isIdentifier(node.name)
        && ['pool', 'maxWorkers', 'minWorkers', 'poolOptions'].includes(node.name.text)) {
        pinned.push(node.name.text);
      }
      ts.forEachChild(node, visit);
    };
    visit(source);

    expect(pinned).toEqual([]);
  });

  it('runs in a forked child worker on a single pool lane', async () => {
    // forks pool => a child process with an IPC channel, not a worker thread. (A threads pool
    // would report isMainThread === false and expose no process.send.)
    const { isMainThread } = await import('node:worker_threads');
    expect(typeof process.send).toBe('function');
    expect(isMainThread).toBe(true);
    // One lane: VITEST_WORKER_ID still increments per isolated file, but POOL_ID stays 1 because
    // no second file runs concurrently.
    expect(process.env.VITEST_POOL_ID).toBe('1');
  });
});
