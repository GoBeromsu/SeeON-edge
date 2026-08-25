import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

/**
 * Narrow recurrence guard for the two files that reach `@/shared/api/client` directly.
 *
 * `client.test.ts` once carried both a namespace and a named static import of the same module, and
 * `clientRetirement.test.ts` exists so the whole-namespace retirement check does not add a second
 * request to `client.test.ts`. This pins that split structurally -- exactly one runtime request in
 * each file and no extra dynamic request -- using the TypeScript AST rather than source prose.
 *
 * This is a recurrence/style guard, not the flake remedy: duplicate static imports transform into
 * sequential awaits and a later dynamic import runs after top-level evaluation, so neither is
 * concurrent. Determinism is owned by `fileParallelism: false` in `vite.config.ts`.
 */

const CLIENT_MODULE = '@/shared/api/client';
const SOURCE_ROOT = resolve(__dirname, '..');

function parse(file: string): ts.SourceFile {
  return ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true);
}

/** True when the declaration survives type erasure and so becomes a runtime module request. */
function isRuntimeImport(statement: ts.ImportDeclaration): boolean {
  const clause = statement.importClause;
  if (clause === undefined) return true;
  if (clause.isTypeOnly) return false;
  const bindings = clause.namedBindings;
  if (clause.name === undefined && bindings !== undefined && ts.isNamedImports(bindings)) {
    return !bindings.elements.every((element) => element.isTypeOnly);
  }
  return true;
}

function staticClientRequests(file: string): number {
  return parse(file).statements
    .filter(ts.isImportDeclaration)
    .filter(isRuntimeImport)
    .map((statement) => statement.moduleSpecifier)
    .filter(ts.isStringLiteral)
    .filter((literal) => literal.text === CLIENT_MODULE)
    .length;
}

function dynamicClientRequests(file: string): number {
  let count = 0;
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node)
      && node.expression.kind === ts.SyntaxKind.ImportKeyword
      && node.arguments.length === 1) {
      const [specifier] = node.arguments;
      if (specifier !== undefined && ts.isStringLiteral(specifier) && specifier.text === CLIENT_MODULE) count += 1;
    }
    ts.forEachChild(node, visit);
  };
  visit(parse(file));
  return count;
}

describe('client module import structure', () => {
  it.each([
    ['shared/api/client.test.ts'],
    ['shared/api/clientRetirement.test.ts'],
  ])('requests the client module exactly once in %s, with no extra dynamic request', (relativePath) => {
    const file = resolve(SOURCE_ROOT, relativePath);

    expect(staticClientRequests(file)).toBe(1);
    expect(dynamicClientRequests(file)).toBe(0);
  });
});
