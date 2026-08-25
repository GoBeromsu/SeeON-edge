import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

/**
 * Structural recurrence guard for the partial-namespace defect.
 *
 * Vite's SSR transform emits one `__vite_ssr_import__` request per top-level runtime import
 * declaration. Two such declarations for the *same* local module therefore issue two concurrent
 * requests, and the second can observe the namespace while the first is still evaluating it --
 * exactly how `client.test.ts` once produced `TypeError: <export> is not a function` at ~1%.
 * A duplicate import is structurally invisible to the test suite (the duplicate-import mutant
 * passed 100/100), so it needs a structural guard rather than a runtime one.
 *
 * A *dynamic* `import()` of a module the same file already imports statically is the same hazard
 * measured from the other side: it was observed failing at ~0.5%. The rule enforced here is
 * therefore one request per module per file, counting static and dynamic together. The retirement
 * guard that needs the whole namespace lives in its own file (`clientRetirement.test.ts`).
 *
 * This parses real TypeScript with the compiler AST -- no regex, no source-prose or formatting
 * pins. Type-only declarations are excluded because they are erased and issue no request.
 */

const SOURCE_ROOT = resolve(__dirname, '..');
const CLIENT_MODULE = '@/shared/api/client';

function parse(file: string): ts.SourceFile {
  return ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true);
}

/** True when the declaration survives type erasure and so becomes a runtime module request. */
function isRuntimeImport(statement: ts.ImportDeclaration): boolean {
  const clause = statement.importClause;
  if (clause === undefined) return true; // bare side-effect import
  if (clause.isTypeOnly) return false;
  const bindings = clause.namedBindings;
  if (clause.name === undefined && bindings !== undefined && ts.isNamedImports(bindings)) {
    // `import { type A, type B } from 'x'` is fully erased.
    return !bindings.elements.every((element) => element.isTypeOnly);
  }
  return true;
}

/** Local specifiers are SSR-transformed per module; bare specifiers are externalized deps. */
function isLocalSpecifier(specifier: string): boolean {
  return specifier.startsWith('@/') || specifier.startsWith('./') || specifier.startsWith('../');
}

function runtimeImportSpecifiers(file: string): string[] {
  return parse(file).statements
    .filter(ts.isImportDeclaration)
    .filter(isRuntimeImport)
    .map((statement) => statement.moduleSpecifier)
    .filter(ts.isStringLiteral)
    .map((literal) => literal.text);
}

function dynamicSpecifiers(file: string): string[] {
  const found: string[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node)
      && node.expression.kind === ts.SyntaxKind.ImportKeyword
      && node.arguments.length === 1) {
      const [specifier] = node.arguments;
      if (specifier !== undefined && ts.isStringLiteral(specifier)) found.push(specifier.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(parse(file));
  return found;
}

function sourceFiles(directory: string): string[] {
  return readdirSync(directory).flatMap((entry) => {
    const full = join(directory, entry);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return /\.tsx?$/.test(entry) ? [full] : [];
  });
}

function duplicates(specifiers: readonly string[]): string[] {
  const seen = new Map<string, number>();
  for (const specifier of specifiers.filter(isLocalSpecifier)) {
    seen.set(specifier, (seen.get(specifier) ?? 0) + 1);
  }
  return [...seen.entries()].filter(([, count]) => count > 1).map(([specifier]) => specifier);
}

describe('module import structure', () => {
  it('imports the client module exactly once at top level in client.test.ts', () => {
    const file = resolve(SOURCE_ROOT, 'shared/api/client.test.ts');

    const clientImports = runtimeImportSpecifiers(file).filter((s) => s === CLIENT_MODULE);

    expect(clientImports).toHaveLength(1);
  });

  it('resolves the retirement namespace from its own file, with exactly one request each', () => {
    const clientTest = resolve(SOURCE_ROOT, 'shared/api/client.test.ts');
    const retirementTest = resolve(SOURCE_ROOT, 'shared/api/clientRetirement.test.ts');

    // No file may reach the module twice, counting static and dynamic requests together.
    expect(dynamicSpecifiers(clientTest).filter((s) => s === CLIENT_MODULE)).toHaveLength(0);
    expect(runtimeImportSpecifiers(retirementTest).filter((s) => s === CLIENT_MODULE)).toHaveLength(1);
    expect(dynamicSpecifiers(retirementTest).filter((s) => s === CLIENT_MODULE)).toHaveLength(0);
  });

  it('has no file requesting the same local module both statically and dynamically', () => {
    const offenders = sourceFiles(SOURCE_ROOT)
      .map((file) => {
        const statics = new Set(runtimeImportSpecifiers(file).filter(isLocalSpecifier));
        const repeated = dynamicSpecifiers(file).filter((s) => isLocalSpecifier(s) && statics.has(s));
        return { file: relative(SOURCE_ROOT, file), repeated };
      })
      .filter((entry) => entry.repeated.length > 0);

    expect(offenders).toEqual([]);
  });

  it('has no file importing the same local module twice at top level', () => {
    const offenders = sourceFiles(SOURCE_ROOT)
      .map((file) => ({ file: relative(SOURCE_ROOT, file), repeated: duplicates(runtimeImportSpecifiers(file)) }))
      .filter((entry) => entry.repeated.length > 0);

    expect(offenders).toEqual([]);
  });
});
