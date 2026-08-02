import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

function readCssBundle(entry: string): string {
  const visited = new Set<string>();

  function readWithImports(path: string): string {
    const absolutePath = resolve(path);
    if (visited.has(absolutePath)) return '';
    visited.add(absolutePath);

    const source = readFileSync(absolutePath, 'utf8');
    const imports = [...source.matchAll(/@import\s+["']([^"']+)["']\s*;/g)]
      .map((match) => readWithImports(resolve(dirname(absolutePath), match[1])));
    return [...imports, source].join('\n');
  }

  return readWithImports(entry);
}

const css = readCssBundle('src/styles.css');
const designContract = readFileSync('../DESIGN.md', 'utf8');
const appReachableComponentPaths = [
  'src/app/App.tsx',
  'src/features/camera-management/CameraManagementPage.tsx',
  'src/features/camera-management/CameraManagementPanel.tsx',
  'src/features/cameras/AddCameraModal.tsx',
  'src/features/cameras/CameraCard.tsx',
  'src/features/cameras/DeleteCameraDialog.tsx',
  'src/features/clips/ClipLabelButtons.tsx',
  'src/features/connection/ConnectionSettingsPanel.tsx',
  'src/features/events/EventHistoryPage.tsx',
  'src/features/operations/OperationsView.tsx',
  'src/features/system/SystemPage.tsx',
  'src/shared/ui/AccessibleDialog.tsx',
  'src/shared/ui/AuthGate.tsx',
  'src/shared/ui/DashboardShell.tsx',
  'src/shared/ui/SeniorAiLabBrand.tsx',
  'src/shared/ui/StatusBadge.tsx',
  'src/shared/ui/SystemPanels.tsx',
] as const;
const appReachableComponentSources = Object.fromEntries(
  appReachableComponentPaths.map((path) => [path, readFileSync(path, 'utf8')]),
);
const eventHistoryPage = appReachableComponentSources['src/features/events/EventHistoryPage.tsx'];

function cssColor(variable: string): string {
  const value = css.match(new RegExp(`${variable}:\\s*(#[0-9a-f]{6})`, 'i'))?.[1];
  if (!value) throw new Error(`Missing CSS color token: ${variable}`);
  return value;
}

function documentedColor(variable: string): string {
  const value = designContract.match(new RegExp(`\`${variable}\`\\s*\\|\\s*\`(#[0-9a-f]{6})\``, 'i'))?.[1];
  if (!value) throw new Error(`Missing documented color token: ${variable}`);
  return value;
}

function relativeLuminance(hex: string): number {
  const channels = hex.slice(1).match(/.{2}/g)?.map((channel) => Number.parseInt(channel, 16) / 255);
  if (!channels || channels.length !== 3) throw new Error(`Invalid hex color: ${hex}`);
  const [red, green, blue] = channels.map((channel) => (
    channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4
  ));
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

function contrastRatio(foreground: string, background: string): number {
  const lighter = Math.max(relativeLuminance(foreground), relativeLuminance(background));
  const darker = Math.min(relativeLuminance(foreground), relativeLuminance(background));
  return (lighter + 0.05) / (darker + 0.05);
}

describe('design token contrast', () => {
  it('keeps faint normal text readable on both dashboard light surfaces', () => {
    const faint = cssColor('--c-ink-faint');

    expect(documentedColor('--c-ink-faint').toLowerCase()).toBe(faint.toLowerCase());
    expect(contrastRatio(faint, cssColor('--c-surface'))).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(faint, cssColor('--c-bg'))).toBeGreaterThanOrEqual(4.5);
  });

  it('keeps media overlay text readable on dark media frames', () => {
    expect(documentedColor('--c-media-text').toLowerCase()).toBe(cssColor('--c-media-text').toLowerCase());
    expect(contrastRatio(cssColor('--c-media-text'), cssColor('--c-media'))).toBeGreaterThanOrEqual(4.5);
  });

  it('uses a quiet single-column authentication card without a marketing brand panel', () => {
    expect(css).toMatch(/\.auth-card\s*\{[^}]*width:\s*min\(420px,\s*100%\)/s);
    expect(css).toMatch(/\.auth-card\s*\{[^}]*min-height:\s*440px/s);
    expect(css).not.toMatch(/\.auth-card\s*\{[^}]*box-shadow:/s);
    expect(css).not.toMatch(/\.auth-brand-panel\s*\{/);
    expect(css).not.toMatch(/\.auth-card\s*\{[^}]*grid-template-columns/s);
  });

  it('keeps reachable event media surfaces on semantic media tokens', () => {
    expect(eventHistoryPage).not.toMatch(/\b(?:bg-black|text-white)(?:\/\d+)?\b/);
    expect(eventHistoryPage).toContain('className="event-media-frame"');
    expect(eventHistoryPage).toContain('event-media-unavailable');
    expect(css).toMatch(/\.event-media-frame\s*\{[^}]*background:\s*var\(--c-media\);/s);
    expect(css).toMatch(/\.event-media-unavailable\s*\{[^}]*color:\s*var\(--c-media-text\);/s);
  });

  it.each(Object.entries(appReachableComponentSources))('keeps App-reachable %s free of raw black/white utility aliases', (_name, source) => {
    expect(source).not.toMatch(/\b(?:bg-black|bg-white|text-black|text-white)(?:\/\d+)?\b/);
  });

  it.each(Object.entries(appReachableComponentSources))('keeps App-reachable %s actions on semantic fills and foregrounds', (_name, source) => {
    const buttonClasses = [...source.matchAll(/<button\b(?:(?!<\/button>)[\s\S])*?\bclassName="([^"]*)"/g)]
      .map((match) => match[1]);
    const forbiddenActionAliases = new Set(['bg-brand', 'hover:bg-brand', 'bg-ink', 'text-white']);
    const rawAliases = buttonClasses
      .flatMap((className) => className.split(/\s+/))
      .filter((className) => forbiddenActionAliases.has(className));

    expect(rawAliases).toEqual([]);
  });

  it('keeps reachable stylesheet declarations free of raw black/white and overlay values', () => {
    const declarations = css.replace(/^\s*--c-[\w-]+:\s*[^;]+;\s*$/gm, '');

    expect(declarations).not.toMatch(/\b(?:color|background(?:-color)?)\s*:\s*(?:black|white|#(?:000|000000|fff|ffffff))\b/i);
    expect(declarations).not.toMatch(/\bbackground(?:-color)?\s*:\s*(?:rgba?\(\s*(?:0\s*,\s*){2}0|color-mix\([^;]*\bblack\b)/i);
  });

  it('keeps the raw brand fill declaration inside the semantic action class only', () => {
    const withoutSemanticAction = css.replace(/\.brand-action\s*\{[^}]*\}/s, '');

    expect(withoutSemanticAction).not.toMatch(/\bbackground(?:-color)?\s*:\s*var\(--c-brand\)/);
  });

  it('documents readable semantic foreground tokens for actions and media overlays', () => {
    expect(documentedColor('--c-action-foreground').toLowerCase()).toBe(cssColor('--c-action-foreground').toLowerCase());
    expect(documentedColor('--c-media-overlay-foreground').toLowerCase()).toBe(cssColor('--c-media-overlay-foreground').toLowerCase());
    expect(contrastRatio(cssColor('--c-action-foreground'), cssColor('--c-brand'))).toBeGreaterThanOrEqual(4.5);
    expect(css).toMatch(/--c-media-overlay:\s*rgba\(13,\s*13,\s*13,\s*0\.72\);/);
    expect(css).toMatch(/\.brand-action\s*\{[^}]*background:\s*var\(--c-brand\);[^}]*color:\s*var\(--c-action-foreground\);/s);
    expect(css).toMatch(/\.media-status-overlay\s*\{[^}]*background:\s*var\(--c-media-overlay\);[^}]*color:\s*var\(--c-media-overlay-foreground\);/s);
  });
});
