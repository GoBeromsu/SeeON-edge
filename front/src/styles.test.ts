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
const responsiveCss = readFileSync('src/styles/responsive-layout.css', 'utf8');
const designContract = readFileSync('../DESIGN.md', 'utf8');

// App.tsx-reachable components that were restyled onto the current token set (front/src/styles/tokens-base.css)
// as part of the shell redesign.
const appReachableComponentPaths = [
  'src/app/App.tsx',
  'src/app/pages/OperationsPage.tsx',
  'src/app/pages/EventsPage.tsx',
  'src/app/pages/SettingsPage.tsx',
  'src/features/account-settings/AccountSettingsModal.tsx',
  'src/features/connection/ConnectionSettingsPanel.tsx',
  'src/features/settings/BedZoneRecognitionPanel.tsx',
  'src/features/settings/CameraEditModal.tsx',
  'src/features/settings/CameraRegisterModal.tsx',
  'src/features/settings/CameraSection.tsx',
  'src/features/settings/CameraTable.tsx',
  'src/features/settings/ClipStorageCard.tsx',
  'src/features/settings/DeleteCameraDialog.tsx',
  'src/features/settings/DetectionSettingsCard.tsx',
  'src/features/settings/FolderBrowserModal.tsx',
  'src/features/settings/ProcessingStatusCard.tsx',
  'src/shared/ui/AccessibleDialog.tsx',
  'src/shared/ui/AuthGate.tsx',
  'src/shared/ui/NavBar.tsx',
  'src/shared/ui/StatusBadge.tsx',
  'src/shared/ui/Toast.tsx',
] as const;
const appReachableComponentSources = Object.fromEntries(
  appReachableComponentPaths.map((path) => [path, readFileSync(path, 'utf8')]),
);

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
  it('keeps muted supporting text readable on both background and card surfaces', () => {
    const mutedForeground = cssColor('--muted-foreground');

    expect(documentedColor('--muted-foreground').toLowerCase()).toBe(mutedForeground.toLowerCase());
    expect(contrastRatio(mutedForeground, cssColor('--background'))).toBeGreaterThanOrEqual(4.5);
    expect(contrastRatio(mutedForeground, cssColor('--card'))).toBeGreaterThanOrEqual(4.5);
  });

  it('keeps primary action text readable on the primary fill', () => {
    expect(documentedColor('--primary').toLowerCase()).toBe(cssColor('--primary').toLowerCase());
    expect(contrastRatio(cssColor('--primary-foreground'), cssColor('--primary'))).toBeGreaterThanOrEqual(4.5);
  });

  it('keeps media overlay text readable on dark media frames', () => {
    expect(css).toMatch(/--media-bg:\s*#0d0d0d;/i);
    expect(css).toMatch(/--media-label-bg:\s*rgba\(0,\s*0,\s*0,\s*0\.72\);/i);
  });

  it('uses a quiet single-column authentication card without a marketing brand panel', () => {
    expect(css).toMatch(/\.auth-card\s*\{[^}]*width:\s*min\(380px,\s*100%\)/s);
    expect(css).toMatch(/\.auth-card\s*\{[^}]*min-height:\s*360px/s);
    expect(css).not.toMatch(/\.auth-card\s*\{[^}]*box-shadow:/s);
    expect(css).not.toMatch(/\.auth-brand-panel\s*\{/);
    expect(css).not.toMatch(/\.auth-card\s*\{[^}]*grid-template-columns/s);
  });

  it('keeps the desktop NavBar a 56px bar with no left rail or bottom tab bar', () => {
    expect(css).toMatch(/\.app-navbar\s*\{[^}]*height:\s*56px/s);
    expect(css).not.toMatch(/\.desktop-rail\s*\{/);
    expect(css).not.toMatch(/\.bottom-tab(?:s|-bar)?\s*\{/);
  });

  it('reflows narrow NavBars without wrapping labels or shrinking interactive targets', () => {
    // Given: the dashboard shell is rendered below the tablet breakpoint.
    const narrowViewportRules = responsiveCss.match(/@media\s*\(max-width:\s*767px\)\s*\{([\s\S]*)\}\s*@media\s*\(min-width:/)?.[1] ?? '';

    // When: the narrow-viewport responsive rules are applied.
    // Then: the shell reflows and every navigation action retains an unbroken 44px target.
    expect(narrowViewportRules).toMatch(/\.app-navbar\s*\{[^}]*display:\s*grid;[^}]*grid-template-areas:\s*"brand brand"\s*"nav actions";/s);
    expect(narrowViewportRules).toMatch(/\.app-nav button\s*\{[^}]*min-height:\s*44px;[^}]*white-space:\s*nowrap;/s);
    expect(narrowViewportRules).toMatch(/\.icon-button,\s*\.logout-button\s*\{[^}]*min-height:\s*44px;/s);
  });

  it.each(Object.entries(appReachableComponentSources))('keeps App-reachable %s free of raw black/white utility aliases', (_name, source) => {
    expect(source).not.toMatch(/\b(?:bg-black|bg-white|text-black|text-white)(?:\/\d+)?\b/);
  });

  it.each(Object.entries(appReachableComponentSources))('keeps App-reachable %s free of retired pre-redesign token aliases', (_name, source) => {
    const retiredAliases = [
      'bg-surface', 'bg-surface2', 'text-ink', 'text-ink-soft', 'text-ink-faint',
      'ring-brand', 'text-brand', 'bg-brand-soft', 'bg-brand',
      'bg-status-dangerBg', 'text-status-danger', 'bg-status-stableBg', 'text-status-stable',
      'bg-status-cautionBg', 'text-status-caution',
    ];
    const classNames = [...source.matchAll(/className=(?:"([^"]*)"|\{`([^`]*)`)/g)]
      .flatMap((match) => (match[1] ?? match[2] ?? '').split(/\s+/));

    const found = classNames.filter((className) => retiredAliases.includes(className));
    expect(found).toEqual([]);
  });

  it('keeps the raw primary fill declaration inside the semantic brand-action class only', () => {
    const withoutSemanticAction = css.replace(/\.brand-action\s*\{[^}]*\}/s, '');

    expect(withoutSemanticAction).not.toMatch(/\bbackground(?:-color)?\s*:\s*var\(--primary\)(?!-)/);
  });

  it('keeps reachable stylesheet declarations free of raw black/white literal values', () => {
    const declarations = css.replace(/^\s*--[\w-]+:\s*[^;]+;\s*$/gm, '');

    expect(declarations).not.toMatch(/\b(?:color|background(?:-color)?)\s*:\s*(?:black|white|#(?:000|000000|fff|ffffff))\b/i);
  });
});
