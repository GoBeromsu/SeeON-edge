import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  applyDetectionPolicy,
  diffDetectionPolicy,
  fetchDetectionPolicies,
  rollbackDetectionPolicy,
} from '@/shared/api/client';
import { PolicyEvidenceCard } from '@/features/settings/PolicyEvidenceCard';
import type {
  CameraRegistry,
  DetectionPolicyCatalog,
  DetectionPolicyDiff,
  EffectiveDetectionPolicy,
} from '@/shared/api/types';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return {
    ...actual,
    fetchDetectionPolicies: vi.fn(),
    diffDetectionPolicy: vi.fn(),
    applyDetectionPolicy: vi.fn(),
    rollbackDetectionPolicy: vi.fn(),
  };
});

vi.mock('@/shared/ui/Toast', () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

const fallPolicy: EffectiveDetectionPolicy = {
  module_id: 'fall',
  module_version: 1,
  schema_id: 'fall.policy',
  schema_version: 1,
  source: 'facility-default',
  facility_revision_id: 4,
  camera_revision_id: null,
  values: { operating_threshold: 0.5 },
  effective_policy_id: 'fall-facility-4',
};

const cameraPolicy: EffectiveDetectionPolicy = {
  ...fallPolicy,
  source: 'camera-override',
  camera_revision_id: 8,
  values: { operating_threshold: 0.75 },
  effective_policy_id: 'fall-camera-8',
};

const catalog: DetectionPolicyCatalog = {
  activation_generation: 12,
  modules: [
    {
      qualified_id: 'fall.v1',
      policy_qualified_id: 'fall.policy.v1',
      units: { operating_threshold: 'probability [0,1]' },
    },
  ],
  effective: {
    defaults: { fall: fallPolicy },
    cameras: { 'cam-1': { fall: cameraPolicy } },
  },
  activations: [],
};

const cameras: CameraRegistry = {
  registry_version: 1,
  cameras: [
    {
      id: 'local-1',
      backend_camera_id: 'cam-1',
      label: '서울 301호',
      rtsp_url_masked: 'rtsp://***',
      floor_name: '3층',
      status: 'online',
      created_at: '2026-08-01T00:00:00Z',
    },
  ],
};

function comparedDiff(overrides: Partial<DetectionPolicyDiff> = {}): DetectionPolicyDiff {
  return {
    changed: true,
    current: fallPolicy,
    proposed: {
      ...fallPolicy,
      values: { operating_threshold: 0.8 },
      facility_revision_id: 0,
      effective_policy_id: 'proposed',
    },
    compared_payload: {
      module_id: 'fall',
      module_version: 1,
      schema_id: 'fall.policy',
      schema_version: 1,
      camera_id: null,
      values: { operating_threshold: 0.8 },
    },
    concurrency_token: 4,
    ...overrides,
  };
}

function setInputValue(input: HTMLInputElement, value: string): void {
  const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
  valueSetter?.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
}

function setSelectValue(select: HTMLSelectElement, value: string): void {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value')?.set;
  setter?.call(select, value);
  select.dispatchEvent(new Event('change', { bubbles: true }));
}

async function renderCard(): Promise<{ host: HTMLElement; root: ReturnType<typeof createRoot> }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  await act(async () => {
    root.render(<PolicyEvidenceCard cameras={cameras} />);
  });
  await act(async () => {
    await Promise.resolve();
  });
  return { host, root };
}

function findButton(host: HTMLElement, label: string): HTMLButtonElement {
  const button = Array.from(host.querySelectorAll('button')).find(
    (candidate) => candidate.textContent === label,
  );
  if (!button) throw new Error(`missing button ${label}`);
  return button as HTMLButtonElement;
}

function thresholdInput(host: HTMLElement): HTMLInputElement {
  const input = host.querySelector('input[type="number"]');
  if (!(input instanceof HTMLInputElement)) throw new Error('missing threshold input');
  return input;
}

function scopeSelect(host: HTMLElement): HTMLSelectElement {
  const labels = Array.from(host.querySelectorAll('label'));
  const scope = labels.find((label) => label.textContent?.includes('적용 범위'));
  const select = scope?.querySelector('select');
  if (!(select instanceof HTMLSelectElement)) throw new Error('missing scope select');
  return select;
}

beforeEach(() => {
  vi.mocked(fetchDetectionPolicies).mockReset();
  vi.mocked(diffDetectionPolicy).mockReset();
  vi.mocked(applyDetectionPolicy).mockReset();
  vi.mocked(rollbackDetectionPolicy).mockReset();
  vi.mocked(fetchDetectionPolicies).mockResolvedValue(catalog);
  vi.mocked(diffDetectionPolicy).mockResolvedValue(comparedDiff());
  vi.mocked(applyDetectionPolicy).mockResolvedValue(undefined);
  vi.mocked(rollbackDetectionPolicy).mockResolvedValue(undefined);
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

describe('PolicyEvidenceCard concurrency', () => {
  it('applies the frozen compared payload and refuses edited-after-diff apply', async () => {
    const { host, root } = await renderCard();

    await act(async () => {
      setInputValue(thresholdInput(host), '0.8');
    });
    await act(async () => {
      findButton(host, '차이 비교').click();
      await Promise.resolve();
    });
    expect(host.textContent).toContain('변경 확인:');
    expect(findButton(host, '검토한 변경 적용').disabled).toBe(false);

    await act(async () => {
      setInputValue(thresholdInput(host), '0.99');
    });
    expect(host.textContent).not.toContain('변경 확인:');
    expect(findButton(host, '검토한 변경 적용').disabled).toBe(true);
    expect(applyDetectionPolicy).not.toHaveBeenCalled();

    vi.mocked(diffDetectionPolicy).mockResolvedValueOnce(
      comparedDiff({
        compared_payload: {
          module_id: 'fall',
          module_version: 1,
          schema_id: 'fall.policy',
          schema_version: 1,
          camera_id: null,
          values: { operating_threshold: 0.8 },
        },
        concurrency_token: 4,
      }),
    );
    await act(async () => {
      setInputValue(thresholdInput(host), '0.8');
    });
    await act(async () => {
      findButton(host, '차이 비교').click();
      await Promise.resolve();
    });
    await act(async () => {
      findButton(host, '검토한 변경 적용').click();
      await Promise.resolve();
    });

    expect(applyDetectionPolicy).toHaveBeenCalledTimes(1);
    expect(applyDetectionPolicy).toHaveBeenCalledWith({
      module_id: 'fall',
      module_version: 1,
      schema_id: 'fall.policy',
      schema_version: 1,
      camera_id: null,
      values: { operating_threshold: 0.8 },
      expected_revision_id: 4,
    });

    act(() => root.unmount());
  });

  it('ignores a stale diff response after scope change', async () => {
    let finishDiff: ((value: DetectionPolicyDiff) => void) | undefined;
    vi.mocked(diffDetectionPolicy).mockImplementationOnce(
      () =>
        new Promise<DetectionPolicyDiff>((resolve) => {
          finishDiff = resolve;
        }),
    );

    const { host, root } = await renderCard();
    await act(async () => {
      findButton(host, '차이 비교').click();
    });
    expect(diffDetectionPolicy).toHaveBeenCalledTimes(1);

    await act(async () => {
      setSelectValue(scopeSelect(host), 'cam-1');
    });

    await act(async () => {
      finishDiff?.(
        comparedDiff({
          compared_payload: {
            module_id: 'fall',
            module_version: 1,
            schema_id: 'fall.policy',
            schema_version: 1,
            camera_id: null,
            values: { operating_threshold: 0.8 },
          },
          concurrency_token: 4,
        }),
      );
      await Promise.resolve();
    });

    expect(host.textContent).not.toContain('변경 확인:');
    expect(findButton(host, '검토한 변경 적용').disabled).toBe(true);

    act(() => root.unmount());
  });

  it('sends rollback CAS token from the current scope revision', async () => {
    const { host, root } = await renderCard();
    await act(async () => {
      setSelectValue(scopeSelect(host), 'cam-1');
      await Promise.resolve();
    });
    await act(async () => {
      findButton(host, '이전 개정으로 되돌리기').click();
      await Promise.resolve();
    });
    expect(rollbackDetectionPolicy).toHaveBeenCalledWith({
      module_id: 'fall',
      module_version: 1,
      camera_id: 'cam-1',
      expected_revision_id: 8,
    });
    act(() => root.unmount());
  });
});
