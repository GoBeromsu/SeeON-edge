import { describe, expect, it } from 'vitest';
import type { CameraRegistry, ConnectionView } from '@/shared/api/types';
import type { CameraTopology, TopologyPreview } from '@/shared/api/topologyClient';
import {
  cameraTotal,
  deriveActiveStep,
  hasCameraCountMismatch,
  isCameraStepComplete,
  isDeviceStepComplete,
  isServerStepComplete,
  mappedCameraTotal,
} from '@/features/connection/wizardSteps';

function connection(overrides: Partial<ConnectionView> = {}): ConnectionView {
  return {
    events_url: 'https://api.eldercare.example/api/v1/events',
    config_url: 'https://api.eldercare.example/api/v1/ml-config',
    facility_code: 'NH-7H2K9M4QXP',
    client_installation_ref: 'aa83ea3f-6e5f-4f45-a401-fb36c38835b6',
    facility_id: 'facility-42',
    edge_installation_id: 'c72bd9a7-3e04-47ba-a8cd-a56e54f98152',
    enrollment_generation: 1,
    facility_token_set: true,
    facility_token_masked: '****ab12',
    enrolled: true,
    configured: true,
    reachable: true,
    last_ok_at: '2026-08-01T00:00:00Z',
    updated_at: '2026-08-01T00:00:00Z',
    ...overrides,
  };
}

function camera(id: string): CameraRegistry['cameras'][number] {
  return { id, label: id, rtsp_url_masked: 'rtsp://redacted/x', floor_name: null, status: 'online', created_at: null };
}

function cameras(count: number): CameraRegistry {
  return { registry_version: 1, cameras: Array.from({ length: count }, (_, index) => camera(`cam-${index}`)) };
}

function topology(overrides: Partial<CameraTopology> = {}): CameraTopology {
  return { registry_version: 1, dirty_registry_version: null, readiness_error: null, unmapped_camera_ids: [], floors: [], ...overrides };
}

function topologyWithMappedCameras(count: number): CameraTopology {
  return topology({
    floors: [
      {
        edge_ref: 'floor-1',
        name: '1층',
        order_index: 0,
        rooms: [
          {
            edge_ref: 'room-1',
            name: '101호',
            room_type: 'ROOM',
            capacity: 1,
            legacy_canonical_space_id: null,
            cameras: Array.from({ length: count }, (_, index) => ({ edge_ref: `cam-${index}`, label: `cam-${index}` })),
          },
        ],
      },
    ],
  });
}

function preview(overrides: Partial<TopologyPreview> = {}): TopologyPreview {
  return {
    confirmation_id: '0197f671-3a31-7a6c-a6e4-83ed412de81c',
    digest: 'a'.repeat(64),
    expires_at: '2026-08-01T00:10:00Z',
    snapshot_id: '0197f671-3a31-7a6c-a6e4-83ed412de81d',
    client_revision: 1,
    server_revision: 1,
    cameras: 0,
    rooms: 0,
    floors: 0,
    confirmed: false,
    ...overrides,
  };
}

describe('isDeviceStepComplete', () => {
  it('is true only once the backend has persisted a full enrollment (enrolled === true)', () => {
    expect(isDeviceStepComplete(connection({ enrolled: true }))).toBe(true);
  });

  it('is false while enrolled is false, even if the weaker configured flag is true', () => {
    expect(isDeviceStepComplete(connection({ enrolled: false, configured: true }))).toBe(false);
  });

  it('is false when there is no connection resource yet', () => {
    expect(isDeviceStepComplete(null)).toBe(false);
  });
});

describe('cameraTotal / mappedCameraTotal', () => {
  it('counts registered cameras', () => {
    expect(cameraTotal(cameras(3))).toBe(3);
    expect(cameraTotal(null)).toBe(0);
  });

  it('sums cameras placed into rooms across all floors', () => {
    expect(mappedCameraTotal(topologyWithMappedCameras(2))).toBe(2);
    expect(mappedCameraTotal(topology())).toBe(0);
    expect(mappedCameraTotal(null)).toBe(0);
  });
});

describe('isCameraStepComplete', () => {
  it('is false when the edge has zero registered cameras, even if nothing is dirty', () => {
    expect(isCameraStepComplete(cameras(0), topology())).toBe(false);
  });

  it('is false while a camera-registry push to the server is still owed (dirty_registry_version set)', () => {
    expect(isCameraStepComplete(cameras(2), topology({ dirty_registry_version: 3 }))).toBe(false);
  });

  it('is false while any camera is unmapped (readiness_error set)', () => {
    expect(isCameraStepComplete(cameras(2), topology({ readiness_error: 'LEGACY_MAPPING_REQUIRED', unmapped_camera_ids: ['cam-0'] }))).toBe(false);
  });

  it('is true once cameras exist, nothing is dirty, and nothing is unmapped', () => {
    expect(isCameraStepComplete(cameras(2), topology())).toBe(true);
  });

  it('is false when the topology has not loaded yet', () => {
    expect(isCameraStepComplete(cameras(2), null)).toBe(false);
  });
});

describe('isServerStepComplete', () => {
  it('is false if step 2 has not completed, regardless of preview state', () => {
    expect(isServerStepComplete(cameras(0), topology(), null)).toBe(false);
  });

  it('is true when step 2 is complete and there is no pending preview', () => {
    expect(isServerStepComplete(cameras(2), topology(), null)).toBe(true);
  });

  it('is false when a preview is pending and has not been confirmed', () => {
    expect(isServerStepComplete(cameras(2), topology(), preview({ confirmed: false }))).toBe(false);
  });

  it('is true once the pending preview has been confirmed', () => {
    expect(isServerStepComplete(cameras(2), topology(), preview({ confirmed: true }))).toBe(true);
  });
});

describe('hasCameraCountMismatch', () => {
  it('is false when every registered camera is mapped into the topology', () => {
    expect(hasCameraCountMismatch(cameras(2), topologyWithMappedCameras(2))).toBe(false);
  });

  it('is true when the mapped total drifts from the edge registry total', () => {
    expect(hasCameraCountMismatch(cameras(3), topologyWithMappedCameras(2))).toBe(true);
  });

  it('is false when the topology has not loaded yet (nothing to compare against)', () => {
    expect(hasCameraCountMismatch(cameras(3), null)).toBe(false);
  });
});

describe('deriveActiveStep', () => {
  it('returns 1 when device enrollment has not completed', () => {
    expect(deriveActiveStep({ connection: connection({ enrolled: false }), cameras: cameras(0), topology: null, preview: null })).toBe(1);
  });

  it('returns 2 once enrolled but cameras are not yet synced', () => {
    expect(deriveActiveStep({ connection: connection({ enrolled: true }), cameras: cameras(0), topology: topology(), preview: null })).toBe(2);
  });

  it('returns 3 once device and camera steps are complete', () => {
    expect(deriveActiveStep({ connection: connection({ enrolled: true }), cameras: cameras(2), topology: topology(), preview: null })).toBe(3);
  });

  it('stays on 3 while an unconfirmed preview is pending -- confirm is never auto-advanced', () => {
    expect(deriveActiveStep({
      connection: connection({ enrolled: true }),
      cameras: cameras(2),
      topology: topology(),
      preview: preview({ confirmed: false }),
    })).toBe(3);
  });
});
