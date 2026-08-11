import type { CameraRegistry, ConnectionView } from '@/shared/api/types';
import type { CameraTopology, TopologyPreview } from '@/shared/api/topologyClient';

/**
 * Pure, server-state-only derivation of the 3-step Edge setup wizard's progress. Deliberately
 * takes no React state / localStorage as input: an operator who refreshes mid-setup must land on
 * the step the server actually reflects, not a step the browser merely remembers (enrollment must
 * survive a restart -- see front/AGENTS.md and the wizard brief).
 */

export type WizardStepNumber = 1 | 2 | 3;

export type WizardServerState = {
  readonly connection: ConnectionView | null;
  readonly cameras: CameraRegistry | null;
  readonly topology: CameraTopology | null;
  readonly preview: TopologyPreview | null;
};

/**
 * "enrolled" (not the looser "configured") is the backend's strict full facility-code +
 * enrollment-token contract: it is only true once a `PUT /connection` has round-tripped through
 * `verify_enrollment` and persisted facility_id + edge_installation_id (see
 * backend/app/features/connection/router.py `_status_response`). That round trip performs the
 * exact same verification as the standalone `POST /connection/test` probe, so treating it as
 * "1단계 test succeeded" is accurate, and -- unlike a client-only "did test return ok" flag -- it
 * survives a reload because it is read back from the connection resource itself.
 */
export function isDeviceStepComplete(connection: ConnectionView | null): boolean {
  return connection?.enrolled === true;
}

export function cameraTotal(cameras: CameraRegistry | null): number {
  return cameras?.cameras.length ?? 0;
}

/** Cameras actually placed into a floor/room in the topology snapshot (i.e. sync-eligible). */
export function mappedCameraTotal(topology: CameraTopology | null): number {
  if (!topology) return 0;
  return topology.floors.reduce(
    (sum, floor) => sum + floor.rooms.reduce((roomSum, room) => roomSum + room.cameras.length, 0),
    0,
  );
}

/**
 * Step 2 completes only once the edge actually has cameras AND the last sync-cameras push
 * succeeded for the current registry (`dirty_registry_version === null`) with no outstanding
 * mapping gaps (`readiness_error === null`). A registry with zero cameras can trivially report
 * "nothing dirty" -- that must never read as success (brief: "zero cameras is a real state").
 */
export function isCameraStepComplete(cameras: CameraRegistry | null, topology: CameraTopology | null): boolean {
  if (cameraTotal(cameras) === 0) return false;
  if (!topology) return false;
  return topology.readiness_error === null && topology.dirty_registry_version === null;
}

/**
 * Step 3 completes once step 2 has succeeded and there is nothing left to confirm: either no
 * preview is pending at all (the sync applied cleanly with nothing needing operator sign-off), or
 * the last-fetched preview has already been confirmed.
 */
export function isServerStepComplete(
  cameras: CameraRegistry | null,
  topology: CameraTopology | null,
  preview: TopologyPreview | null,
): boolean {
  if (!isCameraStepComplete(cameras, topology)) return false;
  return preview === null || preview.confirmed === true;
}

/**
 * Last-line-of-defense check (brief: "the Hub camera count matches the Edge's total"). Once step 2
 * is complete every edge camera must be bound into the topology (readiness_error would otherwise
 * be LEGACY_MAPPING_REQUIRED) -- so any gap here means the two counts drifted for a reason no
 * single-field check above catches, and must be surfaced rather than silently confirmed.
 */
export function hasCameraCountMismatch(cameras: CameraRegistry | null, topology: CameraTopology | null): boolean {
  if (!topology) return false;
  return mappedCameraTotal(topology) !== cameraTotal(cameras);
}

/** The step the operator should land on right now, derived purely from server state. */
export function deriveActiveStep(state: WizardServerState): WizardStepNumber {
  if (!isDeviceStepComplete(state.connection)) return 1;
  if (!isCameraStepComplete(state.cameras, state.topology)) return 2;
  return 3;
}
