import { requestJson } from '@/shared/api/http';
import {
  normalizeCameraTopology,
  normalizeTopologyConfirmation,
  normalizeTopologyPreview,
  normalizeTopologySync,
} from '@/shared/api/topologyNormalizer';
import type {
  CameraTopology,
  TopologyConfirmationInput,
  TopologyConfirmationResult,
  TopologyFloorInput,
  TopologyPreviewResponse,
  TopologyRoomInput,
  TopologySyncResult,
} from '@/shared/api/topologyTypes';

export type * from '@/shared/api/topologyTypes';

export async function fetchTopology(signal?: AbortSignal): Promise<CameraTopology> {
  return normalizeCameraTopology(await requestJson('/cameras/topology', { signal }));
}

export async function createTopologyFloor(input: TopologyFloorInput): Promise<TopologyFloorInput> {
  await requestJson('/cameras/topology/floors', { method: 'POST', body: JSON.stringify(input) });
  return input;
}

export async function updateTopologyFloor(edgeRef: string, input: Omit<TopologyFloorInput, 'edge_ref'>): Promise<void> {
  await requestJson(`/cameras/topology/floors/${encodeURIComponent(edgeRef)}`, { method: 'PATCH', body: JSON.stringify(input) });
}

export async function deleteTopologyFloor(edgeRef: string): Promise<void> {
  await requestJson(`/cameras/topology/floors/${encodeURIComponent(edgeRef)}`, { method: 'DELETE' });
}

export async function createTopologyRoom(input: TopologyRoomInput): Promise<TopologyRoomInput> {
  await requestJson('/cameras/topology/rooms', { method: 'POST', body: JSON.stringify(input) });
  return input;
}

export async function updateTopologyRoom(edgeRef: string, name: string): Promise<void> {
  await requestJson(`/cameras/topology/rooms/${encodeURIComponent(edgeRef)}`, { method: 'PATCH', body: JSON.stringify({ name }) });
}

export async function deleteTopologyRoom(edgeRef: string): Promise<void> {
  await requestJson(`/cameras/topology/rooms/${encodeURIComponent(edgeRef)}`, { method: 'DELETE' });
}

export async function syncTopology(): Promise<TopologySyncResult> {
  return normalizeTopologySync(await requestJson('/connection/sync-cameras', { method: 'POST' }));
}

export async function fetchTopologyPreview(signal?: AbortSignal): Promise<TopologyPreviewResponse> {
  return normalizeTopologyPreview(await requestJson('/connection/topology-preview', { signal }));
}

export async function confirmTopologyPreview(input: TopologyConfirmationInput): Promise<TopologyConfirmationResult> {
  return normalizeTopologyConfirmation(await requestJson('/connection/topology-preview/confirm', {
    method: 'POST',
    body: JSON.stringify(input),
  }));
}
