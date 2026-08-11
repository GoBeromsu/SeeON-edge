export type TopologyCamera = {
  readonly edge_ref: string;
  readonly label: string;
};

export type TopologyRoom = {
  readonly edge_ref: string;
  readonly name: string;
  readonly room_type: 'ROOM';
  readonly capacity: number;
  readonly legacy_canonical_space_id: string | null;
  readonly cameras: readonly TopologyCamera[];
};

export type TopologyFloor = {
  readonly edge_ref: string;
  readonly name: string;
  readonly order_index: number;
  readonly rooms: readonly TopologyRoom[];
};

export type TopologyReadinessError =
  | 'INVALID_SCHEMA'
  | 'INVALID_TOPOLOGY'
  | 'LEGACY_MAPPING_REQUIRED';

export type CameraTopology = {
  readonly registry_version: number;
  readonly dirty_registry_version: number | null;
  readonly readiness_error: TopologyReadinessError | null;
  readonly unmapped_camera_ids: readonly string[];
  readonly floors: readonly TopologyFloor[];
};

export type TopologyFloorInput = {
  readonly edge_ref: string;
  readonly name: string;
  readonly order_index: number;
};

export type TopologyRoomInput = {
  readonly edge_ref: string;
  readonly floor_edge_ref: string;
  readonly name: string;
  readonly legacy_canonical_space_id: string | null;
};

export type TopologySyncStatus = 'pending' | 'synced' | 'failed' | 'disabled';
export type TopologySyncError = 'unconfigured' | 'auth' | 'timeout' | 'unreachable' | 'conflict';

export type TopologySyncResult = {
  readonly status: TopologySyncStatus;
  readonly error_class: TopologySyncError | null;
  readonly detail: string | null;
  readonly last_ok_at: string | null;
  readonly next_retry_at: string | null;
  readonly camera_count: number;
};

export type TopologyPreview = {
  readonly confirmation_id: string;
  readonly digest: string;
  readonly expires_at: string;
  readonly snapshot_id: string;
  readonly client_revision: number;
  readonly server_revision: number;
  readonly cameras: number;
  readonly rooms: number;
  readonly floors: number;
  readonly confirmed: boolean;
};

export type TopologyPreviewResponse = { readonly preview: TopologyPreview | null };

export type TopologyConfirmationInput = {
  readonly confirmation_id: string;
  readonly digest: string;
  readonly client_revision: number;
  readonly server_revision: number;
};

export type TopologyConfirmationResult = {
  readonly snapshot_id: string;
  readonly client_revision: number;
  readonly server_revision: number;
};
