-- Apply once, transactionally, with the destination schema first in search_path.
-- Namespace/role creation and migration ledger stamps belong to the bootstrap.
-- JSON remains text: validation must not rewrite the bytes used by hashes.

CREATE FUNCTION seeon_utc_timestamp(value text) RETURNS boolean
LANGUAGE plpgsql IMMUTABLE STRICT SET search_path = pg_catalog AS $$
DECLARE
    y integer;
    m integer;
    d integer;
    days integer[] := ARRAY[31,28,31,30,31,30,31,31,30,31,30,31];
BEGIN
    IF value !~ '^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]([.][0-9]{1,6})?Z$' THEN
        RETURN false;
    END IF;
    y := substring(value, 1, 4)::integer;
    m := substring(value, 6, 2)::integer;
    d := substring(value, 9, 2)::integer;
    IF y % 4 = 0 AND (y % 100 <> 0 OR y % 400 = 0) THEN
        days[2] := 29;
    END IF;
    RETURN d <= days[m];
END;
$$;

CREATE FUNCTION seeon_relpath(value text) RETURNS boolean
LANGUAGE sql IMMUTABLE STRICT SET search_path = pg_catalog AS $$
    SELECT length(value) BETWEEN 1 AND 512
        AND left(value, 1) <> '/'
        AND strpos(value, chr(92)) = 0
        AND strpos('/' || value || '/', '/../') = 0
$$;

CREATE TABLE schema_migrations (
    version bigint PRIMARY KEY CHECK (version > 0),
    name text NOT NULL UNIQUE,
    checksum text NOT NULL CHECK (length(checksum) = 64),
    applied_at text NOT NULL,
    source_schema_version bigint CHECK (source_schema_version > 0),
    source_db_sha256 text CHECK (source_db_sha256 ~ '^[0-9a-f]{64}$'),
    reconciliation_sha256 text CHECK (reconciliation_sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE credentials (
    id bigint PRIMARY KEY CHECK (id = 1),
    username text NOT NULL CHECK (length(username) BETWEEN 1 AND 128),
    algorithm text NOT NULL CHECK (algorithm = 'scrypt'),
    salt bytea NOT NULL CHECK (octet_length(salt) = 16),
    password_hash bytea NOT NULL CHECK (octet_length(password_hash) = 64),
    updated_at text NOT NULL CHECK (seeon_utc_timestamp(updated_at))
);

CREATE TABLE edge_site (
    id bigint PRIMARY KEY CHECK (id = 1),
    facility_code text CHECK (length(facility_code) BETWEEN 1 AND 64),
    client_installation_ref text CHECK (length(client_installation_ref) BETWEEN 1 AND 128),
    facility_id text CHECK (length(facility_id) BETWEEN 1 AND 128),
    facility_token text CHECK (length(facility_token) BETWEEN 1 AND 512),
    edge_installation_id text CHECK (length(edge_installation_id) BETWEEN 1 AND 128),
    enrollment_generation bigint CHECK (enrollment_generation > 0),
    enrollment_created_at text CHECK (seeon_utc_timestamp(enrollment_created_at)),
    enrollment_updated_at text CHECK (seeon_utc_timestamp(enrollment_updated_at)),
    registry_version bigint NOT NULL DEFAULT 0 CHECK (registry_version >= 0),
    clip_store_subdir text CHECK (seeon_relpath(clip_store_subdir)),
    clip_export_enabled bigint NOT NULL DEFAULT 0 CHECK (clip_export_enabled IN (0, 1)),
    runtime_settings_version bigint NOT NULL DEFAULT 0 CHECK (runtime_settings_version >= 0),
    fall_on bigint CHECK (fall_on IN (0, 1)),
    fall_mode text CHECK (fall_mode IN ('always', 'window')),
    fall_start_time text CHECK (fall_start_time ~ '^([01][0-9]|2[0-3]):[0-5][0-9]$'),
    fall_end_time text CHECK (fall_end_time ~ '^([01][0-9]|2[0-3]):[0-5][0-9]$'),
    bed_exit_on bigint CHECK (bed_exit_on IN (0, 1)),
    bed_exit_mode text CHECK (bed_exit_mode IN ('always', 'window')),
    bed_exit_start_time text CHECK (bed_exit_start_time ~ '^([01][0-9]|2[0-3]):[0-5][0-9]$'),
    bed_exit_end_time text CHECK (bed_exit_end_time ~ '^([01][0-9]|2[0-3]):[0-5][0-9]$'),
    topology_snapshot_registry_version bigint NOT NULL DEFAULT 0
        CHECK (topology_snapshot_registry_version >= 0),
    topology_client_revision bigint NOT NULL DEFAULT 0 CHECK (topology_client_revision >= 0),
    topology_server_revision bigint NOT NULL DEFAULT 0 CHECK (topology_server_revision >= 0),
    topology_pending_snapshot_id text CHECK (length(topology_pending_snapshot_id) BETWEEN 1 AND 128),
    topology_pending_body bytea CHECK (octet_length(topology_pending_body) BETWEEN 1 AND 1048576),
    topology_pending_registry_version bigint CHECK (topology_pending_registry_version >= 0),
    topology_pending_client_revision bigint CHECK (topology_pending_client_revision > 0),
    topology_pending_expected_server_revision bigint
        CHECK (topology_pending_expected_server_revision >= 0),
    topology_consecutive_failures bigint NOT NULL DEFAULT 0 CHECK (topology_consecutive_failures >= 0),
    topology_next_retry_at double precision,
    topology_pause_reason text CHECK (topology_pause_reason IN ('auth', 'forbidden', 'conflict')),
    topology_last_accepted_at double precision,
    topology_dirty_registry_version bigint CHECK (topology_dirty_registry_version >= 1),
    topology_dirty_created_at text CHECK (seeon_utc_timestamp(topology_dirty_created_at)),
    topology_confirmation_id text CHECK (length(topology_confirmation_id) BETWEEN 1 AND 128),
    topology_confirmation_digest text CHECK (topology_confirmation_digest ~ '^[0-9a-f]{64}$'),
    topology_confirmation_expires_at text CHECK (seeon_utc_timestamp(topology_confirmation_expires_at)),
    topology_confirmation_snapshot_id text
        CHECK (length(topology_confirmation_snapshot_id) BETWEEN 1 AND 128),
    topology_confirmation_client_revision bigint CHECK (topology_confirmation_client_revision >= 0),
    topology_confirmation_server_revision bigint CHECK (topology_confirmation_server_revision >= 0),
    topology_confirmation_registry_version bigint CHECK (topology_confirmation_registry_version >= 0),
    topology_confirmation_cameras bigint CHECK (topology_confirmation_cameras >= 0),
    topology_confirmation_rooms bigint CHECK (topology_confirmation_rooms >= 0),
    topology_confirmation_floors bigint CHECK (topology_confirmation_floors >= 0),
    topology_confirmation_confirmed bigint CHECK (topology_confirmation_confirmed IN (0, 1)),
    topology_confirmation_result text CHECK (length(topology_confirmation_result) BETWEEN 1 AND 64),
    storage_state text CHECK (length(storage_state) BETWEEN 1 AND 64),
    recording_suspended bigint CHECK (recording_suspended IN (0, 1)),
    audit_state text CHECK (length(audit_state) BETWEEN 1 AND 64),
    audit_last_success_at text CHECK (seeon_utc_timestamp(audit_last_success_at)),
    degraded_failure_code text CHECK (length(degraded_failure_code) BETWEEN 1 AND 64),
    degraded_observed_at text CHECK (seeon_utc_timestamp(degraded_observed_at)),
    updated_at text NOT NULL CHECK (seeon_utc_timestamp(updated_at)),
    CHECK (num_nonnulls(facility_code, client_installation_ref, facility_id, facility_token,
        edge_installation_id, enrollment_generation, enrollment_created_at,
        enrollment_updated_at) IN (0, 8)),
    CHECK (
        num_nonnulls(fall_on, fall_mode, fall_start_time, fall_end_time) = 0
        OR (fall_on IS NOT NULL AND fall_mode IS NOT NULL AND (
            (fall_mode = 'always' AND fall_start_time IS NULL AND fall_end_time IS NULL)
            OR (fall_mode = 'window' AND fall_start_time IS NOT NULL AND fall_end_time IS NOT NULL)
        ))
    ),
    CHECK (
        num_nonnulls(bed_exit_on, bed_exit_mode, bed_exit_start_time, bed_exit_end_time) = 0
        OR (bed_exit_on IS NOT NULL AND bed_exit_mode IS NOT NULL AND (
            (bed_exit_mode = 'always' AND bed_exit_start_time IS NULL AND bed_exit_end_time IS NULL)
            OR (bed_exit_mode = 'window' AND bed_exit_start_time IS NOT NULL AND bed_exit_end_time IS NOT NULL)
        ))
    ),
    CHECK (num_nonnulls(topology_pending_snapshot_id, topology_pending_body,
        topology_pending_registry_version, topology_pending_client_revision,
        topology_pending_expected_server_revision) IN (0, 5)),
    CHECK (num_nonnulls(topology_dirty_registry_version, topology_dirty_created_at) IN (0, 2)),
    CHECK (
        num_nonnulls(topology_confirmation_id, topology_confirmation_digest,
            topology_confirmation_expires_at, topology_confirmation_snapshot_id,
            topology_confirmation_client_revision, topology_confirmation_server_revision,
            topology_confirmation_registry_version, topology_confirmation_cameras,
            topology_confirmation_rooms, topology_confirmation_floors,
            topology_confirmation_confirmed, topology_confirmation_result) = 0
        OR (num_nonnulls(topology_confirmation_id, topology_confirmation_digest,
            topology_confirmation_expires_at, topology_confirmation_snapshot_id,
            topology_confirmation_client_revision, topology_confirmation_server_revision,
            topology_confirmation_registry_version, topology_confirmation_cameras,
            topology_confirmation_rooms, topology_confirmation_floors,
            topology_confirmation_confirmed) = 11
            AND (topology_confirmation_result IS NULL OR topology_confirmation_confirmed = 1))
    ),
    CHECK (
        num_nonnulls(storage_state, recording_suspended, audit_state,
            audit_last_success_at, degraded_failure_code, degraded_observed_at) = 0
        OR num_nonnulls(storage_state, recording_suspended, audit_state, degraded_observed_at) = 4
    )
);

CREATE TABLE locations (
    location_id text NOT NULL CHECK (length(location_id) BETWEEN 1 AND 128),
    kind text NOT NULL CHECK (kind IN ('FLOOR', 'ROOM')),
    parent_location_id text CHECK (length(parent_location_id) BETWEEN 1 AND 128),
    parent_kind text CHECK (parent_kind = 'FLOOR'),
    name text NOT NULL CHECK (length(name) BETWEEN 1 AND 128),
    order_index bigint NOT NULL CHECK (order_index >= 0),
    capacity bigint CHECK (capacity > 0),
    legacy_space_id text UNIQUE CHECK (length(legacy_space_id) BETWEEN 1 AND 128),
    created_at text NOT NULL CHECK (seeon_utc_timestamp(created_at)),
    updated_at text NOT NULL CHECK (seeon_utc_timestamp(updated_at)),
    PRIMARY KEY (location_id, kind),
    FOREIGN KEY (parent_location_id, parent_kind) REFERENCES locations(location_id, kind)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (
        (kind = 'FLOOR' AND parent_location_id IS NULL AND parent_kind IS NULL)
        OR (kind = 'ROOM' AND parent_location_id IS NOT NULL AND parent_kind IS NOT NULL
            AND parent_kind = 'FLOOR')
    )
);

CREATE TABLE cameras (
    camera_id text PRIMARY KEY CHECK (length(camera_id) BETWEEN 1 AND 128),
    incarnation uuid NOT NULL DEFAULT gen_random_uuid(),
    backend_camera_id text CHECK (length(backend_camera_id) BETWEEN 1 AND 128),
    label text NOT NULL CHECK (length(label) BETWEEN 1 AND 128),
    rtsp_url text NOT NULL CHECK (length(rtsp_url) BETWEEN 1 AND 1024),
    normalized_stream_identity text NOT NULL CHECK (length(normalized_stream_identity) BETWEEN 1 AND 256),
    space_id text CHECK (length(space_id) BETWEEN 1 AND 128),
    room_location_id text CHECK (length(room_location_id) BETWEEN 1 AND 128),
    room_location_kind text CHECK (room_location_kind = 'ROOM'),
    edge_ref text CHECK (length(edge_ref) BETWEEN 1 AND 128),
    mapping_state text NOT NULL CHECK (mapping_state IN ('PENDING', 'UNMAPPED', 'MAPPED')),
    decode_backend text CHECK (length(decode_backend) BETWEEN 1 AND 64),
    floor_override text CHECK (length(floor_override) BETWEEN 1 AND 128),
    never_connected bigint NOT NULL CHECK (never_connected IN (0, 1)),
    last_probed_at text CHECK (seeon_utc_timestamp(last_probed_at)),
    last_ok_at text CHECK (seeon_utc_timestamp(last_ok_at)),
    bed_polygon_json text CHECK (bed_polygon_json IS NULL OR (
        bed_polygon_json IS JSON ARRAY AND octet_length(bed_polygon_json) BETWEEN 1 AND 4096)),
    bed_image_width bigint CHECK (bed_image_width > 0),
    bed_image_height bigint CHECK (bed_image_height > 0),
    bed_recognized_at text CHECK (seeon_utc_timestamp(bed_recognized_at)),
    revision bigint NOT NULL CHECK (revision > 0),
    created_at text NOT NULL CHECK (seeon_utc_timestamp(created_at)),
    updated_at text NOT NULL CHECK (seeon_utc_timestamp(updated_at)),
    FOREIGN KEY (room_location_id, room_location_kind) REFERENCES locations(location_id, kind)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (
        num_nonnulls(room_location_id, room_location_kind, edge_ref) = 0
        OR (num_nonnulls(room_location_id, room_location_kind, edge_ref) = 3 AND room_location_kind = 'ROOM')
    ),
    CHECK (
        (mapping_state = 'MAPPED' AND backend_camera_id IS NOT NULL)
        OR (mapping_state IN ('PENDING', 'UNMAPPED') AND backend_camera_id IS NULL)
    ),
    CHECK (num_nonnulls(bed_polygon_json, bed_image_width, bed_image_height, bed_recognized_at) IN (0, 4))
);
CREATE UNIQUE INDEX cameras_one_room_idx ON cameras(room_location_id) WHERE room_location_id IS NOT NULL;

CREATE TABLE policies (
    policy_id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    facility_id text NOT NULL CHECK (length(facility_id) BETWEEN 1 AND 128),
    camera_id text CHECK (length(camera_id) BETWEEN 1 AND 128),
    module_id text NOT NULL CHECK (length(module_id) BETWEEN 1 AND 128),
    module_version bigint NOT NULL CHECK (module_version > 0),
    schema_id text NOT NULL CHECK (length(schema_id) BETWEEN 1 AND 128),
    schema_version bigint NOT NULL CHECK (schema_version > 0),
    active_values_json text CHECK (active_values_json IS NULL OR (
        active_values_json IS JSON OBJECT AND octet_length(active_values_json) BETWEEN 2 AND 16384)),
    active_content_sha256 text CHECK (active_content_sha256 ~ '^[0-9a-f]{64}$'),
    previous_present bigint NOT NULL CHECK (previous_present IN (0, 1)),
    previous_values_json text CHECK (previous_values_json IS NULL OR (
        previous_values_json IS JSON OBJECT AND octet_length(previous_values_json) BETWEEN 2 AND 16384)),
    previous_content_sha256 text CHECK (previous_content_sha256 ~ '^[0-9a-f]{64}$'),
    activation_generation bigint NOT NULL CHECK (activation_generation > 0),
    status text NOT NULL CHECK (status IN ('pending', 'applied', 'failed')),
    refusal_reason text CHECK (length(refusal_reason) BETWEEN 1 AND 256),
    activated_at text NOT NULL CHECK (seeon_utc_timestamp(activated_at)),
    applied_at text CHECK (seeon_utc_timestamp(applied_at)),
    updated_at text NOT NULL CHECK (seeon_utc_timestamp(updated_at)),
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK ((active_values_json IS NULL) = (active_content_sha256 IS NULL)),
    CHECK (camera_id IS NOT NULL OR (active_values_json IS NOT NULL AND active_content_sha256 IS NOT NULL)),
    CHECK ((previous_values_json IS NULL) = (previous_content_sha256 IS NULL)),
    CHECK (previous_present = 1 OR (previous_values_json IS NULL AND previous_content_sha256 IS NULL)),
    CHECK (
        (status = 'pending' AND applied_at IS NULL AND refusal_reason IS NULL)
        OR (status = 'applied' AND applied_at IS NOT NULL AND refusal_reason IS NULL)
        OR (status = 'failed' AND refusal_reason IS NOT NULL)
    )
);
CREATE UNIQUE INDEX policies_facility_scope_idx ON policies(facility_id, module_id, module_version)
    WHERE camera_id IS NULL;
CREATE UNIQUE INDEX policies_camera_scope_idx ON policies(facility_id, camera_id, module_id, module_version)
    WHERE camera_id IS NOT NULL;

CREATE TABLE clips (
    clip_id text PRIMARY KEY CHECK (length(clip_id) BETWEEN 1 AND 128),
    camera_id text NOT NULL CHECK (length(camera_id) BETWEEN 1 AND 128),
    event_facet text NOT NULL CHECK (event_facet IN ('fall', 'bed-exit', 'other')),
    started_at text NOT NULL CHECK (seeon_utc_timestamp(started_at)),
    finalized_at text CHECK (seeon_utc_timestamp(finalized_at)),
    duration_ms bigint CHECK (duration_ms BETWEEN 1 AND 120000),
    codec text CHECK (length(codec) BETWEEN 1 AND 64),
    mime_type text CHECK (length(mime_type) BETWEEN 1 AND 128),
    manifest_relpath text CHECK (seeon_relpath(manifest_relpath)),
    media_relpath text CHECK (seeon_relpath(media_relpath)),
    thumbnail_relpath text CHECK (seeon_relpath(thumbnail_relpath)),
    manifest_sha256 text CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    media_sha256 text CHECK (media_sha256 ~ '^[0-9a-f]{64}$'),
    thumbnail_sha256 text CHECK (thumbnail_sha256 ~ '^[0-9a-f]{64}$'),
    manifest_size_bytes bigint CHECK (manifest_size_bytes > 0),
    media_size_bytes bigint CHECK (media_size_bytes > 0),
    thumbnail_size_bytes bigint CHECK (thumbnail_size_bytes > 0),
    local_state text NOT NULL CHECK (local_state IN ('AVAILABLE', 'UNAVAILABLE', 'CORRUPT')),
    local_reason text CHECK (length(local_reason) BETWEEN 1 AND 64),
    publish_state text NOT NULL CHECK (publish_state IN ('WAITING', 'PUBLISHED', 'PERMANENT', 'COMPATIBILITY')),
    published_at text CHECK (seeon_utc_timestamp(published_at)),
    last_publish_error_code text CHECK (length(last_publish_error_code) BETWEEN 1 AND 64),
    retention_state text NOT NULL CHECK (retention_state IN ('RETAINED', 'PENDING', 'PURGED')),
    retention_reason text CHECK (length(retention_reason) BETWEEN 1 AND 64),
    retention_requested_at text CHECK (seeon_utc_timestamp(retention_requested_at)),
    retention_updated_at text CHECK (seeon_utc_timestamp(retention_updated_at)),
    revision bigint NOT NULL CHECK (revision > 0),
    created_at text NOT NULL CHECK (seeon_utc_timestamp(created_at)),
    updated_at text NOT NULL CHECK (seeon_utc_timestamp(updated_at)),
    CHECK (num_nonnulls(manifest_relpath, manifest_sha256, manifest_size_bytes) IN (0, 3)),
    CHECK (num_nonnulls(media_relpath, media_sha256, media_size_bytes) IN (0, 3)),
    CHECK (num_nonnulls(thumbnail_relpath, thumbnail_sha256, thumbnail_size_bytes) IN (0, 3)),
    CHECK (
        (local_state = 'AVAILABLE' AND local_reason IS NULL
            AND manifest_relpath IS NOT NULL AND media_relpath IS NOT NULL)
        OR (local_state = 'UNAVAILABLE' AND local_reason IS NOT NULL
            AND manifest_relpath IS NULL AND media_relpath IS NULL AND thumbnail_relpath IS NULL)
        OR (local_state = 'CORRUPT' AND local_reason IS NOT NULL AND manifest_relpath IS NOT NULL)
    ),
    CHECK (
        (publish_state = 'WAITING' AND published_at IS NULL)
        OR (publish_state = 'PUBLISHED' AND published_at IS NOT NULL AND last_publish_error_code IS NULL)
        OR (publish_state IN ('PERMANENT', 'COMPATIBILITY') AND last_publish_error_code IS NULL)
    ),
    CHECK (
        (retention_state = 'RETAINED' AND retention_reason IS NULL
            AND retention_requested_at IS NULL AND retention_updated_at IS NULL)
        OR (retention_state = 'PENDING' AND retention_requested_at IS NOT NULL AND retention_updated_at IS NOT NULL)
        OR (retention_state = 'PURGED' AND retention_reason IS NOT NULL
            AND retention_requested_at IS NOT NULL AND retention_updated_at IS NOT NULL AND media_relpath IS NULL)
    )
);
CREATE INDEX clips_started_at_idx ON clips(started_at DESC, clip_id DESC);
CREATE INDEX clips_camera_started_at_idx ON clips(camera_id, started_at DESC, clip_id DESC);
CREATE INDEX clips_facet_started_at_idx ON clips(event_facet, started_at DESC, clip_id DESC);
CREATE INDEX clips_retention_idx ON clips(retention_state, clip_id);

CREATE TABLE incidents (
    incident_id text PRIMARY KEY CHECK (length(incident_id) BETWEEN 1 AND 128),
    edge_event_id text NOT NULL UNIQUE CHECK (length(edge_event_id) BETWEEN 1 AND 128),
    facility_id text NOT NULL CHECK (length(facility_id) BETWEEN 1 AND 128),
    camera_id text NOT NULL CHECK (length(camera_id) BETWEEN 1 AND 128),
    event_type text NOT NULL CHECK (length(event_type) BETWEEN 1 AND 64),
    probability double precision CHECK (probability >= 0 AND probability <= 1),
    detected_at text NOT NULL CHECK (seeon_utc_timestamp(detected_at)),
    lifecycle_state text NOT NULL CHECK (lifecycle_state IN ('OPEN', 'COMPLETE', 'FAILED')),
    failure_reason text CHECK (length(failure_reason) BETWEEN 1 AND 64),
    backend_event_id text CHECK (length(backend_event_id) BETWEEN 1 AND 128),
    runtime_manifest_sha256 text CHECK (runtime_manifest_sha256 ~ '^[0-9a-f]{64}$'),
    module_qualified_id text CHECK (length(module_qualified_id) BETWEEN 1 AND 128),
    policy_qualified_id text CHECK (length(policy_qualified_id) BETWEEN 1 AND 128),
    provenance_state text NOT NULL CHECK (provenance_state IN ('QUALIFIED', 'MISSING')),
    provenance_missing_reason text CHECK (length(provenance_missing_reason) BETWEEN 1 AND 64),
    review_version bigint NOT NULL CHECK (review_version >= 0),
    review_disposition text CHECK (review_disposition IN ('TP', 'FP')),
    review_actor text CHECK (length(review_actor) BETWEEN 1 AND 128),
    review_at text CHECK (seeon_utc_timestamp(review_at)),
    review_notes text CHECK (length(review_notes) BETWEEN 1 AND 1000),
    revision bigint NOT NULL CHECK (revision > 0),
    created_at text NOT NULL CHECK (seeon_utc_timestamp(created_at)),
    updated_at text NOT NULL CHECK (seeon_utc_timestamp(updated_at)),
    CHECK ((lifecycle_state = 'FAILED') = (failure_reason IS NOT NULL)),
    CHECK (
        (provenance_state = 'QUALIFIED' AND provenance_missing_reason IS NULL
            AND num_nonnulls(backend_event_id, runtime_manifest_sha256, module_qualified_id, policy_qualified_id) = 4)
        OR (provenance_state = 'MISSING' AND provenance_missing_reason IS NOT NULL
            AND num_nonnulls(backend_event_id, runtime_manifest_sha256, module_qualified_id, policy_qualified_id) = 0)
    ),
    CHECK (
        (review_version = 0 AND num_nonnulls(review_disposition, review_actor, review_at, review_notes) = 0)
        OR (review_version > 0 AND num_nonnulls(review_disposition, review_actor, review_at) = 3)
    )
);

CREATE FUNCTION seeon_incident_update() RETURNS trigger
LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
BEGIN
    IF NEW.lifecycle_state IS DISTINCT FROM OLD.lifecycle_state AND NOT (
        (OLD.lifecycle_state = 'OPEN' AND NEW.lifecycle_state IN ('COMPLETE', 'FAILED'))
        OR (OLD.lifecycle_state = 'COMPLETE' AND NEW.lifecycle_state = 'FAILED')
    ) THEN
        RAISE EXCEPTION 'illegal incident lifecycle transition' USING ERRCODE = '23514';
    END IF;
    IF ROW(NEW.incident_id, NEW.edge_event_id, NEW.facility_id, NEW.camera_id, NEW.event_type,
        NEW.detected_at, NEW.backend_event_id, NEW.runtime_manifest_sha256,
        NEW.module_qualified_id, NEW.policy_qualified_id, NEW.provenance_state,
        NEW.provenance_missing_reason, NEW.created_at)
        IS DISTINCT FROM
        ROW(OLD.incident_id, OLD.edge_event_id, OLD.facility_id, OLD.camera_id, OLD.event_type,
        OLD.detected_at, OLD.backend_event_id, OLD.runtime_manifest_sha256,
        OLD.module_qualified_id, OLD.policy_qualified_id, OLD.provenance_state,
        OLD.provenance_missing_reason, OLD.created_at) THEN
        RAISE EXCEPTION 'incident identity and provenance are immutable' USING ERRCODE = '23514';
    END IF;
    IF NEW.revision IS DISTINCT FROM OLD.revision + 1 THEN
        RAISE EXCEPTION 'incident revision must advance exactly once' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER incidents_update_guard BEFORE UPDATE ON incidents
    FOR EACH ROW EXECUTE FUNCTION seeon_incident_update();

CREATE TABLE artifacts (
    incident_id text NOT NULL CHECK (length(incident_id) BETWEEN 1 AND 128),
    kind text NOT NULL CHECK (kind IN ('PRIMARY_CLIP', 'SNAPSHOT')),
    artifact_id text UNIQUE CHECK (length(artifact_id) BETWEEN 1 AND 128),
    clip_id text CHECK (length(clip_id) BETWEEN 1 AND 128),
    state text NOT NULL CHECK (state IN ('PENDING', 'AVAILABLE', 'UNAVAILABLE', 'CORRUPT', 'PURGED')),
    reason text CHECK (length(reason) BETWEEN 1 AND 64),
    contained_relpath text CHECK (seeon_relpath(contained_relpath)),
    content_sha256 text CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes bigint CHECK (size_bytes > 0),
    mime_type text CHECK (length(mime_type) BETWEEN 1 AND 128),
    codec text CHECK (length(codec) BETWEEN 1 AND 64),
    captured_at text CHECK (seeon_utc_timestamp(captured_at)),
    revision bigint NOT NULL CHECK (revision > 0),
    created_at text NOT NULL CHECK (seeon_utc_timestamp(created_at)),
    updated_at text NOT NULL CHECK (seeon_utc_timestamp(updated_at)),
    PRIMARY KEY (incident_id, kind),
    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (clip_id) REFERENCES clips(clip_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK ((kind = 'PRIMARY_CLIP' AND captured_at IS NULL)
        OR (kind = 'SNAPSHOT' AND clip_id IS NULL AND captured_at IS NOT NULL)),
    CHECK (
        (state = 'PENDING' AND num_nonnulls(artifact_id, clip_id, reason, contained_relpath,
            content_sha256, size_bytes, mime_type, codec) = 0)
        OR (state = 'AVAILABLE' AND reason IS NULL
            AND num_nonnulls(artifact_id, contained_relpath, content_sha256, size_bytes, mime_type) = 5)
        OR (state = 'UNAVAILABLE' AND reason IS NOT NULL
            AND num_nonnulls(contained_relpath, content_sha256, size_bytes, mime_type, codec) = 0)
        OR (state = 'CORRUPT' AND reason IS NOT NULL
            AND num_nonnulls(artifact_id, contained_relpath, content_sha256, size_bytes, mime_type) = 5)
        OR (state = 'PURGED' AND artifact_id IS NOT NULL AND reason IS NOT NULL
            AND contained_relpath IS NULL AND num_nonnulls(content_sha256, size_bytes, mime_type) IN (0, 3))
    )
);

CREATE FUNCTION seeon_artifact_update() RETURNS trigger
LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
BEGIN
    IF NEW.state IS DISTINCT FROM OLD.state AND NOT (
        (OLD.state = 'PENDING' AND NEW.state IN ('AVAILABLE', 'UNAVAILABLE', 'CORRUPT'))
        OR (OLD.state = 'AVAILABLE' AND NEW.state IN ('CORRUPT', 'PURGED'))
        OR (OLD.state = 'UNAVAILABLE' AND NEW.state IN ('AVAILABLE', 'PURGED'))
        OR (OLD.state = 'CORRUPT' AND NEW.state = 'PURGED')
    ) THEN
        RAISE EXCEPTION 'illegal artifact state transition' USING ERRCODE = '23514';
    END IF;
    IF OLD.state = 'AVAILABLE' AND NEW.state = 'CORRUPT' AND
        ROW(NEW.contained_relpath, NEW.content_sha256, NEW.size_bytes, NEW.mime_type,
            NEW.codec, NEW.captured_at, NEW.clip_id)
        IS DISTINCT FROM
        ROW(OLD.contained_relpath, OLD.content_sha256, OLD.size_bytes, OLD.mime_type,
            OLD.codec, OLD.captured_at, OLD.clip_id) THEN
        RAISE EXCEPTION 'artifact retained identity is immutable' USING ERRCODE = '23514';
    END IF;
    IF ROW(NEW.incident_id, NEW.kind, NEW.created_at)
        IS DISTINCT FROM ROW(OLD.incident_id, OLD.kind, OLD.created_at)
        OR (OLD.artifact_id IS NOT NULL AND NEW.artifact_id IS DISTINCT FROM OLD.artifact_id) THEN
        RAISE EXCEPTION 'artifact identity is immutable' USING ERRCODE = '23514';
    END IF;
    IF NEW.revision IS DISTINCT FROM OLD.revision + 1 THEN
        RAISE EXCEPTION 'artifact revision must advance exactly once' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER artifacts_update_guard BEFORE UPDATE ON artifacts
    FOR EACH ROW EXECUTE FUNCTION seeon_artifact_update();

CREATE TABLE audit_events (
    audit_id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    occurred_at text NOT NULL CHECK (seeon_utc_timestamp(occurred_at)),
    recorded_at text NOT NULL CHECK (seeon_utc_timestamp(recorded_at)),
    clock_quality text NOT NULL CHECK (clock_quality IN ('trusted', 'untrusted', 'unknown')),
    actor_type text NOT NULL CHECK (actor_type IN ('user', 'service', 'system')),
    actor_id text NOT NULL CHECK (length(actor_id) BETWEEN 1 AND 128),
    auth_mechanism text NOT NULL CHECK (length(auth_mechanism) BETWEEN 1 AND 64),
    action text NOT NULL CHECK (length(action) BETWEEN 1 AND 64),
    target_type text NOT NULL CHECK (length(target_type) BETWEEN 1 AND 64),
    target_id text NOT NULL CHECK (length(target_id) BETWEEN 1 AND 128),
    outcome text NOT NULL CHECK (outcome IN ('success', 'denied', 'failed')),
    reason text CHECK (length(reason) BETWEEN 1 AND 256),
    request_id text CHECK (length(request_id) BETWEEN 1 AND 128),
    interaction_id text CHECK (length(interaction_id) BETWEEN 1 AND 128),
    detail_json text CHECK (detail_json IS NULL OR (
        detail_json IS JSON OBJECT AND octet_length(detail_json) BETWEEN 2 AND 16384)),
    previous_hash text NOT NULL CHECK (previous_hash ~ '^[0-9a-f]{64}$'),
    record_hash text NOT NULL UNIQUE CHECK (record_hash ~ '^[0-9a-f]{64}$'),
    retention_class text NOT NULL CHECK (retention_class IN ('standard', 'legal_hold')),
    hold_reference text CHECK (length(hold_reference) BETWEEN 1 AND 128),
    CHECK ((retention_class = 'standard' AND hold_reference IS NULL)
        OR (retention_class = 'legal_hold' AND hold_reference IS NOT NULL))
);

-- Every hashed payload value is text or null, including detail_json. This is
-- Python json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(',', ':')),
-- not jsonb's key order/spacing, and not recursively decoded detail_json.
CREATE FUNCTION seeon_audit_record_hash(previous_hash text, payload_json text) RETURNS text
LANGUAGE plpgsql IMMUTABLE STRICT SET search_path = pg_catalog AS $$
DECLARE
    canonical pg_catalog.text;
BEGIN
    IF previous_hash !~ '^[0-9a-f]{64}$' OR NOT (payload_json IS JSON OBJECT WITH UNIQUE KEYS) THEN
        RAISE EXCEPTION 'invalid audit hash input' USING ERRCODE = '23514';
    END IF;
    IF EXISTS (SELECT 1 FROM json_each(payload_json::pg_catalog.json) WHERE json_typeof(value) NOT IN ('string', 'null')) THEN
        RAISE EXCEPTION 'audit hash values must be strings or null' USING ERRCODE = '23514';
    END IF;
    SELECT '{' || coalesce(string_agg(to_json(key)::pg_catalog.text || ':' || coalesce(to_json(value)::pg_catalog.text, 'null'),
        ',' ORDER BY key COLLATE pg_catalog."C"), '') || '}'
        INTO canonical FROM json_each_text(payload_json::pg_catalog.json);
    RETURN encode(sha256(decode(previous_hash, 'hex') || convert_to(canonical, 'UTF8')), 'hex');
END;
$$;

CREATE FUNCTION seeon_audit_immutable() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
    RAISE EXCEPTION 'audit events are immutable' USING ERRCODE = '23514';
END;
$$;
CREATE TRIGGER audit_events_immutable_update BEFORE UPDATE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION seeon_audit_immutable();
CREATE TRIGGER audit_events_immutable_delete BEFORE DELETE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION seeon_audit_immutable();
CREATE TRIGGER audit_events_immutable_truncate BEFORE TRUNCATE ON audit_events
    FOR EACH STATEMENT EXECUTE FUNCTION seeon_audit_immutable();

CREATE FUNCTION seeon_audit_serialize() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
    -- Statement scope takes the transaction lock before identity allocation.
    -- No lock upgrade from ROW EXCLUSIVE to EXCLUSIVE (which could deadlock).
    PERFORM pg_advisory_xact_lock(TG_RELID::bigint);
    RETURN NULL;
END;
$$;
CREATE TRIGGER audit_events_serialize BEFORE INSERT ON audit_events
    FOR EACH STATEMENT EXECUTE FUNCTION seeon_audit_serialize();

CREATE FUNCTION seeon_audit_insert() RETURNS trigger
LANGUAGE plpgsql SET search_path FROM CURRENT AS $$
DECLARE
    tail_hash pg_catalog.text;
    tail_id pg_catalog.int8;
    expected_hash pg_catalog.text;
BEGIN
    SELECT record_hash, audit_id INTO tail_hash, tail_id FROM audit_events ORDER BY audit_id DESC LIMIT 1;
    IF (NEW.previous_hash OPERATOR(pg_catalog.=) coalesce(tail_hash, pg_catalog.repeat('0', 64))) IS NOT TRUE THEN
        RAISE EXCEPTION 'audit hash chain is invalid' USING ERRCODE = '23514';
    END IF;
    IF tail_id IS NOT NULL AND NEW.audit_id OPERATOR(pg_catalog.<=) tail_id THEN
        RAISE EXCEPTION 'audit identity must advance' USING ERRCODE = '23514';
    END IF;
    IF (SELECT pg_catalog.count(*) FROM audit_events) OPERATOR(pg_catalog.>=) 1000000 THEN
        RAISE EXCEPTION 'audit capacity exhausted' USING ERRCODE = '23514';
    END IF;
    expected_hash := seeon_audit_record_hash(NEW.previous_hash, pg_catalog.json_build_object(
        'action', NEW.action,
        'actor_id', NEW.actor_id,
        'actor_type', NEW.actor_type,
        'auth_mechanism', NEW.auth_mechanism,
        'clock_quality', NEW.clock_quality,
        'detail_json', NEW.detail_json,
        'hold_reference', NEW.hold_reference,
        'interaction_id', NEW.interaction_id,
        'occurred_at', NEW.occurred_at,
        'outcome', NEW.outcome,
        'previous_hash', NEW.previous_hash,
        'reason', NEW.reason,
        'recorded_at', NEW.recorded_at,
        'request_id', NEW.request_id,
        'retention_class', NEW.retention_class,
        'target_id', NEW.target_id,
        'target_type', NEW.target_type
    )::pg_catalog.text);
    IF (NEW.record_hash OPERATOR(pg_catalog.=) expected_hash) IS NOT TRUE THEN
        RAISE EXCEPTION 'audit record hash is invalid' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER audit_events_insert_guard BEFORE INSERT ON audit_events
    FOR EACH ROW EXECUTE FUNCTION seeon_audit_insert();
-- Also prevents a fork from an old REPEATABLE READ snapshot after lock wait.
CREATE UNIQUE INDEX audit_events_one_successor_idx ON audit_events(previous_hash);
CREATE INDEX audit_events_recorded_idx ON audit_events(recorded_at, audit_id);
CREATE INDEX audit_events_target_idx ON audit_events(target_type, target_id, recorded_at);
CREATE INDEX audit_events_actor_idx ON audit_events(actor_type, actor_id, recorded_at);
