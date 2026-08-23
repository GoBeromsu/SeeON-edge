"""Canonical application DDL for the single edge database.

These statements intentionally mirror the last released catalog, connection,
and worker-state schemas. Runtime packages validate and use them but never
execute them.
"""

from typing import Final

APPLICATION_SCHEMA_STATEMENTS: Final = (
    (
        "CREATE TABLE clips (clip_id TEXT PRIMARY KEY, camera_id TEXT, event_type"
        " TEXT, state TEXT, started_at TEXT, path TEXT, sha256 TEXT, size_bytes I"
        "NTEGER, mime_type TEXT, encoder TEXT, payload_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE snapshots (snapshot_id TEXT PRIMARY KEY, camera_id TEXT, ed"
        "ge_event_id TEXT, captured_at TEXT, path TEXT, sha256 TEXT, size_bytes I"
        "NTEGER, mime_type TEXT, payload_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE events (edge_event_id TEXT PRIMARY KEY, camera_id TEXT, eve"
        "nt_type TEXT, detected_at TEXT, clip_id TEXT, payload_json TEXT NOT NULL"
        ") STRICT"
    ),
    (
        "CREATE TABLE labels (clip_id TEXT PRIMARY KEY, label TEXT, reviewer TEXT"
        ", reviewed_at TEXT, payload_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE cameras (camera_id TEXT PRIMARY KEY, label TEXT, decode_bac"
        "kend TEXT, payload_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE audit (audit_id TEXT PRIMARY KEY, occurred_at TEXT, action "
        "TEXT, payload_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE credentials (id INTEGER PRIMARY KEY CHECK (id = 1), usernam"
        "e TEXT NOT NULL, algorithm TEXT NOT NULL, salt BLOB NOT NULL, password_h"
        "ash BLOB NOT NULL, updated_at TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE camera_registry (id INTEGER PRIMARY KEY CHECK (id = 1), reg"
        "istry_version INTEGER NOT NULL, cameras_json TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE runtime_latency (facility_id TEXT PRIMARY KEY, payload_json"
        " TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE detection_settings (domain TEXT PRIMARY KEY, on_flag INTEGE"
        "R NOT NULL, mode TEXT NOT NULL, start_time TEXT, end_time TEXT) STRICT"
    ),
    (
        "CREATE TABLE camera_bed_zone (camera_id TEXT PRIMARY KEY, polygon_json T"
        "EXT NOT NULL, image_width INTEGER NOT NULL, image_height INTEGER NOT NUL"
        "L, recognized_at TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE camera_topology_floors (edge_ref TEXT PRIMARY KEY, name TEX"
        "T NOT NULL, order_index INTEGER NOT NULL CHECK (order_index >= 0)) STRIC"
        "T"
    ),
    (
        "CREATE TABLE camera_topology_rooms (edge_ref TEXT PRIMARY KEY, floor_edg"
        "e_ref TEXT NOT NULL, name TEXT NOT NULL, room_type TEXT NOT NULL CHECK ("
        "room_type = 'ROOM'), capacity INTEGER NOT NULL CHECK (capacity > 0), leg"
        "acy_canonical_space_id TEXT UNIQUE, FOREIGN KEY (floor_edge_ref) REFEREN"
        "CES camera_topology_floors(edge_ref) ON UPDATE RESTRICT ON DELETE RESTRI"
        "CT) STRICT"
    ),
    (
        "CREATE TABLE camera_topology_cameras (camera_id TEXT PRIMARY KEY, edge_r"
        "ef TEXT NOT NULL UNIQUE, room_edge_ref TEXT NOT NULL UNIQUE, FOREIGN KEY"
        " (room_edge_ref) REFERENCES camera_topology_rooms(edge_ref) ON UPDATE RE"
        "STRICT ON DELETE RESTRICT) STRICT"
    ),
    (
        "CREATE TABLE topology_dirty (id INTEGER PRIMARY KEY CHECK (id = 1), regi"
        "stry_version INTEGER NOT NULL CHECK (registry_version >= 1), created_at "
        "TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE edge_topology_sync_state (\n      id INTEGER PRIMARY KEY CHE"
        "CK (id = 1), edge_installation_id TEXT,\n      enrollment_generation INTE"
        "GER CHECK (enrollment_generation IS NULL OR enrollment_generation > 0),\n"
        "      last_snapshotted_registry_version INTEGER NOT NULL DEFAULT 0 CHECK"
        " (last_snapshotted_registry_version >= 0),\n      last_client_revision IN"
        "TEGER NOT NULL DEFAULT 0 CHECK (last_client_revision >= 0),\n      server"
        "_revision INTEGER NOT NULL DEFAULT 0 CHECK (server_revision >= 0),\n     "
        " pending_snapshot_id TEXT, pending_body BLOB,\n      pending_registry_ver"
        "sion INTEGER CHECK (pending_registry_version IS NULL OR pending_registry"
        "_version >= 0),\n      pending_client_revision INTEGER CHECK (pending_cli"
        "ent_revision IS NULL OR pending_client_revision > 0),\n      pending_expe"
        "cted_server_revision INTEGER CHECK (pending_expected_server_revision IS "
        "NULL OR pending_expected_server_revision >= 0),\n      consecutive_failur"
        "es INTEGER NOT NULL DEFAULT 0 CHECK (consecutive_failures >= 0),\n      n"
        "ext_retry_at REAL,\n      pause_reason TEXT CHECK (pause_reason IS NULL O"
        "R pause_reason IN ('auth', 'forbidden', 'conflict')),\n      last_accepte"
        "d_at REAL\n    ) STRICT"
    ),
    (
        "CREATE TABLE edge_topology_confirmation_preview (\n      id INTEGER PRIMA"
        "RY KEY CHECK (id = 1), confirmation_id TEXT NOT NULL, digest TEXT NOT NU"
        "LL,\n      expires_at TEXT NOT NULL, snapshot_id TEXT NOT NULL, client_re"
        "vision INTEGER NOT NULL,\n      server_revision INTEGER NOT NULL, registr"
        "y_version INTEGER NOT NULL,\n      edge_installation_id TEXT NOT NULL, en"
        "rollment_generation INTEGER NOT NULL,\n      cameras INTEGER NOT NULL, ro"
        "oms INTEGER NOT NULL, floors INTEGER NOT NULL,\n      confirmed INTEGER N"
        "OT NULL DEFAULT 0, terminal_response TEXT\n    ) STRICT"
    ),
    (
        "CREATE TABLE clip_storage_location (id INTEGER PRIMARY KEY CHECK (id = 1"
        "), selected_path TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE clip_listing_generation (id INTEGER PRIMARY KEY CHECK (id ="
        " 1), active_generation INTEGER NOT NULL, next_generation INTEGER NOT NUL"
        "L) STRICT"
    ),
    (
        "CREATE TABLE clip_listing_rows (\n      generation INTEGER NOT NULL, clip"
        "_id TEXT NOT NULL, manifest_path TEXT NOT NULL,\n      manifest_mtime_ns "
        "INTEGER NOT NULL, manifest_size_bytes INTEGER NOT NULL,\n      camera_id "
        "TEXT NOT NULL, event_ref TEXT NOT NULL, event_type TEXT,\n      event_fac"
        "et TEXT NOT NULL CHECK (event_facet IN ('fall','bed-exit','other')),\n   "
        "   started_at TEXT NOT NULL, duration_s REAL NOT NULL, codec TEXT NOT NU"
        "LL,\n      media_path TEXT, video_available INTEGER NOT NULL, video_error"
        " TEXT,\n      finalized INTEGER NOT NULL, size_bytes INTEGER,\n      PRIMA"
        "RY KEY (generation, clip_id), UNIQUE (generation, manifest_path)\n    ) S"
        "TRICT"
    ),
    (
        "CREATE TABLE clip_listing_thumbnails (generation INTEGER NOT NULL, clip_"
        "id TEXT NOT NULL, thumbnail_mtime_ns INTEGER, thumbnail_size_bytes INTEG"
        "ER, thumbnail_available INTEGER NOT NULL, PRIMARY KEY (generation, clip_"
        "id)) STRICT"
    ),
    (
        "CREATE TABLE clip_listing_summary (generation INTEGER NOT NULL, camera_i"
        "d TEXT NOT NULL, event_facet TEXT NOT NULL, count INTEGER NOT NULL, PRIM"
        "ARY KEY (generation, camera_id, event_facet)) STRICT"
    ),
    (
        "CREATE TABLE connection_settings (\n      id INTEGER PRIMARY KEY CHECK (i"
        "d = 1), events_url TEXT, config_url TEXT,\n      facility_id TEXT, facili"
        "ty_token TEXT, updated_at TEXT, facility_code TEXT,\n      client_install"
        "ation_ref TEXT, edge_installation_id TEXT,\n      enrollment_generation I"
        "NTEGER CHECK (enrollment_generation > 0),\n      enrollment_created_at TE"
        "XT, enrollment_updated_at TEXT\n    ) STRICT"
    ),
    (
        "CREATE TABLE connection_store_migrations (version INTEGER PRIMARY KEY, n"
        "ame TEXT NOT NULL, applied_at TEXT NOT NULL, backup_filename TEXT, backu"
        "p_sha256 TEXT, backup_size_bytes INTEGER) STRICT"
    ),
    (
        "CREATE TABLE evidence_events (\n      edge_event_id TEXT PRIMARY KEY, det"
        "ected_at TEXT NOT NULL, payload_json TEXT NOT NULL,\n      state TEXT NOT"
        " NULL CHECK (state IN ('STAGED', 'READY', 'IN_FLIGHT', 'ACKED')),\n      "
        "queued_at REAL NOT NULL, next_attempt_at REAL NOT NULL,\n      attempt_co"
        "unt INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),\n      lease_o"
        "wner TEXT, lease_expires_at REAL,\n      delivery_state TEXT NOT NULL DEF"
        "AULT 'PENDING' CHECK (delivery_state IN ('PENDING', 'ACKED', 'PERMANENT'"
        ", 'COMPATIBILITY')),\n      backend_event_id TEXT, last_error_code TEXT,\n"
        "      operator_only INTEGER NOT NULL DEFAULT 0 CHECK (operator_only IN ("
        "0, 1)),\n      CHECK ((state = 'IN_FLIGHT' AND lease_owner IS NOT NULL AN"
        "D lease_expires_at IS NOT NULL)\n          OR (state != 'IN_FLIGHT' AND l"
        "ease_owner IS NULL AND lease_expires_at IS NULL))\n    ) STRICT"
    ),
    (
        "CREATE TABLE evidence_clips (\n      clip_id TEXT PRIMARY KEY,\n      loca"
        "l_state TEXT NOT NULL DEFAULT 'AWAITING_FINALIZE' CHECK (local_state IN "
        "('AWAITING_FINALIZE', 'VERIFIED', 'UNAVAILABLE', 'CORRUPT')),\n      mani"
        "fest_path TEXT, state_version INTEGER NOT NULL DEFAULT 1 CHECK (state_ve"
        "rsion >= 1),\n      media_relpath TEXT, sha256 TEXT, size_bytes INTEGER C"
        "HECK (size_bytes IS NULL OR size_bytes > 0),\n      mime_type TEXT, codec"
        " TEXT, duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms BET"
        "WEEN 1 AND 120000),\n      clip_start_at TEXT, clip_end_at TEXT, finalize"
        "d_at TEXT,\n      unavailable_reason TEXT CHECK (unavailable_reason IS NU"
        "LL OR unavailable_reason IN ('ENCODER_FAILED','NO_FRAMES','FINALIZE_FAIL"
        "ED','INTERRUPTED_FINALIZE','MISSING','CORRUPT')),\n      publish_state TE"
        "XT NOT NULL DEFAULT 'WAITING' CHECK (publish_state IN ('WAITING','IN_FLI"
        "GHT','PUBLISHED','PERMANENT','COMPATIBILITY')),\n      publish_attempt_co"
        "unt INTEGER NOT NULL DEFAULT 0 CHECK (publish_attempt_count >= 0),\n     "
        " publish_next_attempt_at REAL NOT NULL DEFAULT 0, publish_lease_owner TE"
        "XT,\n      publish_lease_expires_at REAL, remote_state TEXT, backend_ack_"
        "at REAL, last_error_code TEXT\n    ) STRICT"
    ),
    (
        "CREATE TABLE clip_events (clip_id TEXT NOT NULL REFERENCES evidence_clip"
        "s(clip_id) ON DELETE RESTRICT, edge_event_id TEXT NOT NULL UNIQUE REFERE"
        "NCES evidence_events(edge_event_id) ON DELETE RESTRICT, ordinal INTEGER "
        "NOT NULL CHECK (ordinal >= 0), PRIMARY KEY (clip_id, ordinal)) STRICT"
    ),
    (
        "CREATE TABLE config_current (id INTEGER PRIMARY KEY CHECK (id = 1), gene"
        "ration INTEGER NOT NULL CHECK (generation >= 0), config_version INTEGER "
        "NOT NULL CHECK (config_version >= 0), registry_version INTEGER NOT NULL "
        "CHECK (registry_version >= 0), payload_json TEXT NOT NULL, saved_at REAL"
        " NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE config_history (config_version INTEGER PRIMARY KEY CHECK (c"
        "onfig_version >= 0), generation INTEGER NOT NULL CHECK (generation >= 0)"
        ", registry_version INTEGER NOT NULL CHECK (registry_version >= 0), paylo"
        "ad_json TEXT NOT NULL, saved_at REAL NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE faults (id INTEGER PRIMARY KEY CHECK (id = 1), pid INTEGER "
        "NOT NULL, boot_time_iso TEXT NOT NULL, profile TEXT NOT NULL, task TEXT "
        "NOT NULL, stage TEXT NOT NULL, camera_id TEXT NOT NULL, frame_index INTE"
        "GER, pts REAL, frame_shape_json TEXT, frame_hash_sha256 TEXT, model_arti"
        "fact_digest TEXT, invocation_seq INTEGER NOT NULL, exception_type TEXT N"
        "OT NULL, exception_message TEXT NOT NULL, exit_code INTEGER NOT NULL, ac"
        "tion TEXT NOT NULL, fault_time_iso TEXT NOT NULL) STRICT"
    ),
    (
        "CREATE TABLE system_test_runs (validation_run_id TEXT PRIMARY KEY, edge_"
        "event_id TEXT NOT NULL UNIQUE REFERENCES evidence_events(edge_event_id) "
        "ON DELETE RESTRICT) STRICT"
    ),
    (
        "CREATE TABLE control_heartbeats (camera_id TEXT PRIMARY KEY, facility_id"
        " TEXT NOT NULL, received_at REAL NOT NULL, config_version INTEGER) STRIC"
        "T"
    ),
    "CREATE INDEX clips_camera_started_at_idx ON clips(camera_id, started_at)",
    "CREATE INDEX clips_event_type_idx ON clips(event_type)",
    ("CREATE INDEX snapshots_camera_captured_at_idx ON snapshots(camera_id, captured_at)"),
    ("CREATE INDEX events_camera_detected_at_idx ON events(camera_id, detected_at)"),
    (
        "CREATE INDEX clip_listing_global_order_idx ON clip_listing_rows(generati"
        "on, started_at DESC, clip_id DESC)"
    ),
    (
        "CREATE INDEX clip_listing_global_facet_order_idx ON clip_listing_rows(ge"
        "neration, event_facet, started_at DESC, clip_id DESC)"
    ),
    (
        "CREATE INDEX clip_listing_camera_order_idx ON clip_listing_rows(generati"
        "on, camera_id, started_at DESC, clip_id DESC)"
    ),
    (
        "CREATE INDEX clip_listing_camera_facet_order_idx ON clip_listing_rows(ge"
        "neration, camera_id, event_facet, started_at DESC, clip_id DESC)"
    ),
    (
        "CREATE INDEX evidence_events_claim_idx ON evidence_events(state, next_at"
        "tempt_at, lease_expires_at, queued_at, edge_event_id)"
    ),
    (
        "CREATE INDEX evidence_events_delivery_idx ON evidence_events(delivery_st"
        "ate, state, next_attempt_at, lease_expires_at)"
    ),
    (
        "CREATE INDEX evidence_events_operator_claim_idx ON evidence_events(opera"
        "tor_only, delivery_state, state, next_attempt_at, lease_expires_at, edge"
        "_event_id)"
    ),
    ("CREATE INDEX evidence_clips_local_state_idx ON evidence_clips(local_state, clip_id)"),
    (
        "CREATE INDEX evidence_clips_publish_idx ON evidence_clips(publish_state,"
        " local_state, publish_next_attempt_at, publish_lease_expires_at, clip_id"
        ")"
    ),
    ("CREATE INDEX config_history_saved_at_idx ON config_history(saved_at, config_version)"),
    (
        "CREATE TABLE schema_import_receipts (\n      source_name TEXT NOT NULL, b"
        "arrier TEXT NOT NULL, source_schema TEXT NOT NULL,\n      digest TEXT NOT"
        " NULL CHECK (length(digest) = 64), row_count INTEGER NOT NULL,\n      rec"
        "orded_at TEXT NOT NULL, PRIMARY KEY (source_name, barrier)\n    ) STRICT"
    ),
    (
        "CREATE TABLE schema_import_sources (\n      source_name TEXT PRIMARY KEY,"
        " source_schema TEXT NOT NULL,\n      source_sha256 TEXT NOT NULL CHECK (l"
        "ength(source_sha256) = 64),\n      source_size_bytes INTEGER NOT NULL, ta"
        "ble_count INTEGER NOT NULL,\n      row_count INTEGER NOT NULL, completed_"
        "at TEXT NOT NULL\n    ) STRICT"
    ),
)

__all__ = ["APPLICATION_SCHEMA_STATEMENTS"]
