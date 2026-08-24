"""CREATE statements for the schema 18 ten-table contract."""

# noqa: SIZE_OK — pure DDL ledger; splitting it would hide the contract.

from __future__ import annotations

from typing import Final

COMPACT_SCHEMA_CREATE_STATEMENTS: Final = (
    """
    CREATE TABLE credentials (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        username TEXT NOT NULL CHECK (
            length(username) BETWEEN 1 AND 128 AND instr(username, char(0)) = 0
        ),
        algorithm TEXT NOT NULL CHECK (algorithm = 'scrypt'),
        salt BLOB NOT NULL CHECK (length(salt) = 16),
        password_hash BLOB NOT NULL CHECK (length(password_hash) = 64),
        updated_at TEXT NOT NULL CHECK (
            length(updated_at) BETWEEN 20 AND 30
            AND instr(updated_at, char(0)) = 0
            AND substr(updated_at, 5, 1) = '-'
            AND substr(updated_at, 8, 1) = '-'
            AND substr(updated_at, 11, 1) = 'T'
            AND substr(updated_at, 14, 1) = ':'
            AND substr(updated_at, 17, 1) = ':'
            AND substr(updated_at, -1) = 'Z'
            AND datetime(substr(updated_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(updated_at, 1, 19))
                = substr(updated_at, 1, 19)
            AND (
                length(updated_at) = 20
                OR (
                    substr(updated_at, 20, 1) = '.'
                    AND length(updated_at) BETWEEN 22 AND 27
                    AND substr(updated_at, 21, length(updated_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        )
    ) STRICT
    """,
    """
    CREATE TABLE edge_site (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        facility_code TEXT CHECK (
            facility_code IS NULL
            OR (length(facility_code) BETWEEN 1 AND 64 AND instr(facility_code, char(0)) = 0)
        ),
        client_installation_ref TEXT CHECK (
            client_installation_ref IS NULL
            OR (
                length(client_installation_ref) BETWEEN 1 AND 128
                AND instr(client_installation_ref, char(0)) = 0
            )
        ),
        facility_id TEXT CHECK (
            facility_id IS NULL
            OR (length(facility_id) BETWEEN 1 AND 128 AND instr(facility_id, char(0)) = 0)
        ),
        facility_token TEXT CHECK (
            facility_token IS NULL
            OR (length(facility_token) BETWEEN 1 AND 512 AND instr(facility_token, char(0)) = 0)
        ),
        edge_installation_id TEXT CHECK (
            edge_installation_id IS NULL
            OR (
                length(edge_installation_id) BETWEEN 1 AND 128
                AND instr(edge_installation_id, char(0)) = 0
            )
        ),
        enrollment_generation INTEGER CHECK (
            enrollment_generation IS NULL OR enrollment_generation > 0
        ),
        enrollment_created_at TEXT CHECK (
            enrollment_created_at IS NULL
            OR (
                length(enrollment_created_at) BETWEEN 20 AND 30
                AND instr(enrollment_created_at, char(0)) = 0
                AND substr(enrollment_created_at, 5, 1) = '-'
                AND substr(enrollment_created_at, 8, 1) = '-'
                AND substr(enrollment_created_at, 11, 1) = 'T'
                AND substr(enrollment_created_at, 14, 1) = ':'
                AND substr(enrollment_created_at, 17, 1) = ':'
                AND substr(enrollment_created_at, -1) = 'Z'
                AND datetime(substr(enrollment_created_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(enrollment_created_at, 1, 19))
                    = substr(enrollment_created_at, 1, 19)
                AND (
                    length(enrollment_created_at) = 20
                    OR (
                        substr(enrollment_created_at, 20, 1) = '.'
                        AND length(enrollment_created_at) BETWEEN 22 AND 27
                        AND substr(enrollment_created_at, 21, length(enrollment_created_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        enrollment_updated_at TEXT CHECK (
            enrollment_updated_at IS NULL
            OR (
                length(enrollment_updated_at) BETWEEN 20 AND 30
                AND instr(enrollment_updated_at, char(0)) = 0
                AND substr(enrollment_updated_at, 5, 1) = '-'
                AND substr(enrollment_updated_at, 8, 1) = '-'
                AND substr(enrollment_updated_at, 11, 1) = 'T'
                AND substr(enrollment_updated_at, 14, 1) = ':'
                AND substr(enrollment_updated_at, 17, 1) = ':'
                AND substr(enrollment_updated_at, -1) = 'Z'
                AND datetime(substr(enrollment_updated_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(enrollment_updated_at, 1, 19))
                    = substr(enrollment_updated_at, 1, 19)
                AND (
                    length(enrollment_updated_at) = 20
                    OR (
                        substr(enrollment_updated_at, 20, 1) = '.'
                        AND length(enrollment_updated_at) BETWEEN 22 AND 27
                        AND substr(enrollment_updated_at, 21, length(enrollment_updated_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        registry_version INTEGER NOT NULL DEFAULT 0 CHECK (registry_version >= 0),
        clip_store_subdir TEXT CHECK (
            clip_store_subdir IS NULL
            OR (
                length(clip_store_subdir) BETWEEN 1 AND 512
                AND instr(clip_store_subdir, char(0)) = 0
                AND substr(clip_store_subdir, 1, 1) != '/'
                AND instr(clip_store_subdir, '\\') = 0
                AND instr('/' || clip_store_subdir || '/', '/../') = 0
            )
        ),
        clip_export_enabled INTEGER NOT NULL DEFAULT 0
            CHECK (clip_export_enabled IN (0, 1)),
        runtime_settings_version INTEGER NOT NULL DEFAULT 0
            CHECK (runtime_settings_version >= 0),
        fall_on INTEGER CHECK (fall_on IS NULL OR fall_on IN (0, 1)),
        fall_mode TEXT CHECK (fall_mode IS NULL OR fall_mode IN ('always', 'window')),
        fall_start_time TEXT CHECK (
            fall_start_time IS NULL
            OR (length(fall_start_time) = 5 AND instr(fall_start_time, char(0)) = 0 
            AND substr(fall_start_time, 3, 1) = ':'
            AND substr(fall_start_time, 1, 2) GLOB '[0-9][0-9]'
            AND substr(fall_start_time, 4, 2) GLOB '[0-9][0-9]'
            AND CAST(substr(fall_start_time, 1, 2) AS INTEGER) BETWEEN 0 AND 23
            AND CAST(substr(fall_start_time, 4, 2) AS INTEGER) BETWEEN 0 AND 59)
        ),
        fall_end_time TEXT CHECK (
            fall_end_time IS NULL
            OR (length(fall_end_time) = 5 AND instr(fall_end_time, char(0)) = 0 
            AND substr(fall_end_time, 3, 1) = ':'
            AND substr(fall_end_time, 1, 2) GLOB '[0-9][0-9]'
            AND substr(fall_end_time, 4, 2) GLOB '[0-9][0-9]'
            AND CAST(substr(fall_end_time, 1, 2) AS INTEGER) BETWEEN 0 AND 23
            AND CAST(substr(fall_end_time, 4, 2) AS INTEGER) BETWEEN 0 AND 59)
        ),
        bed_exit_on INTEGER CHECK (bed_exit_on IS NULL OR bed_exit_on IN (0, 1)),
        bed_exit_mode TEXT CHECK (bed_exit_mode IS NULL OR bed_exit_mode IN ('always', 'window')),
        bed_exit_start_time TEXT CHECK (
            bed_exit_start_time IS NULL
            OR (length(bed_exit_start_time) = 5 AND instr(bed_exit_start_time, char(0)) = 0 
            AND substr(bed_exit_start_time, 3, 1) = ':'
            AND substr(bed_exit_start_time, 1, 2) GLOB '[0-9][0-9]'
            AND substr(bed_exit_start_time, 4, 2) GLOB '[0-9][0-9]'
            AND CAST(substr(bed_exit_start_time, 1, 2) AS INTEGER) BETWEEN 0 AND 23
            AND CAST(substr(bed_exit_start_time, 4, 2) AS INTEGER) BETWEEN 0 AND 59)
        ),
        bed_exit_end_time TEXT CHECK (
            bed_exit_end_time IS NULL
            OR (length(bed_exit_end_time) = 5 AND instr(bed_exit_end_time, char(0)) = 0 
            AND substr(bed_exit_end_time, 3, 1) = ':'
            AND substr(bed_exit_end_time, 1, 2) GLOB '[0-9][0-9]'
            AND substr(bed_exit_end_time, 4, 2) GLOB '[0-9][0-9]'
            AND CAST(substr(bed_exit_end_time, 1, 2) AS INTEGER) BETWEEN 0 AND 23
            AND CAST(substr(bed_exit_end_time, 4, 2) AS INTEGER) BETWEEN 0 AND 59)
        ),
        topology_snapshot_registry_version INTEGER NOT NULL DEFAULT 0
            CHECK (topology_snapshot_registry_version >= 0),
        topology_client_revision INTEGER NOT NULL DEFAULT 0
            CHECK (topology_client_revision >= 0),
        topology_server_revision INTEGER NOT NULL DEFAULT 0
            CHECK (topology_server_revision >= 0),
        topology_pending_snapshot_id TEXT CHECK (
            topology_pending_snapshot_id IS NULL
            OR (
                length(topology_pending_snapshot_id) BETWEEN 1 AND 128
                AND instr(topology_pending_snapshot_id, char(0)) = 0
            )
        ),
        topology_pending_body BLOB CHECK (
            topology_pending_body IS NULL OR length(topology_pending_body) BETWEEN 1 AND 1048576
        ),
        topology_pending_registry_version INTEGER CHECK (
            topology_pending_registry_version IS NULL OR topology_pending_registry_version >= 0
        ),
        topology_pending_client_revision INTEGER CHECK (
            topology_pending_client_revision IS NULL OR topology_pending_client_revision > 0
        ),
        topology_pending_expected_server_revision INTEGER CHECK (
            topology_pending_expected_server_revision IS NULL
            OR topology_pending_expected_server_revision >= 0
        ),
        topology_consecutive_failures INTEGER NOT NULL DEFAULT 0
            CHECK (topology_consecutive_failures >= 0),
        topology_next_retry_at REAL,
        topology_pause_reason TEXT CHECK (
            topology_pause_reason IS NULL
            OR topology_pause_reason IN ('auth', 'forbidden', 'conflict')
        ),
        topology_last_accepted_at REAL,
        topology_dirty_registry_version INTEGER CHECK (
            topology_dirty_registry_version IS NULL OR topology_dirty_registry_version >= 1
        ),
        topology_dirty_created_at TEXT CHECK (
            topology_dirty_created_at IS NULL
            OR (
                length(topology_dirty_created_at) BETWEEN 20 AND 30
                AND instr(topology_dirty_created_at, char(0)) = 0
                AND substr(topology_dirty_created_at, 5, 1) = '-'
                AND substr(topology_dirty_created_at, 8, 1) = '-'
                AND substr(topology_dirty_created_at, 11, 1) = 'T'
                AND substr(topology_dirty_created_at, 14, 1) = ':'
                AND substr(topology_dirty_created_at, 17, 1) = ':'
                AND substr(topology_dirty_created_at, -1) = 'Z'
                AND datetime(substr(topology_dirty_created_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(topology_dirty_created_at, 1, 19))
                    = substr(topology_dirty_created_at, 1, 19)
                AND (
                    length(topology_dirty_created_at) = 20
                    OR (
                        substr(topology_dirty_created_at, 20, 1) = '.'
                        AND length(topology_dirty_created_at) BETWEEN 22 AND 27
                        AND substr(
                            topology_dirty_created_at,
                            21,
                            length(topology_dirty_created_at) - 21
                        )
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        topology_confirmation_id TEXT CHECK (
            topology_confirmation_id IS NULL
            OR (
                length(topology_confirmation_id) BETWEEN 1 AND 128
                AND instr(topology_confirmation_id, char(0)) = 0
            )
        ),
        topology_confirmation_digest TEXT CHECK (
            topology_confirmation_digest IS NULL
            OR (
                length(topology_confirmation_digest) = 64
                AND topology_confirmation_digest NOT GLOB '*[^0-9a-f]*'
            )
        ),
        topology_confirmation_expires_at TEXT CHECK (
            topology_confirmation_expires_at IS NULL
            OR (
                length(topology_confirmation_expires_at) BETWEEN 20 AND 30
                AND instr(topology_confirmation_expires_at, char(0)) = 0
                AND substr(topology_confirmation_expires_at, 5, 1) = '-'
                AND substr(topology_confirmation_expires_at, 8, 1) = '-'
                AND substr(topology_confirmation_expires_at, 11, 1) = 'T'
                AND substr(topology_confirmation_expires_at, 14, 1) = ':'
                AND substr(topology_confirmation_expires_at, 17, 1) = ':'
                AND substr(topology_confirmation_expires_at, -1) = 'Z'
                AND datetime(substr(topology_confirmation_expires_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(topology_confirmation_expires_at, 1, 19))
                    = substr(topology_confirmation_expires_at, 1, 19)
                AND (
                    length(topology_confirmation_expires_at) = 20
                    OR (
                        substr(topology_confirmation_expires_at, 20, 1) = '.'
                        AND length(topology_confirmation_expires_at) BETWEEN 22 AND 27
                        AND substr(
                            topology_confirmation_expires_at,
                            21,
                            length(topology_confirmation_expires_at) - 21
                        )
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        topology_confirmation_snapshot_id TEXT CHECK (
            topology_confirmation_snapshot_id IS NULL
            OR (
                length(topology_confirmation_snapshot_id) BETWEEN 1 AND 128
                AND instr(topology_confirmation_snapshot_id, char(0)) = 0
            )
        ),
        topology_confirmation_client_revision INTEGER CHECK (
            topology_confirmation_client_revision IS NULL
            OR topology_confirmation_client_revision >= 0
        ),
        topology_confirmation_server_revision INTEGER CHECK (
            topology_confirmation_server_revision IS NULL
            OR topology_confirmation_server_revision >= 0
        ),
        topology_confirmation_registry_version INTEGER CHECK (
            topology_confirmation_registry_version IS NULL
            OR topology_confirmation_registry_version >= 0
        ),
        topology_confirmation_cameras INTEGER CHECK (
            topology_confirmation_cameras IS NULL OR topology_confirmation_cameras >= 0
        ),
        topology_confirmation_rooms INTEGER CHECK (
            topology_confirmation_rooms IS NULL OR topology_confirmation_rooms >= 0
        ),
        topology_confirmation_floors INTEGER CHECK (
            topology_confirmation_floors IS NULL OR topology_confirmation_floors >= 0
        ),
        topology_confirmation_confirmed INTEGER CHECK (
            topology_confirmation_confirmed IS NULL OR topology_confirmation_confirmed IN (0, 1)
        ),
        topology_confirmation_result TEXT CHECK (
            topology_confirmation_result IS NULL
            OR (
                length(topology_confirmation_result) BETWEEN 1 AND 64
                AND instr(topology_confirmation_result, char(0)) = 0
            )
        ),
        storage_state TEXT CHECK (
            storage_state IS NULL
            OR (
                length(storage_state) BETWEEN 1 AND 64 AND instr(storage_state, char(0)) = 0
            )
        ),
        recording_suspended INTEGER CHECK (
            recording_suspended IS NULL OR recording_suspended IN (0, 1)
        ),
        audit_state TEXT CHECK (
            audit_state IS NULL
            OR (length(audit_state) BETWEEN 1 AND 64 AND instr(audit_state, char(0)) = 0)
        ),
        audit_last_success_at TEXT CHECK (
            audit_last_success_at IS NULL
            OR (
                length(audit_last_success_at) BETWEEN 20 AND 30
                AND instr(audit_last_success_at, char(0)) = 0
                AND substr(audit_last_success_at, 5, 1) = '-'
                AND substr(audit_last_success_at, 8, 1) = '-'
                AND substr(audit_last_success_at, 11, 1) = 'T'
                AND substr(audit_last_success_at, 14, 1) = ':'
                AND substr(audit_last_success_at, 17, 1) = ':'
                AND substr(audit_last_success_at, -1) = 'Z'
                AND datetime(substr(audit_last_success_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(audit_last_success_at, 1, 19))
                    = substr(audit_last_success_at, 1, 19)
                AND (
                    length(audit_last_success_at) = 20
                    OR (
                        substr(audit_last_success_at, 20, 1) = '.'
                        AND length(audit_last_success_at) BETWEEN 22 AND 27
                        AND substr(audit_last_success_at, 21, length(audit_last_success_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        degraded_failure_code TEXT CHECK (
            degraded_failure_code IS NULL
            OR (
                length(degraded_failure_code) BETWEEN 1 AND 64
                AND instr(degraded_failure_code, char(0)) = 0
            )
        ),
        degraded_observed_at TEXT CHECK (
            degraded_observed_at IS NULL
            OR (
                length(degraded_observed_at) BETWEEN 20 AND 30
                AND instr(degraded_observed_at, char(0)) = 0
                AND substr(degraded_observed_at, 5, 1) = '-'
                AND substr(degraded_observed_at, 8, 1) = '-'
                AND substr(degraded_observed_at, 11, 1) = 'T'
                AND substr(degraded_observed_at, 14, 1) = ':'
                AND substr(degraded_observed_at, 17, 1) = ':'
                AND substr(degraded_observed_at, -1) = 'Z'
                AND datetime(substr(degraded_observed_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(degraded_observed_at, 1, 19))
                    = substr(degraded_observed_at, 1, 19)
                AND (
                    length(degraded_observed_at) = 20
                    OR (
                        substr(degraded_observed_at, 20, 1) = '.'
                        AND length(degraded_observed_at) BETWEEN 22 AND 27
                        AND substr(degraded_observed_at, 21, length(degraded_observed_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        updated_at TEXT NOT NULL CHECK (
            length(updated_at) BETWEEN 20 AND 30
            AND instr(updated_at, char(0)) = 0
            AND substr(updated_at, 5, 1) = '-'
            AND substr(updated_at, 8, 1) = '-'
            AND substr(updated_at, 11, 1) = 'T'
            AND substr(updated_at, 14, 1) = ':'
            AND substr(updated_at, 17, 1) = ':'
            AND substr(updated_at, -1) = 'Z'
            AND datetime(substr(updated_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(updated_at, 1, 19))
                = substr(updated_at, 1, 19)
            AND (
                length(updated_at) = 20
                OR (
                    substr(updated_at, 20, 1) = '.'
                    AND length(updated_at) BETWEEN 22 AND 27
                    AND substr(updated_at, 21, length(updated_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        CHECK (
            (
                facility_code IS NULL
                AND client_installation_ref IS NULL
                AND facility_id IS NULL
                AND facility_token IS NULL
                AND edge_installation_id IS NULL
                AND enrollment_generation IS NULL
                AND enrollment_created_at IS NULL
                AND enrollment_updated_at IS NULL
            )
            OR (
                facility_code IS NOT NULL
                AND client_installation_ref IS NOT NULL
                AND facility_id IS NOT NULL
                AND facility_token IS NOT NULL
                AND edge_installation_id IS NOT NULL
                AND enrollment_generation IS NOT NULL
                AND enrollment_created_at IS NOT NULL
                AND enrollment_updated_at IS NOT NULL
            )
        ),
        CHECK (
            (
                fall_on IS NULL
                AND fall_mode IS NULL
                AND fall_start_time IS NULL
                AND fall_end_time IS NULL
            )
            OR (
                fall_on IS NOT NULL
                AND fall_mode IS NOT NULL
                AND (
                    (fall_mode = 'always' AND fall_start_time IS NULL AND fall_end_time IS NULL)
                    OR (
                        fall_mode = 'window'
                        AND fall_start_time IS NOT NULL
                        AND fall_end_time IS NOT NULL
                    )
                )
            )
        ),
        CHECK (
            (
                bed_exit_on IS NULL
                AND bed_exit_mode IS NULL
                AND bed_exit_start_time IS NULL
                AND bed_exit_end_time IS NULL
            )
            OR (
                bed_exit_on IS NOT NULL
                AND bed_exit_mode IS NOT NULL
                AND (
                    (
                        bed_exit_mode = 'always'
                        AND bed_exit_start_time IS NULL
                        AND bed_exit_end_time IS NULL
                    )
                    OR (
                        bed_exit_mode = 'window'
                        AND bed_exit_start_time IS NOT NULL
                        AND bed_exit_end_time IS NOT NULL
                    )
                )
            )
        ),
        CHECK (
            (
                topology_pending_snapshot_id IS NULL
                AND topology_pending_body IS NULL
                AND topology_pending_registry_version IS NULL
                AND topology_pending_client_revision IS NULL
                AND topology_pending_expected_server_revision IS NULL
            )
            OR (
                topology_pending_snapshot_id IS NOT NULL
                AND topology_pending_body IS NOT NULL
                AND topology_pending_registry_version IS NOT NULL
                AND topology_pending_client_revision IS NOT NULL
                AND topology_pending_expected_server_revision IS NOT NULL
            )
        ),
        CHECK (
            (
                topology_dirty_registry_version IS NULL
                AND topology_dirty_created_at IS NULL
            )
            OR (
                topology_dirty_registry_version IS NOT NULL
                AND topology_dirty_created_at IS NOT NULL
            )
        ),
        CHECK (
            (
                topology_confirmation_id IS NULL
                AND topology_confirmation_digest IS NULL
                AND topology_confirmation_expires_at IS NULL
                AND topology_confirmation_snapshot_id IS NULL
                AND topology_confirmation_client_revision IS NULL
                AND topology_confirmation_server_revision IS NULL
                AND topology_confirmation_registry_version IS NULL
                AND topology_confirmation_cameras IS NULL
                AND topology_confirmation_rooms IS NULL
                AND topology_confirmation_floors IS NULL
                AND topology_confirmation_confirmed IS NULL
                AND topology_confirmation_result IS NULL
            )
            OR (
                topology_confirmation_id IS NOT NULL
                AND topology_confirmation_digest IS NOT NULL
                AND topology_confirmation_expires_at IS NOT NULL
                AND topology_confirmation_snapshot_id IS NOT NULL
                AND topology_confirmation_client_revision IS NOT NULL
                AND topology_confirmation_server_revision IS NOT NULL
                AND topology_confirmation_registry_version IS NOT NULL
                AND topology_confirmation_cameras IS NOT NULL
                AND topology_confirmation_rooms IS NOT NULL
                AND topology_confirmation_floors IS NOT NULL
                AND topology_confirmation_confirmed IS NOT NULL
                AND (
                    topology_confirmation_result IS NULL
                    OR topology_confirmation_confirmed = 1
                )
            )
        ),
        CHECK (
            (
                storage_state IS NULL
                AND recording_suspended IS NULL
                AND audit_state IS NULL
                AND audit_last_success_at IS NULL
                AND degraded_failure_code IS NULL
                AND degraded_observed_at IS NULL
            )
            OR (
                storage_state IS NOT NULL
                AND recording_suspended IS NOT NULL
                AND audit_state IS NOT NULL
                AND degraded_observed_at IS NOT NULL
            )
        )
    ) STRICT
    """,
    """
    CREATE TABLE locations (
        location_id TEXT NOT NULL CHECK (
            length(location_id) BETWEEN 1 AND 128 AND instr(location_id, char(0)) = 0
        ),
        kind TEXT NOT NULL CHECK (kind IN ('FLOOR', 'ROOM')),
        parent_location_id TEXT CHECK (
            parent_location_id IS NULL
            OR (
                length(parent_location_id) BETWEEN 1 AND 128
                AND instr(parent_location_id, char(0)) = 0
            )
        ),
        parent_kind TEXT CHECK (parent_kind IS NULL OR parent_kind = 'FLOOR'),
        name TEXT NOT NULL CHECK (
            length(name) BETWEEN 1 AND 128 AND instr(name, char(0)) = 0
        ),
        order_index INTEGER NOT NULL CHECK (order_index >= 0),
        capacity INTEGER CHECK (capacity IS NULL OR capacity > 0),
        legacy_space_id TEXT UNIQUE CHECK (
            legacy_space_id IS NULL
            OR (length(legacy_space_id) BETWEEN 1 AND 128 AND instr(legacy_space_id, char(0)) = 0)
        ),
        created_at TEXT NOT NULL CHECK (
            length(created_at) BETWEEN 20 AND 30
            AND instr(created_at, char(0)) = 0
            AND substr(created_at, 5, 1) = '-'
            AND substr(created_at, 8, 1) = '-'
            AND substr(created_at, 11, 1) = 'T'
            AND substr(created_at, 14, 1) = ':'
            AND substr(created_at, 17, 1) = ':'
            AND substr(created_at, -1) = 'Z'
            AND datetime(substr(created_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(created_at, 1, 19))
                = substr(created_at, 1, 19)
            AND (
                length(created_at) = 20
                OR (
                    substr(created_at, 20, 1) = '.'
                    AND length(created_at) BETWEEN 22 AND 27
                    AND substr(created_at, 21, length(created_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        updated_at TEXT NOT NULL CHECK (
            length(updated_at) BETWEEN 20 AND 30
            AND instr(updated_at, char(0)) = 0
            AND substr(updated_at, 5, 1) = '-'
            AND substr(updated_at, 8, 1) = '-'
            AND substr(updated_at, 11, 1) = 'T'
            AND substr(updated_at, 14, 1) = ':'
            AND substr(updated_at, 17, 1) = ':'
            AND substr(updated_at, -1) = 'Z'
            AND datetime(substr(updated_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(updated_at, 1, 19))
                = substr(updated_at, 1, 19)
            AND (
                length(updated_at) = 20
                OR (
                    substr(updated_at, 20, 1) = '.'
                    AND length(updated_at) BETWEEN 22 AND 27
                    AND substr(updated_at, 21, length(updated_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        PRIMARY KEY (location_id, kind),
        FOREIGN KEY (parent_location_id, parent_kind)
            REFERENCES locations(location_id, kind)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        CHECK (
            (
                kind = 'FLOOR'
                AND parent_location_id IS NULL
                AND parent_kind IS NULL
            )
            OR (
                kind = 'ROOM'
                AND parent_location_id IS NOT NULL
                AND parent_kind = 'FLOOR'
            )
        )
    ) STRICT
    """,
    """
    CREATE TABLE cameras (
        camera_id TEXT PRIMARY KEY CHECK (
            length(camera_id) BETWEEN 1 AND 128 AND instr(camera_id, char(0)) = 0
        ),
        backend_camera_id TEXT CHECK (
            backend_camera_id IS NULL
            OR (
                length(backend_camera_id) BETWEEN 1 AND 128
                AND instr(backend_camera_id, char(0)) = 0
            )
        ),
        label TEXT NOT NULL CHECK (
            length(label) BETWEEN 1 AND 128 AND instr(label, char(0)) = 0
        ),
        rtsp_url TEXT NOT NULL CHECK (
            length(rtsp_url) BETWEEN 1 AND 1024 AND instr(rtsp_url, char(0)) = 0
        ),
        normalized_stream_identity TEXT NOT NULL CHECK (
            length(normalized_stream_identity) BETWEEN 1 AND 256
            AND instr(normalized_stream_identity, char(0)) = 0
        ),
        space_id TEXT CHECK (
            space_id IS NULL
            OR (length(space_id) BETWEEN 1 AND 128 AND instr(space_id, char(0)) = 0)
        ),
        room_location_id TEXT CHECK (
            room_location_id IS NULL
            OR (
                length(room_location_id) BETWEEN 1 AND 128
                AND instr(room_location_id, char(0)) = 0
            )
        ),
        room_location_kind TEXT CHECK (room_location_kind IS NULL OR room_location_kind = 'ROOM'),
        edge_ref TEXT CHECK (
            edge_ref IS NULL
            OR (length(edge_ref) BETWEEN 1 AND 128 AND instr(edge_ref, char(0)) = 0)
        ),
        mapping_state TEXT NOT NULL CHECK (mapping_state IN ('PENDING', 'UNMAPPED', 'MAPPED')),
        decode_backend TEXT CHECK (
            decode_backend IS NULL
            OR (length(decode_backend) BETWEEN 1 AND 64 AND instr(decode_backend, char(0)) = 0)
        ),
        floor_override TEXT CHECK (
            floor_override IS NULL
            OR (length(floor_override) BETWEEN 1 AND 128 AND instr(floor_override, char(0)) = 0)
        ),
        never_connected INTEGER NOT NULL CHECK (never_connected IN (0, 1)),
        last_probed_at TEXT CHECK (
            last_probed_at IS NULL
            OR (
                length(last_probed_at) BETWEEN 20 AND 30
                AND instr(last_probed_at, char(0)) = 0
                AND substr(last_probed_at, 5, 1) = '-'
                AND substr(last_probed_at, 8, 1) = '-'
                AND substr(last_probed_at, 11, 1) = 'T'
                AND substr(last_probed_at, 14, 1) = ':'
                AND substr(last_probed_at, 17, 1) = ':'
                AND substr(last_probed_at, -1) = 'Z'
                AND datetime(substr(last_probed_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(last_probed_at, 1, 19))
                    = substr(last_probed_at, 1, 19)
                AND (
                    length(last_probed_at) = 20
                    OR (
                        substr(last_probed_at, 20, 1) = '.'
                        AND length(last_probed_at) BETWEEN 22 AND 27
                        AND substr(last_probed_at, 21, length(last_probed_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        last_ok_at TEXT CHECK (
            last_ok_at IS NULL
            OR (
                length(last_ok_at) BETWEEN 20 AND 30
                AND instr(last_ok_at, char(0)) = 0
                AND substr(last_ok_at, 5, 1) = '-'
                AND substr(last_ok_at, 8, 1) = '-'
                AND substr(last_ok_at, 11, 1) = 'T'
                AND substr(last_ok_at, 14, 1) = ':'
                AND substr(last_ok_at, 17, 1) = ':'
                AND substr(last_ok_at, -1) = 'Z'
                AND datetime(substr(last_ok_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(last_ok_at, 1, 19))
                    = substr(last_ok_at, 1, 19)
                AND (
                    length(last_ok_at) = 20
                    OR (
                        substr(last_ok_at, 20, 1) = '.'
                        AND length(last_ok_at) BETWEEN 22 AND 27
                        AND substr(last_ok_at, 21, length(last_ok_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        bed_polygon_json TEXT CHECK (
            bed_polygon_json IS NULL
            OR (
                json_valid(bed_polygon_json)
                AND json_type(bed_polygon_json) = 'array'
                AND length(CAST(bed_polygon_json AS BLOB)) BETWEEN 1 AND 4096
            )
        ),
        bed_image_width INTEGER CHECK (bed_image_width IS NULL OR bed_image_width > 0),
        bed_image_height INTEGER CHECK (bed_image_height IS NULL OR bed_image_height > 0),
        bed_recognized_at TEXT CHECK (
            bed_recognized_at IS NULL
            OR (
                length(bed_recognized_at) BETWEEN 20 AND 30
                AND instr(bed_recognized_at, char(0)) = 0
                AND substr(bed_recognized_at, 5, 1) = '-'
                AND substr(bed_recognized_at, 8, 1) = '-'
                AND substr(bed_recognized_at, 11, 1) = 'T'
                AND substr(bed_recognized_at, 14, 1) = ':'
                AND substr(bed_recognized_at, 17, 1) = ':'
                AND substr(bed_recognized_at, -1) = 'Z'
                AND datetime(substr(bed_recognized_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(bed_recognized_at, 1, 19))
                    = substr(bed_recognized_at, 1, 19)
                AND (
                    length(bed_recognized_at) = 20
                    OR (
                        substr(bed_recognized_at, 20, 1) = '.'
                        AND length(bed_recognized_at) BETWEEN 22 AND 27
                        AND substr(bed_recognized_at, 21, length(bed_recognized_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        revision INTEGER NOT NULL CHECK (revision > 0),
        created_at TEXT NOT NULL CHECK (
            length(created_at) BETWEEN 20 AND 30
            AND instr(created_at, char(0)) = 0
            AND substr(created_at, 5, 1) = '-'
            AND substr(created_at, 8, 1) = '-'
            AND substr(created_at, 11, 1) = 'T'
            AND substr(created_at, 14, 1) = ':'
            AND substr(created_at, 17, 1) = ':'
            AND substr(created_at, -1) = 'Z'
            AND datetime(substr(created_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(created_at, 1, 19))
                = substr(created_at, 1, 19)
            AND (
                length(created_at) = 20
                OR (
                    substr(created_at, 20, 1) = '.'
                    AND length(created_at) BETWEEN 22 AND 27
                    AND substr(created_at, 21, length(created_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        updated_at TEXT NOT NULL CHECK (
            length(updated_at) BETWEEN 20 AND 30
            AND instr(updated_at, char(0)) = 0
            AND substr(updated_at, 5, 1) = '-'
            AND substr(updated_at, 8, 1) = '-'
            AND substr(updated_at, 11, 1) = 'T'
            AND substr(updated_at, 14, 1) = ':'
            AND substr(updated_at, 17, 1) = ':'
            AND substr(updated_at, -1) = 'Z'
            AND datetime(substr(updated_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(updated_at, 1, 19))
                = substr(updated_at, 1, 19)
            AND (
                length(updated_at) = 20
                OR (
                    substr(updated_at, 20, 1) = '.'
                    AND length(updated_at) BETWEEN 22 AND 27
                    AND substr(updated_at, 21, length(updated_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        FOREIGN KEY (room_location_id, room_location_kind)
            REFERENCES locations(location_id, kind)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        CHECK (
            (
                room_location_id IS NULL
                AND room_location_kind IS NULL
                AND edge_ref IS NULL
            )
            OR (
                room_location_id IS NOT NULL
                AND room_location_kind = 'ROOM'
                AND edge_ref IS NOT NULL
            )
        ),
        CHECK (
            (
                mapping_state = 'MAPPED'
                AND backend_camera_id IS NOT NULL
            )
            OR (
                mapping_state IN ('PENDING', 'UNMAPPED')
                AND backend_camera_id IS NULL
            )
        ),
        CHECK (
            (
                bed_polygon_json IS NULL
                AND bed_image_width IS NULL
                AND bed_image_height IS NULL
                AND bed_recognized_at IS NULL
            )
            OR (
                bed_polygon_json IS NOT NULL
                AND bed_image_width IS NOT NULL
                AND bed_image_height IS NOT NULL
                AND bed_recognized_at IS NOT NULL
            )
        )
    ) STRICT
    """,
    """
    CREATE UNIQUE INDEX cameras_one_room_idx
    ON cameras(room_location_id)
    WHERE room_location_id IS NOT NULL
    """,
    """
    CREATE TABLE policies (
        policy_id INTEGER PRIMARY KEY,
        facility_id TEXT NOT NULL CHECK (
            length(facility_id) BETWEEN 1 AND 128 AND instr(facility_id, char(0)) = 0
        ),
        camera_id TEXT CHECK (
            camera_id IS NULL
            OR (length(camera_id) BETWEEN 1 AND 128 AND instr(camera_id, char(0)) = 0)
        ),
        module_id TEXT NOT NULL CHECK (
            length(module_id) BETWEEN 1 AND 128 AND instr(module_id, char(0)) = 0
        ),
        module_version INTEGER NOT NULL CHECK (module_version > 0),
        schema_id TEXT NOT NULL CHECK (
            length(schema_id) BETWEEN 1 AND 128 AND instr(schema_id, char(0)) = 0
        ),
        schema_version INTEGER NOT NULL CHECK (schema_version > 0),
        active_values_json TEXT CHECK (
            active_values_json IS NULL
            OR (
                json_valid(active_values_json)
                AND json_type(active_values_json) = 'object'
                AND length(CAST(active_values_json AS BLOB)) BETWEEN 2 AND 16384
            )
        ),
        active_content_sha256 TEXT CHECK (
            active_content_sha256 IS NULL
            OR (
                length(active_content_sha256) = 64
                AND active_content_sha256 NOT GLOB '*[^0-9a-f]*'
            )
        ),
        previous_present INTEGER NOT NULL CHECK (previous_present IN (0, 1)),
        previous_values_json TEXT CHECK (
            previous_values_json IS NULL
            OR (
                json_valid(previous_values_json)
                AND json_type(previous_values_json) = 'object'
                AND length(CAST(previous_values_json AS BLOB)) BETWEEN 2 AND 16384
            )
        ),
        previous_content_sha256 TEXT CHECK (
            previous_content_sha256 IS NULL
            OR (
                length(previous_content_sha256) = 64
                AND previous_content_sha256 NOT GLOB '*[^0-9a-f]*'
            )
        ),
        activation_generation INTEGER NOT NULL CHECK (activation_generation > 0),
        status TEXT NOT NULL CHECK (status IN ('pending', 'applied', 'failed')),
        refusal_reason TEXT CHECK (
            refusal_reason IS NULL
            OR (length(refusal_reason) BETWEEN 1 AND 256 AND instr(refusal_reason, char(0)) = 0)
        ),
        activated_at TEXT NOT NULL CHECK (
            length(activated_at) BETWEEN 20 AND 30
            AND instr(activated_at, char(0)) = 0
            AND substr(activated_at, 5, 1) = '-'
            AND substr(activated_at, 8, 1) = '-'
            AND substr(activated_at, 11, 1) = 'T'
            AND substr(activated_at, 14, 1) = ':'
            AND substr(activated_at, 17, 1) = ':'
            AND substr(activated_at, -1) = 'Z'
            AND datetime(substr(activated_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(activated_at, 1, 19))
                = substr(activated_at, 1, 19)
            AND (
                length(activated_at) = 20
                OR (
                    substr(activated_at, 20, 1) = '.'
                    AND length(activated_at) BETWEEN 22 AND 27
                    AND substr(activated_at, 21, length(activated_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        applied_at TEXT CHECK (
            applied_at IS NULL
            OR (
                length(applied_at) BETWEEN 20 AND 30
                AND instr(applied_at, char(0)) = 0
                AND substr(applied_at, 5, 1) = '-'
                AND substr(applied_at, 8, 1) = '-'
                AND substr(applied_at, 11, 1) = 'T'
                AND substr(applied_at, 14, 1) = ':'
                AND substr(applied_at, 17, 1) = ':'
                AND substr(applied_at, -1) = 'Z'
                AND datetime(substr(applied_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(applied_at, 1, 19))
                    = substr(applied_at, 1, 19)
                AND (
                    length(applied_at) = 20
                    OR (
                        substr(applied_at, 20, 1) = '.'
                        AND length(applied_at) BETWEEN 22 AND 27
                        AND substr(applied_at, 21, length(applied_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        updated_at TEXT NOT NULL CHECK (
            length(updated_at) BETWEEN 20 AND 30
            AND instr(updated_at, char(0)) = 0
            AND substr(updated_at, 5, 1) = '-'
            AND substr(updated_at, 8, 1) = '-'
            AND substr(updated_at, 11, 1) = 'T'
            AND substr(updated_at, 14, 1) = ':'
            AND substr(updated_at, 17, 1) = ':'
            AND substr(updated_at, -1) = 'Z'
            AND datetime(substr(updated_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(updated_at, 1, 19))
                = substr(updated_at, 1, 19)
            AND (
                length(updated_at) = 20
                OR (
                    substr(updated_at, 20, 1) = '.'
                    AND length(updated_at) BETWEEN 22 AND 27
                    AND substr(updated_at, 21, length(updated_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        CHECK (
            (active_values_json IS NULL) = (active_content_sha256 IS NULL)
        ),
        CHECK (
            camera_id IS NOT NULL OR (
                active_values_json IS NOT NULL AND active_content_sha256 IS NOT NULL
            )
        ),
        CHECK (
            (previous_values_json IS NULL) = (previous_content_sha256 IS NULL)
        ),
        CHECK (
            previous_present = 1
            OR (previous_values_json IS NULL AND previous_content_sha256 IS NULL)
        ),
        CHECK (
            (status = 'pending' AND applied_at IS NULL AND refusal_reason IS NULL)
            OR (status = 'applied' AND applied_at IS NOT NULL AND refusal_reason IS NULL)
            OR (status = 'failed' AND refusal_reason IS NOT NULL)
        )
    ) STRICT
    """,
    """
    CREATE UNIQUE INDEX policies_facility_scope_idx
    ON policies(facility_id, module_id, module_version)
    WHERE camera_id IS NULL
    """,
    """
    CREATE UNIQUE INDEX policies_camera_scope_idx
    ON policies(facility_id, camera_id, module_id, module_version)
    WHERE camera_id IS NOT NULL
    """,
    """
    CREATE TABLE clips (
        clip_id TEXT PRIMARY KEY CHECK (
            length(clip_id) BETWEEN 1 AND 128 AND instr(clip_id, char(0)) = 0
        ),
        camera_id TEXT NOT NULL CHECK (
            length(camera_id) BETWEEN 1 AND 128 AND instr(camera_id, char(0)) = 0
        ),
        event_facet TEXT NOT NULL CHECK (event_facet IN ('fall', 'bed-exit', 'other')),
        started_at TEXT NOT NULL CHECK (
            length(started_at) BETWEEN 20 AND 30
            AND instr(started_at, char(0)) = 0
            AND substr(started_at, 5, 1) = '-'
            AND substr(started_at, 8, 1) = '-'
            AND substr(started_at, 11, 1) = 'T'
            AND substr(started_at, 14, 1) = ':'
            AND substr(started_at, 17, 1) = ':'
            AND substr(started_at, -1) = 'Z'
            AND datetime(substr(started_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(started_at, 1, 19))
                = substr(started_at, 1, 19)
            AND (
                length(started_at) = 20
                OR (
                    substr(started_at, 20, 1) = '.'
                    AND length(started_at) BETWEEN 22 AND 27
                    AND substr(started_at, 21, length(started_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        finalized_at TEXT CHECK (
            finalized_at IS NULL
            OR (
                length(finalized_at) BETWEEN 20 AND 30
                AND instr(finalized_at, char(0)) = 0
                AND substr(finalized_at, 5, 1) = '-'
                AND substr(finalized_at, 8, 1) = '-'
                AND substr(finalized_at, 11, 1) = 'T'
                AND substr(finalized_at, 14, 1) = ':'
                AND substr(finalized_at, 17, 1) = ':'
                AND substr(finalized_at, -1) = 'Z'
                AND datetime(substr(finalized_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(finalized_at, 1, 19))
                    = substr(finalized_at, 1, 19)
                AND (
                    length(finalized_at) = 20
                    OR (
                        substr(finalized_at, 20, 1) = '.'
                        AND length(finalized_at) BETWEEN 22 AND 27
                        AND substr(finalized_at, 21, length(finalized_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms BETWEEN 1 AND 120000),
        codec TEXT CHECK (
            codec IS NULL OR (length(codec) BETWEEN 1 AND 64 AND instr(codec, char(0)) = 0)
        ),
        mime_type TEXT CHECK (
            mime_type IS NULL
            OR (length(mime_type) BETWEEN 1 AND 128 AND instr(mime_type, char(0)) = 0)
        ),
        manifest_relpath TEXT CHECK (
            manifest_relpath IS NULL
            OR (
                length(manifest_relpath) BETWEEN 1 AND 512
                AND instr(manifest_relpath, char(0)) = 0
                AND substr(manifest_relpath, 1, 1) != '/'
                AND instr(manifest_relpath, '\\') = 0
                AND instr('/' || manifest_relpath || '/', '/../') = 0
            )
        ),
        media_relpath TEXT CHECK (
            media_relpath IS NULL
            OR (
                length(media_relpath) BETWEEN 1 AND 512
                AND instr(media_relpath, char(0)) = 0
                AND substr(media_relpath, 1, 1) != '/'
                AND instr(media_relpath, '\\') = 0
                AND instr('/' || media_relpath || '/', '/../') = 0
            )
        ),
        thumbnail_relpath TEXT CHECK (
            thumbnail_relpath IS NULL
            OR (
                length(thumbnail_relpath) BETWEEN 1 AND 512
                AND instr(thumbnail_relpath, char(0)) = 0
                AND substr(thumbnail_relpath, 1, 1) != '/'
                AND instr(thumbnail_relpath, '\\') = 0
                AND instr('/' || thumbnail_relpath || '/', '/../') = 0
            )
        ),
        manifest_sha256 TEXT CHECK (
            manifest_sha256 IS NULL
            OR (length(manifest_sha256) = 64 AND manifest_sha256 NOT GLOB '*[^0-9a-f]*')
        ),
        media_sha256 TEXT CHECK (
            media_sha256 IS NULL
            OR (length(media_sha256) = 64 AND media_sha256 NOT GLOB '*[^0-9a-f]*')
        ),
        thumbnail_sha256 TEXT CHECK (
            thumbnail_sha256 IS NULL
            OR (length(thumbnail_sha256) = 64 AND thumbnail_sha256 NOT GLOB '*[^0-9a-f]*')
        ),
        manifest_size_bytes INTEGER CHECK (
            manifest_size_bytes IS NULL OR manifest_size_bytes > 0
        ),
        media_size_bytes INTEGER CHECK (media_size_bytes IS NULL OR media_size_bytes > 0),
        thumbnail_size_bytes INTEGER CHECK (
            thumbnail_size_bytes IS NULL OR thumbnail_size_bytes > 0
        ),
        local_state TEXT NOT NULL CHECK (local_state IN ('AVAILABLE', 'UNAVAILABLE', 'CORRUPT')),
        local_reason TEXT CHECK (
            local_reason IS NULL
            OR (length(local_reason) BETWEEN 1 AND 64 AND instr(local_reason, char(0)) = 0)
        ),
        publish_state TEXT NOT NULL CHECK (
            publish_state IN ('WAITING', 'PUBLISHED', 'PERMANENT', 'COMPATIBILITY')
        ),
        published_at TEXT CHECK (
            published_at IS NULL
            OR (
                length(published_at) BETWEEN 20 AND 30
                AND instr(published_at, char(0)) = 0
                AND substr(published_at, 5, 1) = '-'
                AND substr(published_at, 8, 1) = '-'
                AND substr(published_at, 11, 1) = 'T'
                AND substr(published_at, 14, 1) = ':'
                AND substr(published_at, 17, 1) = ':'
                AND substr(published_at, -1) = 'Z'
                AND datetime(substr(published_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(published_at, 1, 19))
                    = substr(published_at, 1, 19)
                AND (
                    length(published_at) = 20
                    OR (
                        substr(published_at, 20, 1) = '.'
                        AND length(published_at) BETWEEN 22 AND 27
                        AND substr(published_at, 21, length(published_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        last_publish_error_code TEXT CHECK (
            last_publish_error_code IS NULL
            OR (
                length(last_publish_error_code) BETWEEN 1 AND 64
                AND instr(last_publish_error_code, char(0)) = 0
            )
        ),
        retention_state TEXT NOT NULL CHECK (
            retention_state IN ('RETAINED', 'PENDING', 'PURGED')
        ),
        retention_reason TEXT CHECK (
            retention_reason IS NULL
            OR (length(retention_reason) BETWEEN 1 AND 64 AND instr(retention_reason, char(0)) = 0)
        ),
        retention_requested_at TEXT CHECK (
            retention_requested_at IS NULL
            OR (
                length(retention_requested_at) BETWEEN 20 AND 30
                AND instr(retention_requested_at, char(0)) = 0
                AND substr(retention_requested_at, 5, 1) = '-'
                AND substr(retention_requested_at, 8, 1) = '-'
                AND substr(retention_requested_at, 11, 1) = 'T'
                AND substr(retention_requested_at, 14, 1) = ':'
                AND substr(retention_requested_at, 17, 1) = ':'
                AND substr(retention_requested_at, -1) = 'Z'
                AND datetime(substr(retention_requested_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(retention_requested_at, 1, 19))
                    = substr(retention_requested_at, 1, 19)
                AND (
                    length(retention_requested_at) = 20
                    OR (
                        substr(retention_requested_at, 20, 1) = '.'
                        AND length(retention_requested_at) BETWEEN 22 AND 27
                        AND substr(retention_requested_at, 21, length(retention_requested_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        retention_updated_at TEXT CHECK (
            retention_updated_at IS NULL
            OR (
                length(retention_updated_at) BETWEEN 20 AND 30
                AND instr(retention_updated_at, char(0)) = 0
                AND substr(retention_updated_at, 5, 1) = '-'
                AND substr(retention_updated_at, 8, 1) = '-'
                AND substr(retention_updated_at, 11, 1) = 'T'
                AND substr(retention_updated_at, 14, 1) = ':'
                AND substr(retention_updated_at, 17, 1) = ':'
                AND substr(retention_updated_at, -1) = 'Z'
                AND datetime(substr(retention_updated_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(retention_updated_at, 1, 19))
                    = substr(retention_updated_at, 1, 19)
                AND (
                    length(retention_updated_at) = 20
                    OR (
                        substr(retention_updated_at, 20, 1) = '.'
                        AND length(retention_updated_at) BETWEEN 22 AND 27
                        AND substr(retention_updated_at, 21, length(retention_updated_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        revision INTEGER NOT NULL CHECK (revision > 0),
        created_at TEXT NOT NULL CHECK (
            length(created_at) BETWEEN 20 AND 30
            AND instr(created_at, char(0)) = 0
            AND substr(created_at, 5, 1) = '-'
            AND substr(created_at, 8, 1) = '-'
            AND substr(created_at, 11, 1) = 'T'
            AND substr(created_at, 14, 1) = ':'
            AND substr(created_at, 17, 1) = ':'
            AND substr(created_at, -1) = 'Z'
            AND datetime(substr(created_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(created_at, 1, 19))
                = substr(created_at, 1, 19)
            AND (
                length(created_at) = 20
                OR (
                    substr(created_at, 20, 1) = '.'
                    AND length(created_at) BETWEEN 22 AND 27
                    AND substr(created_at, 21, length(created_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        updated_at TEXT NOT NULL CHECK (
            length(updated_at) BETWEEN 20 AND 30
            AND instr(updated_at, char(0)) = 0
            AND substr(updated_at, 5, 1) = '-'
            AND substr(updated_at, 8, 1) = '-'
            AND substr(updated_at, 11, 1) = 'T'
            AND substr(updated_at, 14, 1) = ':'
            AND substr(updated_at, 17, 1) = ':'
            AND substr(updated_at, -1) = 'Z'
            AND datetime(substr(updated_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(updated_at, 1, 19))
                = substr(updated_at, 1, 19)
            AND (
                length(updated_at) = 20
                OR (
                    substr(updated_at, 20, 1) = '.'
                    AND length(updated_at) BETWEEN 22 AND 27
                    AND substr(updated_at, 21, length(updated_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        CHECK (
            (manifest_relpath IS NULL) = (manifest_sha256 IS NULL)
            AND (manifest_relpath IS NULL) = (manifest_size_bytes IS NULL)
        ),
        CHECK (
            (media_relpath IS NULL) = (media_sha256 IS NULL)
            AND (media_relpath IS NULL) = (media_size_bytes IS NULL)
        ),
        CHECK (
            (thumbnail_relpath IS NULL) = (thumbnail_sha256 IS NULL)
            AND (thumbnail_relpath IS NULL) = (thumbnail_size_bytes IS NULL)
        ),
        CHECK (
            (
                local_state = 'AVAILABLE'
                AND local_reason IS NULL
                AND manifest_relpath IS NOT NULL
                AND media_relpath IS NOT NULL
            )
            OR (
                local_state = 'UNAVAILABLE'
                AND local_reason IS NOT NULL
                AND manifest_relpath IS NULL
                AND media_relpath IS NULL
                AND thumbnail_relpath IS NULL
            )
            OR (
                local_state = 'CORRUPT'
                AND local_reason IS NOT NULL
                AND manifest_relpath IS NOT NULL
            )
        ),
        CHECK (
            (
                publish_state = 'WAITING'
                AND published_at IS NULL
            )
            OR (
                publish_state = 'PUBLISHED'
                AND published_at IS NOT NULL
                AND last_publish_error_code IS NULL
            )
            OR (
                publish_state IN ('PERMANENT', 'COMPATIBILITY')
                AND last_publish_error_code IS NULL
            )
        ),
        CHECK (
            (
                retention_state = 'RETAINED'
                AND retention_reason IS NULL
                AND retention_requested_at IS NULL
                AND retention_updated_at IS NULL
            )
            OR (
                retention_state = 'PENDING'
                AND retention_requested_at IS NOT NULL
                AND retention_updated_at IS NOT NULL
            )
            OR (
                retention_state = 'PURGED'
                AND retention_reason IS NOT NULL
                AND retention_requested_at IS NOT NULL
                AND retention_updated_at IS NOT NULL
                AND media_relpath IS NULL
            )
        )
    ) STRICT
    """,
    "CREATE INDEX clips_started_at_idx ON clips(started_at DESC, clip_id DESC)",
    "CREATE INDEX clips_camera_started_at_idx ON clips(camera_id, started_at DESC, clip_id DESC)",
    """
    CREATE INDEX clips_facet_started_at_idx
    ON clips(event_facet, started_at DESC, clip_id DESC)
    """,
    "CREATE INDEX clips_retention_idx ON clips(retention_state, clip_id)",
    """
    CREATE TABLE incidents (
        incident_id TEXT PRIMARY KEY CHECK (
            length(incident_id) BETWEEN 1 AND 128 AND instr(incident_id, char(0)) = 0
        ),
        edge_event_id TEXT NOT NULL UNIQUE CHECK (
            length(edge_event_id) BETWEEN 1 AND 128 AND instr(edge_event_id, char(0)) = 0
        ),
        facility_id TEXT NOT NULL CHECK (
            length(facility_id) BETWEEN 1 AND 128 AND instr(facility_id, char(0)) = 0
        ),
        camera_id TEXT NOT NULL CHECK (
            length(camera_id) BETWEEN 1 AND 128 AND instr(camera_id, char(0)) = 0
        ),
        event_type TEXT NOT NULL CHECK (
            length(event_type) BETWEEN 1 AND 64 AND instr(event_type, char(0)) = 0
        ),
        probability REAL CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1)),
        detected_at TEXT NOT NULL CHECK (
            length(detected_at) BETWEEN 20 AND 30
            AND instr(detected_at, char(0)) = 0
            AND substr(detected_at, 5, 1) = '-'
            AND substr(detected_at, 8, 1) = '-'
            AND substr(detected_at, 11, 1) = 'T'
            AND substr(detected_at, 14, 1) = ':'
            AND substr(detected_at, 17, 1) = ':'
            AND substr(detected_at, -1) = 'Z'
            AND datetime(substr(detected_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(detected_at, 1, 19))
                = substr(detected_at, 1, 19)
            AND (
                length(detected_at) = 20
                OR (
                    substr(detected_at, 20, 1) = '.'
                    AND length(detected_at) BETWEEN 22 AND 27
                    AND substr(detected_at, 21, length(detected_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        lifecycle_state TEXT NOT NULL CHECK (lifecycle_state IN ('OPEN', 'COMPLETE', 'FAILED')),
        failure_reason TEXT CHECK (
            failure_reason IS NULL
            OR (length(failure_reason) BETWEEN 1 AND 64 AND instr(failure_reason, char(0)) = 0)
        ),
        backend_event_id TEXT CHECK (
            backend_event_id IS NULL
            OR (length(backend_event_id) BETWEEN 1 AND 128 AND instr(backend_event_id, char(0)) = 0)
        ),
        runtime_manifest_sha256 TEXT CHECK (
            runtime_manifest_sha256 IS NULL
            OR (
                length(runtime_manifest_sha256) = 64
                AND runtime_manifest_sha256 NOT GLOB '*[^0-9a-f]*'
            )
        ),
        module_qualified_id TEXT CHECK (
            module_qualified_id IS NULL
            OR (
                length(module_qualified_id) BETWEEN 1 AND 128
                AND instr(module_qualified_id, char(0)) = 0
            )
        ),
        policy_qualified_id TEXT CHECK (
            policy_qualified_id IS NULL
            OR (
                length(policy_qualified_id) BETWEEN 1 AND 128
                AND instr(policy_qualified_id, char(0)) = 0
            )
        ),
        provenance_state TEXT NOT NULL CHECK (provenance_state IN ('QUALIFIED', 'MISSING')),
        provenance_missing_reason TEXT CHECK (
            provenance_missing_reason IS NULL
            OR (
                length(provenance_missing_reason) BETWEEN 1 AND 64
                AND instr(provenance_missing_reason, char(0)) = 0
            )
        ),
        review_version INTEGER NOT NULL CHECK (review_version >= 0),
        review_disposition TEXT CHECK (
            review_disposition IS NULL OR review_disposition IN ('TP', 'FP')
        ),
        review_actor TEXT CHECK (
            review_actor IS NULL
            OR (length(review_actor) BETWEEN 1 AND 128 AND instr(review_actor, char(0)) = 0)
        ),
        review_at TEXT CHECK (
            review_at IS NULL
            OR (
                length(review_at) BETWEEN 20 AND 30
                AND instr(review_at, char(0)) = 0
                AND substr(review_at, 5, 1) = '-'
                AND substr(review_at, 8, 1) = '-'
                AND substr(review_at, 11, 1) = 'T'
                AND substr(review_at, 14, 1) = ':'
                AND substr(review_at, 17, 1) = ':'
                AND substr(review_at, -1) = 'Z'
                AND datetime(substr(review_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(review_at, 1, 19))
                    = substr(review_at, 1, 19)
                AND (
                    length(review_at) = 20
                    OR (
                        substr(review_at, 20, 1) = '.'
                        AND length(review_at) BETWEEN 22 AND 27
                        AND substr(review_at, 21, length(review_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        review_notes TEXT CHECK (
            review_notes IS NULL
            OR (length(review_notes) BETWEEN 1 AND 1000 AND instr(review_notes, char(0)) = 0)
        ),
        revision INTEGER NOT NULL CHECK (revision > 0),
        created_at TEXT NOT NULL CHECK (
            length(created_at) BETWEEN 20 AND 30
            AND instr(created_at, char(0)) = 0
            AND substr(created_at, 5, 1) = '-'
            AND substr(created_at, 8, 1) = '-'
            AND substr(created_at, 11, 1) = 'T'
            AND substr(created_at, 14, 1) = ':'
            AND substr(created_at, 17, 1) = ':'
            AND substr(created_at, -1) = 'Z'
            AND datetime(substr(created_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(created_at, 1, 19))
                = substr(created_at, 1, 19)
            AND (
                length(created_at) = 20
                OR (
                    substr(created_at, 20, 1) = '.'
                    AND length(created_at) BETWEEN 22 AND 27
                    AND substr(created_at, 21, length(created_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        updated_at TEXT NOT NULL CHECK (
            length(updated_at) BETWEEN 20 AND 30
            AND instr(updated_at, char(0)) = 0
            AND substr(updated_at, 5, 1) = '-'
            AND substr(updated_at, 8, 1) = '-'
            AND substr(updated_at, 11, 1) = 'T'
            AND substr(updated_at, 14, 1) = ':'
            AND substr(updated_at, 17, 1) = ':'
            AND substr(updated_at, -1) = 'Z'
            AND datetime(substr(updated_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(updated_at, 1, 19))
                = substr(updated_at, 1, 19)
            AND (
                length(updated_at) = 20
                OR (
                    substr(updated_at, 20, 1) = '.'
                    AND length(updated_at) BETWEEN 22 AND 27
                    AND substr(updated_at, 21, length(updated_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        CHECK ((lifecycle_state = 'FAILED') = (failure_reason IS NOT NULL)),
        CHECK (
            (
                provenance_state = 'QUALIFIED'
                AND provenance_missing_reason IS NULL
                AND backend_event_id IS NOT NULL
                AND runtime_manifest_sha256 IS NOT NULL
                AND module_qualified_id IS NOT NULL
                AND policy_qualified_id IS NOT NULL
            )
            OR (
                provenance_state = 'MISSING'
                AND provenance_missing_reason IS NOT NULL
                AND backend_event_id IS NULL
                AND runtime_manifest_sha256 IS NULL
                AND module_qualified_id IS NULL
                AND policy_qualified_id IS NULL
            )
        ),
        CHECK (
            (
                review_version = 0
                AND review_disposition IS NULL
                AND review_actor IS NULL
                AND review_at IS NULL
                AND review_notes IS NULL
            )
            OR (
                review_version > 0
                AND review_disposition IS NOT NULL
                AND review_actor IS NOT NULL
                AND review_at IS NOT NULL
            )
        )
    ) STRICT
    """,
    """
    CREATE TRIGGER incidents_legal_lifecycle
    BEFORE UPDATE OF lifecycle_state ON incidents
    WHEN NEW.lifecycle_state IS NOT OLD.lifecycle_state AND NOT (
        (OLD.lifecycle_state = 'OPEN' AND NEW.lifecycle_state IN ('COMPLETE', 'FAILED'))
        OR (OLD.lifecycle_state = 'COMPLETE' AND NEW.lifecycle_state = 'FAILED')
    )
    BEGIN
        SELECT RAISE(ABORT, 'illegal incident lifecycle transition');
    END
    """,
    """
    CREATE TRIGGER incidents_immutable_identity
    BEFORE UPDATE ON incidents
    WHEN NEW.incident_id IS NOT OLD.incident_id
      OR NEW.edge_event_id IS NOT OLD.edge_event_id
      OR NEW.facility_id IS NOT OLD.facility_id
      OR NEW.camera_id IS NOT OLD.camera_id
      OR NEW.event_type IS NOT OLD.event_type
      OR NEW.detected_at IS NOT OLD.detected_at
      OR NEW.backend_event_id IS NOT OLD.backend_event_id
      OR NEW.runtime_manifest_sha256 IS NOT OLD.runtime_manifest_sha256
      OR NEW.module_qualified_id IS NOT OLD.module_qualified_id
      OR NEW.policy_qualified_id IS NOT OLD.policy_qualified_id
      OR NEW.provenance_state IS NOT OLD.provenance_state
      OR NEW.provenance_missing_reason IS NOT OLD.provenance_missing_reason
      OR NEW.created_at IS NOT OLD.created_at
    BEGIN
        SELECT RAISE(ABORT, 'incident identity and provenance are immutable');
    END
    """,
    """
    CREATE TRIGGER incidents_revision_guard
    BEFORE UPDATE ON incidents
    WHEN NEW.revision != OLD.revision + 1
    BEGIN
        SELECT RAISE(ABORT, 'incident revision must advance exactly once');
    END
    """,
    """
    CREATE TABLE artifacts (
        incident_id TEXT NOT NULL CHECK (
            length(incident_id) BETWEEN 1 AND 128 AND instr(incident_id, char(0)) = 0
        ),
        kind TEXT NOT NULL CHECK (kind IN ('PRIMARY_CLIP', 'SNAPSHOT')),
        artifact_id TEXT UNIQUE CHECK (
            artifact_id IS NULL
            OR (length(artifact_id) BETWEEN 1 AND 128 AND instr(artifact_id, char(0)) = 0)
        ),
        clip_id TEXT CHECK (
            clip_id IS NULL
            OR (length(clip_id) BETWEEN 1 AND 128 AND instr(clip_id, char(0)) = 0)
        ),
        state TEXT NOT NULL CHECK (
            state IN ('PENDING', 'AVAILABLE', 'UNAVAILABLE', 'CORRUPT', 'PURGED')
        ),
        reason TEXT CHECK (
            reason IS NULL
            OR (length(reason) BETWEEN 1 AND 64 AND instr(reason, char(0)) = 0)
        ),
        contained_relpath TEXT CHECK (
            contained_relpath IS NULL
            OR (
                length(contained_relpath) BETWEEN 1 AND 512
                AND instr(contained_relpath, char(0)) = 0
                AND substr(contained_relpath, 1, 1) != '/'
                AND instr(contained_relpath, '\\') = 0
                AND instr('/' || contained_relpath || '/', '/../') = 0
            )
        ),
        content_sha256 TEXT CHECK (
            content_sha256 IS NULL
            OR (length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*')
        ),
        size_bytes INTEGER CHECK (size_bytes IS NULL OR size_bytes > 0),
        mime_type TEXT CHECK (
            mime_type IS NULL
            OR (length(mime_type) BETWEEN 1 AND 128 AND instr(mime_type, char(0)) = 0)
        ),
        codec TEXT CHECK (
            codec IS NULL OR (length(codec) BETWEEN 1 AND 64 AND instr(codec, char(0)) = 0)
        ),
        captured_at TEXT CHECK (
            captured_at IS NULL
            OR (
                length(captured_at) BETWEEN 20 AND 30
                AND instr(captured_at, char(0)) = 0
                AND substr(captured_at, 5, 1) = '-'
                AND substr(captured_at, 8, 1) = '-'
                AND substr(captured_at, 11, 1) = 'T'
                AND substr(captured_at, 14, 1) = ':'
                AND substr(captured_at, 17, 1) = ':'
                AND substr(captured_at, -1) = 'Z'
                AND datetime(substr(captured_at, 1, 19)) IS NOT NULL
                AND strftime('%Y-%m-%dT%H:%M:%S', substr(captured_at, 1, 19))
                    = substr(captured_at, 1, 19)
                AND (
                    length(captured_at) = 20
                    OR (
                        substr(captured_at, 20, 1) = '.'
                        AND length(captured_at) BETWEEN 22 AND 27
                        AND substr(captured_at, 21, length(captured_at) - 21)
                            NOT GLOB '*[^0-9]*'
                    )
                )
            )
        ),
        revision INTEGER NOT NULL CHECK (revision > 0),
        created_at TEXT NOT NULL CHECK (
            length(created_at) BETWEEN 20 AND 30
            AND instr(created_at, char(0)) = 0
            AND substr(created_at, 5, 1) = '-'
            AND substr(created_at, 8, 1) = '-'
            AND substr(created_at, 11, 1) = 'T'
            AND substr(created_at, 14, 1) = ':'
            AND substr(created_at, 17, 1) = ':'
            AND substr(created_at, -1) = 'Z'
            AND datetime(substr(created_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(created_at, 1, 19))
                = substr(created_at, 1, 19)
            AND (
                length(created_at) = 20
                OR (
                    substr(created_at, 20, 1) = '.'
                    AND length(created_at) BETWEEN 22 AND 27
                    AND substr(created_at, 21, length(created_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        updated_at TEXT NOT NULL CHECK (
            length(updated_at) BETWEEN 20 AND 30
            AND instr(updated_at, char(0)) = 0
            AND substr(updated_at, 5, 1) = '-'
            AND substr(updated_at, 8, 1) = '-'
            AND substr(updated_at, 11, 1) = 'T'
            AND substr(updated_at, 14, 1) = ':'
            AND substr(updated_at, 17, 1) = ':'
            AND substr(updated_at, -1) = 'Z'
            AND datetime(substr(updated_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(updated_at, 1, 19))
                = substr(updated_at, 1, 19)
            AND (
                length(updated_at) = 20
                OR (
                    substr(updated_at, 20, 1) = '.'
                    AND length(updated_at) BETWEEN 22 AND 27
                    AND substr(updated_at, 21, length(updated_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        PRIMARY KEY (incident_id, kind),
        FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        FOREIGN KEY (clip_id) REFERENCES clips(clip_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT,
        CHECK (
            (kind = 'PRIMARY_CLIP' AND captured_at IS NULL)
            OR (kind = 'SNAPSHOT' AND clip_id IS NULL AND captured_at IS NOT NULL)
        ),
        CHECK (
            (
                state = 'PENDING'
                AND artifact_id IS NULL
                AND clip_id IS NULL
                AND reason IS NULL
                AND contained_relpath IS NULL
                AND content_sha256 IS NULL
                AND size_bytes IS NULL
                AND mime_type IS NULL
                AND codec IS NULL
            )
            OR (
                state = 'AVAILABLE'
                AND artifact_id IS NOT NULL
                AND reason IS NULL
                AND contained_relpath IS NOT NULL
                AND content_sha256 IS NOT NULL
                AND size_bytes IS NOT NULL
                AND mime_type IS NOT NULL
            )
            OR (
                state = 'UNAVAILABLE'
                AND reason IS NOT NULL
                AND contained_relpath IS NULL
                AND content_sha256 IS NULL
                AND size_bytes IS NULL
                AND mime_type IS NULL
                AND codec IS NULL
            )
            OR (
                state = 'CORRUPT'
                AND reason IS NOT NULL
                AND artifact_id IS NOT NULL
                AND contained_relpath IS NOT NULL
                AND content_sha256 IS NOT NULL
                AND size_bytes IS NOT NULL
                AND mime_type IS NOT NULL
            )
            OR (
                state = 'PURGED'
                AND artifact_id IS NOT NULL
                AND reason IS NOT NULL
                AND contained_relpath IS NULL
                AND (
                    (
                        content_sha256 IS NOT NULL
                        AND size_bytes IS NOT NULL
                        AND mime_type IS NOT NULL
                    )
                    OR (
                        content_sha256 IS NULL
                        AND size_bytes IS NULL
                        AND mime_type IS NULL
                    )
                )
            )
        )
    ) STRICT
    """,
    """
    CREATE TRIGGER artifacts_legal_transition
    BEFORE UPDATE OF state ON artifacts
    WHEN NEW.state IS NOT OLD.state AND NOT (
        (OLD.state = 'PENDING' AND NEW.state IN ('AVAILABLE', 'UNAVAILABLE', 'CORRUPT'))
        OR (OLD.state = 'AVAILABLE' AND NEW.state IN ('CORRUPT', 'PURGED'))
        OR (OLD.state = 'UNAVAILABLE' AND NEW.state IN ('AVAILABLE', 'PURGED'))
        OR (OLD.state = 'CORRUPT' AND NEW.state = 'PURGED')
    )
    BEGIN
        SELECT RAISE(ABORT, 'illegal artifact state transition');
    END
    """,
    """
    CREATE TRIGGER artifacts_corrupt_preserves_identity
    BEFORE UPDATE ON artifacts
    WHEN OLD.state = 'AVAILABLE' AND NEW.state = 'CORRUPT' AND (
        NEW.contained_relpath IS NOT OLD.contained_relpath
        OR NEW.content_sha256 IS NOT OLD.content_sha256
        OR NEW.size_bytes IS NOT OLD.size_bytes
        OR NEW.mime_type IS NOT OLD.mime_type
        OR NEW.codec IS NOT OLD.codec
        OR NEW.captured_at IS NOT OLD.captured_at
    )
    BEGIN
        SELECT RAISE(ABORT, 'artifact retained identity is immutable');
    END
    """,
    """
    CREATE TRIGGER artifacts_immutable_identity
    BEFORE UPDATE ON artifacts
    WHEN NEW.incident_id IS NOT OLD.incident_id
      OR NEW.kind IS NOT OLD.kind
      OR (OLD.artifact_id IS NOT NULL AND NEW.artifact_id IS NOT OLD.artifact_id)
      OR NEW.created_at IS NOT OLD.created_at
    BEGIN
        SELECT RAISE(ABORT, 'artifact identity is immutable');
    END
    """,
    """
    CREATE TRIGGER artifacts_revision_guard
    BEFORE UPDATE ON artifacts
    WHEN NEW.revision != OLD.revision + 1
    BEGIN
        SELECT RAISE(ABORT, 'artifact revision must advance exactly once');
    END
    """,
    """
    CREATE TABLE audit_events (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        occurred_at TEXT NOT NULL CHECK (
            length(occurred_at) BETWEEN 20 AND 30
            AND instr(occurred_at, char(0)) = 0
            AND substr(occurred_at, 5, 1) = '-'
            AND substr(occurred_at, 8, 1) = '-'
            AND substr(occurred_at, 11, 1) = 'T'
            AND substr(occurred_at, 14, 1) = ':'
            AND substr(occurred_at, 17, 1) = ':'
            AND substr(occurred_at, -1) = 'Z'
            AND datetime(substr(occurred_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(occurred_at, 1, 19))
                = substr(occurred_at, 1, 19)
            AND (
                length(occurred_at) = 20
                OR (
                    substr(occurred_at, 20, 1) = '.'
                    AND length(occurred_at) BETWEEN 22 AND 27
                    AND substr(occurred_at, 21, length(occurred_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        recorded_at TEXT NOT NULL CHECK (
            length(recorded_at) BETWEEN 20 AND 30
            AND instr(recorded_at, char(0)) = 0
            AND substr(recorded_at, 5, 1) = '-'
            AND substr(recorded_at, 8, 1) = '-'
            AND substr(recorded_at, 11, 1) = 'T'
            AND substr(recorded_at, 14, 1) = ':'
            AND substr(recorded_at, 17, 1) = ':'
            AND substr(recorded_at, -1) = 'Z'
            AND datetime(substr(recorded_at, 1, 19)) IS NOT NULL
            AND strftime('%Y-%m-%dT%H:%M:%S', substr(recorded_at, 1, 19))
                = substr(recorded_at, 1, 19)
            AND (
                length(recorded_at) = 20
                OR (
                    substr(recorded_at, 20, 1) = '.'
                    AND length(recorded_at) BETWEEN 22 AND 27
                    AND substr(recorded_at, 21, length(recorded_at) - 21)
                        NOT GLOB '*[^0-9]*'
                )
            )
        ),
        clock_quality TEXT NOT NULL CHECK (clock_quality IN ('trusted', 'untrusted', 'unknown')),
        actor_type TEXT NOT NULL CHECK (actor_type IN ('user', 'service', 'system')),
        actor_id TEXT NOT NULL CHECK (
            length(actor_id) BETWEEN 1 AND 128 AND instr(actor_id, char(0)) = 0
        ),
        auth_mechanism TEXT NOT NULL CHECK (
            length(auth_mechanism) BETWEEN 1 AND 64 AND instr(auth_mechanism, char(0)) = 0
        ),
        action TEXT NOT NULL CHECK (
            length(action) BETWEEN 1 AND 64 AND instr(action, char(0)) = 0
        ),
        target_type TEXT NOT NULL CHECK (
            length(target_type) BETWEEN 1 AND 64 AND instr(target_type, char(0)) = 0
        ),
        target_id TEXT NOT NULL CHECK (
            length(target_id) BETWEEN 1 AND 128 AND instr(target_id, char(0)) = 0
        ),
        outcome TEXT NOT NULL CHECK (outcome IN ('success', 'denied', 'failed')),
        reason TEXT CHECK (
            reason IS NULL OR (length(reason) BETWEEN 1 AND 256 AND instr(reason, char(0)) = 0)
        ),
        request_id TEXT CHECK (
            request_id IS NULL
            OR (length(request_id) BETWEEN 1 AND 128 AND instr(request_id, char(0)) = 0)
        ),
        interaction_id TEXT CHECK (
            interaction_id IS NULL
            OR (length(interaction_id) BETWEEN 1 AND 128 AND instr(interaction_id, char(0)) = 0)
        ),
        detail_json TEXT CHECK (
            detail_json IS NULL
            OR (
                json_valid(detail_json)
                AND json_type(detail_json) = 'object'
                AND length(CAST(detail_json AS BLOB)) BETWEEN 2 AND 16384
            )
        ),
        previous_hash TEXT NOT NULL CHECK (
            length(previous_hash) = 64 AND previous_hash NOT GLOB '*[^0-9a-f]*'
        ),
        record_hash TEXT NOT NULL UNIQUE CHECK (
            length(record_hash) = 64 AND record_hash NOT GLOB '*[^0-9a-f]*'
        ),
        retention_class TEXT NOT NULL CHECK (
            retention_class IN ('standard', 'legal_hold')
        ),
        hold_reference TEXT CHECK (
            hold_reference IS NULL
            OR (length(hold_reference) BETWEEN 1 AND 128 AND instr(hold_reference, char(0)) = 0)
        ),
        CHECK (
            (retention_class = 'standard' AND hold_reference IS NULL)
            OR (retention_class = 'legal_hold' AND hold_reference IS NOT NULL)
        )
    ) STRICT
    """,
    """
    CREATE TRIGGER audit_events_immutable_update
    BEFORE UPDATE ON audit_events
    BEGIN
        SELECT RAISE(ABORT, 'audit events are immutable');
    END
    """,
    """
    CREATE TRIGGER audit_events_immutable_delete
    BEFORE DELETE ON audit_events
    BEGIN
        SELECT RAISE(ABORT, 'audit events are immutable');
    END
    """,
    """
    CREATE TRIGGER audit_events_chain
    BEFORE INSERT ON audit_events
    WHEN NOT (
        (
            NOT EXISTS (SELECT 1 FROM audit_events)
            AND NEW.previous_hash =
                '0000000000000000000000000000000000000000000000000000000000000000'
        )
        OR (
            EXISTS (SELECT 1 FROM audit_events)
            AND NEW.previous_hash = (
                SELECT record_hash FROM audit_events ORDER BY audit_id DESC LIMIT 1
            )
        )
    )
    BEGIN
        SELECT RAISE(ABORT, 'audit hash chain is invalid');
    END
    """,
    """
    CREATE TRIGGER audit_events_record_hash
    BEFORE INSERT ON audit_events
    WHEN NEW.record_hash != seeon_audit_record_hash(
        NEW.previous_hash,
        json_object(
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
        )
    )
    BEGIN
        SELECT RAISE(ABORT, 'audit record hash is invalid');
    END
    """,
    """
    CREATE TRIGGER audit_events_capacity
    BEFORE INSERT ON audit_events
    WHEN (SELECT COUNT(*) FROM audit_events) >= 1000000
    BEGIN
        SELECT RAISE(ABORT, 'audit capacity exhausted');
    END
    """,
    "CREATE INDEX audit_events_recorded_idx ON audit_events(recorded_at, audit_id)",
    """
    CREATE INDEX audit_events_target_idx
    ON audit_events(target_type, target_id, recorded_at)
    """,
    "CREATE INDEX audit_events_actor_idx ON audit_events(actor_type, actor_id, recorded_at)",
)

__all__ = ["COMPACT_SCHEMA_CREATE_STATEMENTS"]
