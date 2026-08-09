"""SQLite schema and statements for immutable clip listing generations."""

CREATE_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS clip_listing_generation (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        active_generation INTEGER NOT NULL,
        next_generation INTEGER NOT NULL
    ) STRICT""",
    """INSERT OR IGNORE INTO clip_listing_generation
        (id, active_generation, next_generation) VALUES (1, 0, 1)""",
    """CREATE TABLE IF NOT EXISTS clip_listing_rows (
        generation INTEGER NOT NULL,
        clip_id TEXT NOT NULL,
        manifest_path TEXT NOT NULL,
        manifest_mtime_ns INTEGER NOT NULL,
        manifest_size_bytes INTEGER NOT NULL,
        camera_id TEXT NOT NULL,
        event_ref TEXT NOT NULL,
        event_type TEXT,
        event_facet TEXT NOT NULL CHECK (event_facet IN ('fall','bed-exit','other')),
        started_at TEXT NOT NULL,
        duration_s REAL NOT NULL,
        codec TEXT NOT NULL,
        media_path TEXT,
        video_available INTEGER NOT NULL,
        video_error TEXT,
        finalized INTEGER NOT NULL,
        size_bytes INTEGER,
        PRIMARY KEY (generation, clip_id),
        UNIQUE (generation, manifest_path)
    ) STRICT""",
    """CREATE TABLE IF NOT EXISTS clip_listing_summary (
        generation INTEGER NOT NULL,
        camera_id TEXT NOT NULL,
        event_facet TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY (generation, camera_id, event_facet)
    ) STRICT""",
    """CREATE INDEX IF NOT EXISTS clip_listing_global_order_idx
        ON clip_listing_rows(generation, started_at DESC, clip_id DESC)""",
    """CREATE INDEX IF NOT EXISTS clip_listing_global_facet_order_idx
        ON clip_listing_rows(generation, event_facet, started_at DESC, clip_id DESC)""",
    """CREATE INDEX IF NOT EXISTS clip_listing_camera_order_idx
        ON clip_listing_rows(generation, camera_id, started_at DESC, clip_id DESC)""",
    """CREATE INDEX IF NOT EXISTS clip_listing_camera_facet_order_idx
        ON clip_listing_rows(
            generation, camera_id, event_facet, started_at DESC, clip_id DESC
        )""",
)
SELECT_ACTIVE_GENERATION = (
    "SELECT active_generation FROM clip_listing_generation WHERE id = 1"
)
SELECT_ACTIVE_CLIPS = """SELECT manifest_path, manifest_mtime_ns,
    manifest_size_bytes, clip_id, camera_id, event_ref, event_type, event_facet,
    started_at, duration_s, codec, media_path, video_available, video_error,
    finalized, size_bytes FROM clip_listing_rows
    WHERE generation = (SELECT active_generation FROM clip_listing_generation WHERE id = 1)"""
INSERT_CLIP = "INSERT INTO clip_listing_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
INSERT_SUMMARY = "INSERT INTO clip_listing_summary VALUES (?,?,?,?)"
SELECT_NEXT_GENERATION = "SELECT next_generation FROM clip_listing_generation WHERE id = 1"
ADVANCE_NEXT_GENERATION = (
    "UPDATE clip_listing_generation SET next_generation = next_generation + 1 WHERE id = 1"
)
ACTIVATE_GENERATION = (
    "UPDATE clip_listing_generation SET active_generation = ? WHERE id = 1"
)
DELETE_OLD_ROWS = "DELETE FROM clip_listing_rows WHERE generation != ?"
DELETE_OLD_SUMMARIES = "DELETE FROM clip_listing_summary WHERE generation != ?"

__all__ = [
    "ACTIVATE_GENERATION",
    "ADVANCE_NEXT_GENERATION",
    "CREATE_STATEMENTS",
    "DELETE_OLD_ROWS",
    "DELETE_OLD_SUMMARIES",
    "INSERT_CLIP",
    "INSERT_SUMMARY",
    "SELECT_ACTIVE_CLIPS",
    "SELECT_ACTIVE_GENERATION",
    "SELECT_NEXT_GENERATION",
]
