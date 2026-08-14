"""SQLite schema and statements for immutable clip listing generations."""

CREATE_GENERATION_TABLE = """CREATE TABLE IF NOT EXISTS clip_listing_generation (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        active_generation INTEGER NOT NULL,
        next_generation INTEGER NOT NULL
    ) STRICT"""
INITIALIZE_GENERATION = """INSERT OR IGNORE INTO clip_listing_generation
        (id, active_generation, next_generation) VALUES (1, 0, 1)"""
CREATE_ROWS_TABLE = """CREATE TABLE IF NOT EXISTS clip_listing_rows (
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
    ) STRICT"""
KNOWN_DRAFT_ROWS_TABLE = CREATE_ROWS_TABLE.replace(
    "        size_bytes INTEGER,\n",
    "        size_bytes INTEGER,\n"
    "        thumbnail_mtime_ns INTEGER,\n"
    "        thumbnail_size_bytes INTEGER,\n"
    "        thumbnail_available INTEGER NOT NULL DEFAULT 0,\n",
)
CREATE_THUMBNAILS_TABLE = """CREATE TABLE IF NOT EXISTS clip_listing_thumbnails (
        generation INTEGER NOT NULL,
        clip_id TEXT NOT NULL,
        thumbnail_mtime_ns INTEGER,
        thumbnail_size_bytes INTEGER,
        thumbnail_available INTEGER NOT NULL,
        PRIMARY KEY (generation, clip_id)
    ) STRICT"""
CREATE_SUMMARY_TABLE = """CREATE TABLE IF NOT EXISTS clip_listing_summary (
        generation INTEGER NOT NULL,
        camera_id TEXT NOT NULL,
        event_facet TEXT NOT NULL,
        count INTEGER NOT NULL,
        PRIMARY KEY (generation, camera_id, event_facet)
    ) STRICT"""
CREATE_INDEX_STATEMENTS = (
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
CREATE_STATEMENTS = (
    CREATE_GENERATION_TABLE,
    INITIALIZE_GENERATION,
    CREATE_ROWS_TABLE,
    CREATE_THUMBNAILS_TABLE,
    CREATE_SUMMARY_TABLE,
    *CREATE_INDEX_STATEMENTS,
)
SELECT_ACTIVE_GENERATION = (
    "SELECT active_generation FROM clip_listing_generation WHERE id = 1"
)
SELECT_ACTIVE_CLIPS = """SELECT rows.manifest_path, rows.manifest_mtime_ns,
    rows.manifest_size_bytes, rows.clip_id, rows.camera_id, rows.event_ref,
    rows.event_type, rows.event_facet, rows.started_at, rows.duration_s, rows.codec,
    rows.media_path, rows.video_available, rows.video_error, rows.finalized,
    rows.size_bytes, thumbnails.thumbnail_mtime_ns, thumbnails.thumbnail_size_bytes,
    COALESCE(thumbnails.thumbnail_available, 0)
    FROM clip_listing_rows AS rows
    LEFT JOIN clip_listing_thumbnails AS thumbnails
      ON thumbnails.generation = rows.generation AND thumbnails.clip_id = rows.clip_id
    WHERE rows.generation = (
        SELECT active_generation FROM clip_listing_generation WHERE id = 1
    )"""
INSERT_CLIP = """INSERT INTO clip_listing_rows (
    generation, clip_id, manifest_path, manifest_mtime_ns, manifest_size_bytes,
    camera_id, event_ref, event_type, event_facet, started_at, duration_s, codec,
    media_path, video_available, video_error, finalized, size_bytes
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
INSERT_THUMBNAIL = """INSERT INTO clip_listing_thumbnails (
    generation, clip_id, thumbnail_mtime_ns, thumbnail_size_bytes, thumbnail_available
) VALUES (?,?,?,?,?)"""
INSERT_SUMMARY = "INSERT INTO clip_listing_summary VALUES (?,?,?,?)"
SELECT_NEXT_GENERATION = "SELECT next_generation FROM clip_listing_generation WHERE id = 1"
ADVANCE_NEXT_GENERATION = (
    "UPDATE clip_listing_generation SET next_generation = next_generation + 1 WHERE id = 1"
)
ACTIVATE_GENERATION = (
    "UPDATE clip_listing_generation SET active_generation = ? WHERE id = 1"
)
DELETE_OLD_ROWS = "DELETE FROM clip_listing_rows WHERE generation != ?"
DELETE_OLD_THUMBNAILS = "DELETE FROM clip_listing_thumbnails WHERE generation != ?"
DELETE_OLD_SUMMARIES = "DELETE FROM clip_listing_summary WHERE generation != ?"

__all__ = [
    "ACTIVATE_GENERATION",
    "ADVANCE_NEXT_GENERATION",
    "CREATE_GENERATION_TABLE",
    "CREATE_INDEX_STATEMENTS",
    "CREATE_ROWS_TABLE",
    "CREATE_SUMMARY_TABLE",
    "CREATE_STATEMENTS",
    "CREATE_THUMBNAILS_TABLE",
    "DELETE_OLD_ROWS",
    "DELETE_OLD_SUMMARIES",
    "DELETE_OLD_THUMBNAILS",
    "INSERT_CLIP",
    "INSERT_SUMMARY",
    "INSERT_THUMBNAIL",
    "KNOWN_DRAFT_ROWS_TABLE",
    "SELECT_ACTIVE_CLIPS",
    "SELECT_ACTIVE_GENERATION",
    "SELECT_NEXT_GENERATION",
]
