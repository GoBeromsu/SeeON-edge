"""Migration-only promotion and classification of legacy clip labels."""

from __future__ import annotations

import sqlite3
from typing import Final

LEGACY_LABEL_MIGRATION_STATEMENTS: Final = (
    """
    INSERT INTO control_evidence_review_revisions (
        review_id, incident_id, clip_id, review_version, actor_id,
        reviewed_at, disposition, notes
    )
    SELECT 'review:' || lower(hex(randomblob(16))), primary_record.incident_id,
           legacy.clip_id, 1, legacy.reviewer, legacy.reviewed_at, legacy.label, NULL
    FROM labels AS legacy
    JOIN evidence_primary_clips AS primary_record ON primary_record.clip_id = legacy.clip_id
    WHERE legacy.label IN ('TRUE_POSITIVE', 'FALSE_POSITIVE')
      AND length(legacy.reviewer) BETWEEN 1 AND 128
      AND instr(legacy.reviewer, char(0)) = 0
      AND length(legacy.reviewed_at) BETWEEN 1 AND 64
      AND instr(legacy.reviewed_at, char(0)) = 0
      AND (SELECT count(*) FROM evidence_primary_clips AS candidate
           WHERE candidate.clip_id = legacy.clip_id) = 1
      AND NOT EXISTS (
          SELECT 1 FROM control_evidence_review_state AS current
          WHERE current.incident_id = primary_record.incident_id
      )
      AND NOT EXISTS (
          SELECT 1 FROM control_legacy_label_migrations AS classified
          WHERE classified.source_clip_id = legacy.clip_id
      )
    """,
    """
    INSERT INTO control_evidence_review_state (incident_id, clip_id, current_version)
    SELECT revision.incident_id, revision.clip_id, 1
    FROM control_evidence_review_revisions AS revision
    JOIN labels AS legacy
      ON legacy.clip_id = revision.clip_id
     AND legacy.reviewer = revision.actor_id
     AND legacy.reviewed_at = revision.reviewed_at
     AND legacy.label = revision.disposition
    WHERE revision.review_version = 1
      AND NOT EXISTS (
          SELECT 1 FROM control_evidence_review_state AS current
          WHERE current.incident_id = revision.incident_id
      )
      AND NOT EXISTS (
          SELECT 1 FROM control_legacy_label_migrations AS classified
          WHERE classified.source_clip_id = legacy.clip_id
      )
    """,
    """
    INSERT INTO control_legacy_label_migrations (
        source_clip_id, classification, incident_id, review_id
    )
    SELECT legacy.clip_id,
           CASE
             WHEN legacy.label NOT IN ('TRUE_POSITIVE', 'FALSE_POSITIVE')
                  OR legacy.label IS NULL THEN 'UNSUPPORTED_DISPOSITION'
             WHEN length(legacy.reviewer) NOT BETWEEN 1 AND 128
                  OR instr(legacy.reviewer, char(0)) != 0
                  OR length(legacy.reviewed_at) NOT BETWEEN 1 AND 64
                  OR instr(legacy.reviewed_at, char(0)) != 0 THEN 'UNSAFE_METADATA'
             WHEN NOT EXISTS (
                  SELECT 1 FROM evidence_clips AS clip
                  WHERE clip.clip_id = legacy.clip_id) THEN 'ORPHAN_CLIP'
             WHEN NOT EXISTS (
                  SELECT 1 FROM evidence_primary_clips AS primary_record
                  WHERE primary_record.clip_id = legacy.clip_id) THEN 'ORPHAN_INCIDENT'
             WHEN (SELECT count(*) FROM evidence_primary_clips AS primary_record
                   WHERE primary_record.clip_id = legacy.clip_id) != 1
                  THEN 'AMBIGUOUS_INCIDENT'
             WHEN EXISTS (
                  SELECT 1 FROM control_evidence_review_state AS current
                  JOIN control_evidence_review_revisions AS revision
                    ON revision.incident_id = current.incident_id
                   AND revision.clip_id = current.clip_id
                   AND revision.review_version = current.current_version
                  WHERE current.clip_id = legacy.clip_id
                    AND current.current_version = 1
                    AND revision.actor_id = legacy.reviewer
                    AND revision.reviewed_at = legacy.reviewed_at
                    AND revision.disposition = legacy.label) THEN 'MIGRATED'
             ELSE 'REVIEW_EXISTS'
           END,
           CASE WHEN EXISTS (
                  SELECT 1 FROM control_evidence_review_state AS current
                  JOIN control_evidence_review_revisions AS revision
                    ON revision.incident_id = current.incident_id
                   AND revision.clip_id = current.clip_id
                   AND revision.review_version = current.current_version
                  WHERE current.clip_id = legacy.clip_id
                    AND current.current_version = 1
                    AND revision.actor_id = legacy.reviewer
                    AND revision.reviewed_at = legacy.reviewed_at
                    AND revision.disposition = legacy.label)
                THEN (SELECT current.incident_id
                      FROM control_evidence_review_state AS current
                      WHERE current.clip_id = legacy.clip_id)
                ELSE NULL END,
           CASE WHEN EXISTS (
                  SELECT 1 FROM control_evidence_review_state AS current
                  JOIN control_evidence_review_revisions AS revision
                    ON revision.incident_id = current.incident_id
                   AND revision.clip_id = current.clip_id
                   AND revision.review_version = current.current_version
                  WHERE current.clip_id = legacy.clip_id
                    AND current.current_version = 1
                    AND revision.actor_id = legacy.reviewer
                    AND revision.reviewed_at = legacy.reviewed_at
                    AND revision.disposition = legacy.label)
                THEN (SELECT revision.review_id
                      FROM control_evidence_review_state AS current
                      JOIN control_evidence_review_revisions AS revision
                        ON revision.incident_id = current.incident_id
                       AND revision.clip_id = current.clip_id
                       AND revision.review_version = current.current_version
                      WHERE current.clip_id = legacy.clip_id)
                ELSE NULL END
    FROM labels AS legacy
    WHERE NOT EXISTS (
        SELECT 1 FROM control_legacy_label_migrations AS classified
        WHERE classified.source_clip_id = legacy.clip_id
    )
    """,
)


def classify_legacy_labels(connection: sqlite3.Connection) -> None:
    """Classify newly imported labels once; never reconsider an explicit outcome."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in LEGACY_LABEL_MIGRATION_STATEMENTS:
            connection.execute(statement)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


__all__ = ["LEGACY_LABEL_MIGRATION_STATEMENTS", "classify_legacy_labels"]
