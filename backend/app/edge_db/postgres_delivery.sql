-- Product namespace only. Bootstrap explicitly installs the writer/sender token.
-- No default authority, receiver acceptance, or irreversible queue deletion.
CREATE TABLE deployment_authority (
    singleton integer PRIMARY KEY CHECK (singleton = 1),
    generation bigint NOT NULL CHECK (generation > 0),
    writer_token uuid NOT NULL,
    accepting boolean NOT NULL,
    egress_enabled boolean NOT NULL
);

CREATE TABLE event_outbox (
    edge_event_id text PRIMARY KEY REFERENCES incidents(edge_event_id),
    envelope text NOT NULL CHECK (envelope IS JSON OBJECT WITH UNIQUE KEYS),
    envelope_sha256 text NOT NULL CHECK (envelope_sha256 ~ '^[0-9a-f]{64}$'),
    envelope_bytes bigint NOT NULL CHECK (
        envelope_bytes = octet_length(envelope) AND envelope_bytes BETWEEN 2 AND 524288
    ),
    backend_camera_id text CHECK (length(backend_camera_id) BETWEEN 1 AND 128),
    state text NOT NULL CHECK (state IN ('LOCAL_ONLY', 'PENDING', 'IN_FLIGHT', 'SENT', 'REJECTED', 'EXHAUSTED')),
    accepted_generation bigint NOT NULL CHECK (accepted_generation > 0),
    accepted_at timestamptz NOT NULL,
    attempt_count bigint NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    retry_at timestamptz NOT NULL,
    active_attempt uuid,
    lease_until timestamptz,
    CHECK ((state = 'IN_FLIGHT') = (active_attempt IS NOT NULL AND lease_until IS NOT NULL)),
    CHECK ((active_attempt IS NULL) = (lease_until IS NULL)),
    CHECK (state = 'LOCAL_ONLY' OR backend_camera_id IS NOT NULL),
    CHECK (envelope_sha256 = encode(sha256(convert_to(envelope, 'UTF8')), 'hex'))
);
CREATE INDEX event_outbox_ready ON event_outbox(retry_at, accepted_at, edge_event_id)
    WHERE state IN ('PENDING', 'IN_FLIGHT');

CREATE TABLE event_delivery_attempts (
    attempt_id uuid PRIMARY KEY,
    edge_event_id text NOT NULL REFERENCES event_outbox(edge_event_id),
    ordinal bigint NOT NULL CHECK (ordinal > 0),
    writer_generation bigint NOT NULL CHECK (writer_generation > 0),
    started_at timestamptz NOT NULL,
    UNIQUE (edge_event_id, ordinal),
    UNIQUE (attempt_id, edge_event_id, ordinal)
);
-- Claims insert their attempt before advancing the outbox, so no deferral is needed.
ALTER TABLE event_outbox ADD CONSTRAINT event_outbox_active_attempt_fk
    FOREIGN KEY (active_attempt, edge_event_id, attempt_count)
    REFERENCES event_delivery_attempts(attempt_id, edge_event_id, ordinal);

CREATE TABLE event_delivery_results (
    attempt_id uuid PRIMARY KEY REFERENCES event_delivery_attempts(attempt_id),
    finished_at timestamptz NOT NULL,
    outcome text NOT NULL CHECK (outcome IN ('SENT', 'RETRY', 'REJECTED', 'UNKNOWN')),
    reason text NOT NULL CHECK (reason ~ '^[A-Z][A-Z0-9_]{0,63}$'),
    http_status integer CHECK (http_status BETWEEN 100 AND 599),
    backend_event_id text CHECK (length(backend_event_id) BETWEEN 1 AND 128),
    CHECK ((outcome = 'SENT') = (backend_event_id IS NOT NULL))
);

-- One actual request completion per attempt, separate from automatic lease expiry.
-- Only classified, bounded metadata is retained; never response bodies or credentials.
CREATE TABLE event_delivery_observations (
    attempt_id uuid PRIMARY KEY,
    edge_event_id text NOT NULL,
    ordinal bigint NOT NULL CHECK (ordinal > 0),
    observed_at timestamptz NOT NULL,
    outcome text NOT NULL CHECK (outcome IN ('SENT', 'RETRY', 'REJECTED', 'UNKNOWN')),
    reason text NOT NULL CHECK (reason ~ '^[A-Z][A-Z0-9_]{0,63}$'),
    http_status integer CHECK (http_status BETWEEN 100 AND 599),
    backend_event_id text CHECK (length(backend_event_id) BETWEEN 1 AND 128),
    FOREIGN KEY (attempt_id, edge_event_id, ordinal)
        REFERENCES event_delivery_attempts(attempt_id, edge_event_id, ordinal),
    CHECK ((outcome = 'SENT') = (backend_event_id IS NOT NULL))
);

CREATE FUNCTION seeon_outbox_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
    IF TG_OP <> 'UPDATE' THEN
        RAISE EXCEPTION 'accepted delivery history is retained' USING ERRCODE = '23514';
    END IF;
    IF (NEW.edge_event_id, NEW.envelope, NEW.envelope_sha256, NEW.envelope_bytes,
        NEW.backend_camera_id, NEW.accepted_generation, NEW.accepted_at)
        IS DISTINCT FROM
       (OLD.edge_event_id, OLD.envelope, OLD.envelope_sha256, OLD.envelope_bytes,
        OLD.backend_camera_id, OLD.accepted_generation, OLD.accepted_at) THEN
        RAISE EXCEPTION 'accepted envelope identity is immutable' USING ERRCODE = '23514';
    END IF;
    IF OLD.state IN ('LOCAL_ONLY', 'SENT', 'REJECTED', 'EXHAUSTED') AND NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'terminal delivery state is immutable' USING ERRCODE = '23514';
    END IF;
    IF NEW.attempt_count NOT IN (OLD.attempt_count, OLD.attempt_count + 1) THEN
        RAISE EXCEPTION 'delivery ordinal must advance by one' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER event_outbox_update_guard BEFORE UPDATE OR DELETE ON event_outbox
    FOR EACH ROW EXECUTE FUNCTION seeon_outbox_guard();
CREATE TRIGGER event_outbox_truncate_guard BEFORE TRUNCATE ON event_outbox
    FOR EACH STATEMENT EXECUTE FUNCTION seeon_outbox_guard();
CREATE TRIGGER event_attempts_immutable BEFORE UPDATE OR DELETE ON event_delivery_attempts
    FOR EACH ROW EXECUTE FUNCTION seeon_audit_immutable();
CREATE TRIGGER event_attempts_no_truncate BEFORE TRUNCATE ON event_delivery_attempts
    FOR EACH STATEMENT EXECUTE FUNCTION seeon_audit_immutable();
CREATE TRIGGER event_results_immutable BEFORE UPDATE OR DELETE ON event_delivery_results
    FOR EACH ROW EXECUTE FUNCTION seeon_audit_immutable();
CREATE TRIGGER event_results_no_truncate BEFORE TRUNCATE ON event_delivery_results
    FOR EACH STATEMENT EXECUTE FUNCTION seeon_audit_immutable();
CREATE TRIGGER event_observations_immutable BEFORE UPDATE OR DELETE ON event_delivery_observations
    FOR EACH ROW EXECUTE FUNCTION seeon_audit_immutable();
CREATE TRIGGER event_observations_no_truncate BEFORE TRUNCATE ON event_delivery_observations
    FOR EACH STATEMENT EXECUTE FUNCTION seeon_audit_immutable();
