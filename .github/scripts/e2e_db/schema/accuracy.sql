CREATE TABLE IF NOT EXISTS result (
    id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS accuracy (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id integer NOT NULL REFERENCES run (id),
    host_id integer NOT NULL REFERENCES host (id),
    container_id integer NOT NULL REFERENCES container (id),
    torch_id integer NOT NULL REFERENCES torch (id),
    python_id smallint NOT NULL REFERENCES python (id),
    device_id smallint NOT NULL REFERENCES device (id),
    backend_id smallint NOT NULL REFERENCES backend (id),
    suite_id smallint NOT NULL REFERENCES suite (id),
    dtype_id smallint NOT NULL REFERENCES dtype (id),
    mode_id smallint NOT NULL REFERENCES mode (id),
    model_id integer NOT NULL REFERENCES model (id),
    batch_size integer,
    result_id smallint REFERENCES result (id),
    issue_id integer REFERENCES issue (id),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, device_id, backend_id, suite_id, dtype_id, mode_id, model_id)
);
CREATE INDEX IF NOT EXISTS accuracy_key_idx ON accuracy (suite_id, dtype_id, mode_id, model_id, run_id);

CREATE OR REPLACE VIEW v_accuracy AS
SELECT a.id, r.name AS run, r.type AS run_type, r.started_at, h.hostname, c.name AS container,
       t.name AS torch, t.version AS torch_version, t.commit AS torch_commit, p.version AS python,
       dv.name AS device, b.name AS backend,
       s.name AS suite, d.name AS dtype, m.name AS mode, mo.name AS model, a.batch_size,
       rs.name AS result, i.repo AS issue_repo, i.number AS issue_number, a.created_at
FROM accuracy a
JOIN run r ON r.id = a.run_id
JOIN host h ON h.id = a.host_id
JOIN container c ON c.id = a.container_id
JOIN torch t ON t.id = a.torch_id
JOIN python p ON p.id = a.python_id
JOIN device dv ON dv.id = a.device_id
JOIN backend b ON b.id = a.backend_id
JOIN suite s ON s.id = a.suite_id
JOIN dtype d ON d.id = a.dtype_id
JOIN mode m ON m.id = a.mode_id
JOIN model mo ON mo.id = a.model_id
LEFT JOIN result rs ON rs.id = a.result_id
LEFT JOIN issue i ON i.id = a.issue_id;
