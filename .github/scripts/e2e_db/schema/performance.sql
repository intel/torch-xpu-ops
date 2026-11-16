CREATE TABLE IF NOT EXISTS performance (
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
    eager_latency real,
    inductor_latency real,
    issue_id integer REFERENCES issue (id),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, device_id, backend_id, suite_id, dtype_id, mode_id, model_id)
);
-- Upgrades databases created before result_id existed.
ALTER TABLE performance ADD COLUMN IF NOT EXISTS result_id smallint REFERENCES result (id);
CREATE INDEX IF NOT EXISTS performance_key_idx ON performance (suite_id, dtype_id, mode_id, model_id, run_id);

DROP VIEW IF EXISTS v_performance;
CREATE VIEW v_performance AS
SELECT pf.id, r.name AS run, r.type AS run_type, r.started_at, h.hostname, c.name AS container,
       t.name AS torch, t.version AS torch_version, t.commit AS torch_commit, p.version AS python,
       dv.name AS device, b.name AS backend,
       s.name AS suite, d.name AS dtype, m.name AS mode, mo.name AS model, pf.batch_size,
       rs.name AS result, pf.eager_latency, pf.inductor_latency,
       pf.eager_latency / NULLIF(pf.inductor_latency, 0) AS speedup,
       i.repo AS issue_repo, i.number AS issue_number, pf.created_at
FROM performance pf
JOIN run r ON r.id = pf.run_id
JOIN host h ON h.id = pf.host_id
JOIN container c ON c.id = pf.container_id
JOIN torch t ON t.id = pf.torch_id
JOIN python p ON p.id = pf.python_id
JOIN device dv ON dv.id = pf.device_id
JOIN backend b ON b.id = pf.backend_id
JOIN suite s ON s.id = pf.suite_id
JOIN dtype d ON d.id = pf.dtype_id
JOIN mode m ON m.id = pf.mode_id
JOIN model mo ON mo.id = pf.model_id
LEFT JOIN result rs ON rs.id = pf.result_id
LEFT JOIN issue i ON i.id = pf.issue_id;
