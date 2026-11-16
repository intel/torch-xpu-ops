# E2E results database

Stores every E2E inductor run (CI, nightly, weekly, acceptance, on-demand) in PostgreSQL so runs
can be compared later, e.g. CI vs. nightly, nightly vs. previous nightly, weekly vs. previous
weekly, on-demand vs. latest weekly, and so E2E summary reports can query history.

| File | Purpose |
| --- | --- |
| `schema/common.sql` | Lookup tables shared by both databases |
| `schema/accuracy.sql`, `schema/performance.sql` | Main result table and readable view per database |
| `deploy_db.sh` | Starts the PostgreSQL container on the database host and applies the schema |
| `collect_env.py` | Writes the test-environment JSON (`--meta`) in the test environment |
| `load_results.py` | Loads `inductor-*.csv` results plus the environment JSON into the databases |

## Schema

There are two databases, `accuracy` and `performance`, with the same layout. Rows are routed by the
CSV `scenario` column.

Lookup tables (`common.sql`) store each value once; main tables only hold small integer ids:

- `run`: one row per test run. `name` is unique (GitHub run id, plus `-target`/`-baseline` for
  acceptance), `type` is one of `ci`, `nightly`, `weekly`, `acceptance`, `on-demand`, and
  `started_at`/`url` say when and where it ran. Shards of one run share the row.
- `host`, `container`, `torch`, `python`: the machine and software environment. A new row is
  created only when any field changes (e.g. a driver upgrade), so environment changes are visible
  per run.
- `device`, `backend`, `suite`, `dtype`, `mode`, `model`, `result`: names of the case dimensions
  and outcomes (`pass` or the failure reason).
- `issue`: GitHub issues that can be linked to a case through `issue_id`.

Main tables:

- `accuracy`: `result_id` per case.
- `performance`: `result_id`, `eager_latency` and `inductor_latency` per case. Latencies are NULL
  when the case failed; speedup is `eager_latency / inductor_latency`.

Each case is unique per `(run, device, backend, suite, dtype, mode, model)`, so re-loading a run
updates rows instead of duplicating them. That unique index serves run-vs-run comparisons, and
`(suite, dtype, mode, model, run)` serves the history of a single case. Use the views
`v_accuracy` / `v_performance` for readable output.

## CI flow

`.github/workflows/_linux_e2e.yml`:

1. The `test` job runs `collect_env.py` after the tests. The JSON is encrypted with
   `E2E_DB_PASSWORD` and added to the (public) result artifact as `e2e-env.json.enc`.
2. The `database` job on `xpu-ubuntu-24.04` downloads the artifact, decrypts the environment and
   runs `load_results.py`. It is non-blocking and skips when the secrets are unavailable
   (e.g. pull requests from forks).

Required repository secrets: `E2E_DB_HOST` and `E2E_DB_PASSWORD` (the `POSTGRES_PASSWORD` that
`deploy_db.sh` generated). Optional environment overrides: `E2E_DB_PORT` (5432), `E2E_DB_USER`
(`e2e`). The loader never prints connection details or database error text, because CI logs are
public.

## Deploy or upgrade

On the database host:

```bash
./deploy_db.sh
```

The script is idempotent: it creates the container on first use (with a random password in
`${E2E_DB_DIR:-$HOME/e2e-db}/postgres.env`, readable by the owner only), otherwise starts it, and
then applies the schema to both databases. Schema files only add tables, columns, indexes and
views, so re-running them on an existing database is safe.

## Load manually

```bash
python collect_env.py --output env.json            # in the test environment
E2E_DB_HOST=<host> E2E_DB_PASSWORD=<password> \
python load_results.py --run <name> --run-type nightly \
    --meta env.json --input <dir with inductor-*.csv> [--started-at <ISO 8601>] [--run-url <url>]
```

## Example: latest nightly vs. previous nightly

```sql
WITH n AS (
    SELECT id, row_number() OVER (ORDER BY started_at DESC) AS rn FROM run WHERE type = 'nightly'
)
SELECT s.name AS suite, d.name AS dtype, m.name AS mode, mo.name AS model,
       rb.name AS baseline, rt.name AS target
FROM accuracy t
JOIN accuracy b USING (device_id, backend_id, suite_id, dtype_id, mode_id, model_id)
JOIN suite s ON s.id = t.suite_id
JOIN dtype d ON d.id = t.dtype_id
JOIN mode m ON m.id = t.mode_id
JOIN model mo ON mo.id = t.model_id
LEFT JOIN result rt ON rt.id = t.result_id
LEFT JOIN result rb ON rb.id = b.result_id
WHERE t.run_id = (SELECT id FROM n WHERE rn = 1)
  AND b.run_id = (SELECT id FROM n WHERE rn = 2)
  AND rt.name IS DISTINCT FROM rb.name;
```

To compare other runs, pick the two run ids from `run` by `type` and `started_at` (or by `name`).

## Limitations

- Inside a container only the root disk is visible, and the user-space driver stack belongs to the
  image, so those fields go to `container` and the host's disk and driver fields stay NULL.
- PT2E quantization variants (symm/asymm) of the same model share one case key, so the last loaded
  variant wins.
