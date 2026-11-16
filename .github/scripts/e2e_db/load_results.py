#!/usr/bin/env python3
"""Load E2E inductor result CSVs into the accuracy/performance PostgreSQL databases.

Rows are routed by their scenario column (database name == scenario name).
Each call belongs to one run (--run/--run-type); shards of the same run may be loaded
separately with their own --meta, and re-loading a case updates it instead of duplicating.
The --meta JSON describes the test environment, e.g.
  {"host": {"hostname": ..., "memory": ..., "disk_root": ..., "os": ..., "kernel": ...},
   "container": {"name": ..., "os": ...},
   "torch": {"name": ..., "version": ..., "commit": ...},
   "python": {"version": ...}}
Optional keys are the remaining columns of the matching table in schema/common.sql.
Connection settings come from E2E_DB_HOST, E2E_DB_PORT, E2E_DB_USER and E2E_DB_PASSWORD.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import psycopg

ENV_COLS = {
    "host": ["hostname", "cpu", "gpu", "memory", "disk_root", "disk_home", "disk_opt", "disk_var",
             "os", "kernel", "compute_runtime", "igc", "l0", "gcc"],
    "container": ["name", "os", "compute_runtime", "igc", "l0", "gcc"],
    "torch": ["name", "version", "commit", "repo", "torchvision", "torchaudio", "torchbench",
              "transformers", "timm", "pytest"],
    "python": ["version"],
}
RUN_TYPES = ["ci", "nightly", "weekly", "acceptance", "on-demand"]
KEY_COLS = ["run_id", "device_id", "backend_id", "suite_id", "dtype_id", "mode_id", "model_id"]


def to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_float(value):
    if value is None:
        return None
    try:
        return float(str(value).strip().rstrip("x"))
    except ValueError:
        return None


def upsert_name(cur, table, name, cache):
    key = (table, name)
    if key not in cache:
        cur.execute(
            f"INSERT INTO {table} (name) VALUES (%s) "
            "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
            (name,),
        )
        cache[key] = cur.fetchone()[0]
    return cache[key]


def upsert_env(cur, table, fields):
    unknown = set(fields) - set(ENV_COLS[table])
    if unknown:
        sys.exit(f"unknown {table} fields in --meta: {sorted(unknown)}")
    cols = ENV_COLS[table]
    col_list = ", ".join(cols)
    cur.execute(
        f"INSERT INTO {table} ({col_list}) VALUES ({', '.join(['%s'] * len(cols))}) "
        f"ON CONFLICT ({col_list}) DO UPDATE SET {cols[0]} = EXCLUDED.{cols[0]} RETURNING id",
        [fields.get(c) for c in cols],
    )
    return cur.fetchone()[0]


def upsert_run(cur, args):
    cur.execute(
        "INSERT INTO run (name, type, started_at, url) VALUES (%s, %s, COALESCE(%s, now()), %s) "
        "ON CONFLICT (name) DO UPDATE SET type = EXCLUDED.type, "
        "started_at = LEAST(run.started_at, EXCLUDED.started_at), url = COALESCE(EXCLUDED.url, run.url) "
        "RETURNING id",
        (args.run, args.run_type, args.started_at or None, args.run_url or None),
    )
    return cur.fetchone()[0]


def build_values(cur, db, env_ids, row, cache):
    values = {
        **env_ids,
        "device_id": upsert_name(cur, "device", row["dev"], cache),
        "backend_id": upsert_name(cur, "backend", row.get("backend") or "inductor", cache),
        "suite_id": upsert_name(cur, "suite", row["suite"], cache),
        "dtype_id": upsert_name(cur, "dtype", row["dtype"], cache),
        "mode_id": upsert_name(cur, "mode", row["mode"], cache),
        "model_id": upsert_name(cur, "model", row["name"], cache),
        # Failed cases record batch size 0 as a placeholder.
        "batch_size": to_int(row.get("batch_size")) or None,
    }
    status = row.get("status")
    if db == "accuracy":
        result = row.get("accuracy") or status
        values["result_id"] = upsert_name(cur, "result", result, cache) if result else None
    else:
        values["result_id"] = upsert_name(cur, "result", status, cache) if status else None
        # Failed cases carry 0 placeholders; keep them NULL so they never look like real timings.
        failed = status not in (None, "", "pass")
        inductor = None if failed else to_float(row.get("abs_latency")) or None
        eager = None if failed else to_float(row.get("eager_latency")) or None
        speedup = to_float(row.get("speedup"))
        if eager is None and inductor is not None and speedup:
            eager = inductor * speedup
        values["eager_latency"] = eager
        values["inductor_latency"] = inductor
    return values


def write_rows(conn, db, args, meta, rows):
    with conn, conn.cursor() as cur:
        cache = {}
        env_ids = {"run_id": upsert_run(cur, args)}
        env_ids.update({f"{table}_id": upsert_env(cur, table, meta.get(table, {})) for table in ENV_COLS})
        for row in rows:
            values = build_values(cur, db, env_ids, row, cache)
            cols = list(values)
            updates = [c for c in cols if c not in KEY_COLS]
            cur.execute(
                f"INSERT INTO {db} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
                f"ON CONFLICT ({', '.join(KEY_COLS)}) DO UPDATE SET "
                f"{', '.join(f'{c} = EXCLUDED.{c}' for c in updates)}, created_at = now()",
                [values[c] for c in cols],
            )


def load_db(db, args, meta, rows):
    # Driver errors carry the server address and row values; keep them out of public CI logs.
    try:
        conn = psycopg.connect(
            host=os.environ["E2E_DB_HOST"],
            port=int(os.environ.get("E2E_DB_PORT") or "5432"),
            user=os.environ.get("E2E_DB_USER") or "e2e",
            password=os.environ["E2E_DB_PASSWORD"],
            dbname=db,
            connect_timeout=15,
        )
    except psycopg.Error:
        sys.exit(f"{db}: cannot connect to the E2E database")
    try:
        write_rows(conn, db, args, meta, rows)
    except psycopg.Error as e:
        sys.exit(f"{db}: database error {type(e).__name__} (sqlstate {e.sqlstate})")
    print(f"{db}: loaded {len(rows)} rows for {args.run_type} run '{args.run}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="unique run name, e.g. the GitHub run id")
    parser.add_argument("--run-type", required=True, choices=RUN_TYPES)
    parser.add_argument("--run-url", help="link to the CI job")
    parser.add_argument("--started-at", help="run start time (ISO 8601); defaults to now")
    parser.add_argument("--meta", required=True, type=Path, help="JSON file describing host/container/torch/python")
    parser.add_argument("--input", required=True, type=Path, help="directory searched recursively for inductor-*.csv")
    args = parser.parse_args()
    meta = json.loads(args.meta.read_text(encoding="utf-8"))

    files = sorted(args.input.rglob("inductor-*.csv"))
    if not files:
        sys.exit(f"no inductor-*.csv files under {args.input}")

    batches = {"accuracy": [], "performance": []}
    for path in files:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("scenario") in batches:
                    batches[row["scenario"]].append(row)

    for db, rows in batches.items():
        if rows:
            load_db(db, args, meta, rows)


if __name__ == "__main__":
    main()
