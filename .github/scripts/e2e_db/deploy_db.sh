#!/usr/bin/env bash
# Start the E2E results PostgreSQL container on the host. Data lives in a host bind mount.
# Usage (on skx-3725): ./deploy_db.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="${E2E_DB_DIR:-/home/jenkins/e2e-db}"
CONTAINER="${E2E_DB_CONTAINER:-e2e-results-pg}"
IMAGE="${E2E_DB_IMAGE:-postgres:16}"
PORT="${E2E_DB_PORT:-5432}"
DB_USER=e2e
ENV_FILE="$DB_DIR/postgres.env"

mkdir -p "$DB_DIR/pgdata"
if [[ ! -f "$ENV_FILE" ]]; then
    umask 077
    printf 'POSTGRES_USER=%s\nPOSTGRES_PASSWORD=%s\nPOSTGRES_DB=postgres\n' "$DB_USER" "$(openssl rand -hex 24)" > "$ENV_FILE"
    echo "Created $ENV_FILE (readable by owner only)"
fi

# The postgres image runs as uid 999; the bind-mounted data dir must be owned by it.
docker run --rm --entrypoint chown -v "$DB_DIR/pgdata:/var/lib/postgresql/data" "$IMAGE" -R 999:999 /var/lib/postgresql/data

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    docker update --restart always "$CONTAINER" >/dev/null
    docker start "$CONTAINER" >/dev/null
else
    docker run -d --name "$CONTAINER" --restart always \
        --env-file "$ENV_FILE" \
        -p "${PORT}:5432" \
        -v "$DB_DIR/pgdata:/var/lib/postgresql/data" \
        "$IMAGE" >/dev/null
fi

for _ in $(seq 1 60); do
    if docker exec "$CONTAINER" pg_isready -U "$DB_USER" -q; then
        break
    fi
    sleep 1
done
docker exec "$CONTAINER" pg_isready -U "$DB_USER"

docker exec -i "$CONTAINER" psql -v ON_ERROR_STOP=1 -q -U "$DB_USER" -d postgres <<'SQL'
SELECT 'CREATE DATABASE accuracy' WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'accuracy')\gexec
SELECT 'CREATE DATABASE performance' WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'performance')\gexec
SQL

for db in accuracy performance; do
    cat "$SCRIPT_DIR/schema/common.sql" "$SCRIPT_DIR/schema/$db.sql" |
        docker exec -i "$CONTAINER" psql -v ON_ERROR_STOP=1 -q -U "$DB_USER" -d "$db"
    echo "Schema applied to database: $db"
done
