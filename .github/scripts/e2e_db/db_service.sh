#!/usr/bin/env bash
# E2E results PostgreSQL service: start, keep alive, back up and restore.
#
# Usage (on the database host):
#   db_service.sh start               create or upgrade the container, apply the schema, install cron
#   db_service.sh watchdog            restart the container if it is missing, stopped or unhealthy
#   db_service.sh backup              purge old results, archive WAL, take a base backup when due, prune, sync
#   db_service.sh purge               delete result rows of runs older than E2E_DB_DATA_DAYS
#   db_service.sh restore [DIR]       rebuild the database from a backup directory, then start
#   db_service.sh status              show container health and backup state
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_DIR="${E2E_DB_DIR:-${HOME}/e2e-db}"
SERVICE_ENV="${DB_DIR}/service.env"
# Settings saved by `start`; values already set in the environment take precedence.
if [[ -f "${SERVICE_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${SERVICE_ENV}"
fi
CONTAINER="${E2E_DB_CONTAINER:-e2e-results-pg}"
IMAGE="${E2E_DB_IMAGE:-postgres:16}"
PORT="${E2E_DB_PORT:-5432}"
# Loaders run on this host; publish on loopback only so the database is not reachable remotely.
BIND="${E2E_DB_BIND:-127.0.0.1}"
BACKUP_DIR="${E2E_DB_BACKUP_DIR:-${DB_DIR}/backup}"
BACKUP_REMOTE="${E2E_DB_BACKUP_REMOTE:-}"
BASE_DAYS="${E2E_DB_BASE_DAYS:-7}"
RETENTION_DAYS="${E2E_DB_RETENTION_DAYS:-90}"
DATA_DAYS="${E2E_DB_DATA_DAYS:-180}"
INSTALL_CRON="${E2E_DB_CRON:-1}"
DB_USER=e2e
ENV_FILE="${DB_DIR}/postgres.env"
PGDATA_DIR="${DB_DIR}/pgdata"
# Compressed WAL archive; an already archived segment counts as success so retries never block.
ARCHIVE_CMD='test -f /backup/wal/%f.gz || (gzip -c %p > /backup/wal/%f.gz.part && mv /backup/wal/%f.gz.part /backup/wal/%f.gz)'
RESTORE_CMD='gunzip -c /backup/wal/%f.gz > %p'

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# chown -R and rsync --delete run on these paths; refuse anything but a dedicated directory.
check_dir() {
    local name="$1" dir="$2"
    [[ "${dir}" == /* ]] || die "${name} must be an absolute path: '${dir}'"
    dir="$(realpath -m "${dir}")"
    case "${dir}" in
        / | /home | /root | /opt | /var | /tmp | /usr | /etc | /srv | /mnt | "$(realpath -m "${HOME}")")
            die "${name} must be a dedicated directory, not '${dir}'" ;;
    esac
}

check_remote() {
    local path="${BACKUP_REMOTE#*:}"
    if [[ "${BACKUP_REMOTE}" == *:* ]]; then
        [[ "${BACKUP_REMOTE%%:*}" =~ ^[A-Za-z0-9._@-]+$ ]] || die "E2E_DB_BACKUP_REMOTE host is invalid"
        [[ "${path}" == /* && "$(realpath -m "${path}")" != / ]] \
            || die "E2E_DB_BACKUP_REMOTE must be host:/dedicated/absolute/path"
    else
        check_dir E2E_DB_BACKUP_REMOTE "${BACKUP_REMOTE}"
    fi
}

check_dir E2E_DB_DIR "${DB_DIR}"
check_dir E2E_DB_BACKUP_DIR "${BACKUP_DIR}"
if [[ -n "${BACKUP_REMOTE}" ]]; then
    check_remote
fi
# These are used in arithmetic and SQL.
for days in BASE_DAYS RETENTION_DAYS DATA_DAYS; do
    [[ "${!days}" =~ ^[1-9][0-9]*$ ]] || die "E2E_DB_${days} must be a positive integer"
done

container_state() {
    local state
    # docker inspect prints an empty line before failing on a missing container.
    if state="$(docker inspect -f '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' \
            "${CONTAINER}" 2>/dev/null)"; then
        echo "${state% }"
    else
        echo missing
    fi
}

wait_ready() {
    for _ in $(seq 1 "${E2E_DB_WAIT:-600}"); do
        if docker exec "${CONTAINER}" pg_isready -h 127.0.0.1 -U "${DB_USER}" -q 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    die "database is not ready, see: docker logs ${CONTAINER}"
}

# Docker creates missing bind-mount paths as root; make the service and backup directories
# exist and belong to this user before anything is written to them.
ensure_dirs() {
    docker run --rm --entrypoint sh -v "${DB_DIR}:/svc" -v "${BACKUP_DIR}:/backup" "${IMAGE}" -c \
        'mkdir -p /svc/pgdata /backup/base /backup/wal && chown "$1" /svc && chown -R "$1" /svc/pgdata /backup && chmod 700 /svc /backup' \
        sh "$(id -u):$(id -g)"
}

save_settings() {
    {
        for var in E2E_DB_CONTAINER E2E_DB_IMAGE E2E_DB_PORT E2E_DB_BIND E2E_DB_BACKUP_DIR E2E_DB_BACKUP_REMOTE \
                   E2E_DB_BASE_DAYS E2E_DB_RETENTION_DAYS E2E_DB_DATA_DAYS; do
            # Quote removal inside the unquoted ${VAR:-word} turns %q output back into the value.
            # shellcheck disable=SC2016
            printf '%s=${%s:-%q}\n' "${var}" "${var}" "${!var:-}"
        done
    } > "${SERVICE_ENV}"
}

install_cron() {
    local env_prefix log_file block
    env_prefix="E2E_DB_DIR=$(printf %q "${DB_DIR}")"
    log_file="$(printf %q "${DB_DIR}/service.log")"
    block="# BEGIN e2e-db
*/5 * * * * ${env_prefix} $(printf %q "${SCRIPT_DIR}/db_service.sh") watchdog >> ${log_file} 2>&1
17 * * * * ${env_prefix} $(printf %q "${SCRIPT_DIR}/db_service.sh") backup >> ${log_file} 2>&1
# END e2e-db"
    { (crontab -l 2>/dev/null || true) | sed '/^# BEGIN e2e-db$/,/^# END e2e-db$/d'; echo "${block}"; } | crontab -
    log "cron installed: watchdog every 5 minutes, backup hourly"
}

start() {
    E2E_DB_CONTAINER="${CONTAINER}" E2E_DB_IMAGE="${IMAGE}" E2E_DB_PORT="${PORT}" E2E_DB_BIND="${BIND}" \
        E2E_DB_BACKUP_DIR="${BACKUP_DIR}" E2E_DB_BACKUP_REMOTE="${BACKUP_REMOTE}" \
        E2E_DB_BASE_DAYS="${BASE_DAYS}" E2E_DB_RETENTION_DAYS="${RETENTION_DAYS}" E2E_DB_DATA_DAYS="${DATA_DAYS}"
    ensure_dirs
    if [[ ! -f "${ENV_FILE}" ]]; then
        printf 'POSTGRES_USER=%s\nPOSTGRES_PASSWORD=%s\nPOSTGRES_DB=postgres\n' "${DB_USER}" "$(openssl rand -hex 24)" \
            > "${ENV_FILE}"
        log "created ${ENV_FILE} (readable by owner only)"
    fi
    save_settings
    # A node restoring from the backup needs the same credentials.
    cp -p "${ENV_FILE}" "${BACKUP_DIR}/postgres.env"
    if ! systemctl is-enabled docker >/dev/null 2>&1; then
        log "WARNING: docker is not enabled at boot; run: sudo systemctl enable docker"
    fi

    # Postgres runs as the host user so backups stay readable for offsite sync and restore.
    local uid_gid
    uid_gid="$(id -u):$(id -g)"
    if [[ "$(container_state)" != missing ]]; then
        # Recreate so configuration changes apply; data lives in the bind mount.
        docker stop -t 120 "${CONTAINER}" >/dev/null
        docker rm "${CONTAINER}" >/dev/null
    fi
    docker run -d --name "${CONTAINER}" --restart always --user "${uid_gid}" --stop-timeout 120 \
        --env-file "${ENV_FILE}" -p "${BIND}:${PORT}:5432" \
        -v "${PGDATA_DIR}:/var/lib/postgresql/data" -v "${BACKUP_DIR}:/backup" \
        --health-cmd "pg_isready -h 127.0.0.1 -U ${DB_USER}" \
        --health-interval 30s --health-timeout 10s --health-retries 5 \
        "${IMAGE}" postgres -c wal_level=replica -c archive_mode=on -c "archive_command=${ARCHIVE_CMD}" >/dev/null
    wait_ready

    docker exec -i "${CONTAINER}" psql -v ON_ERROR_STOP=1 -q -U "${DB_USER}" -d postgres <<'SQL'
SELECT 'CREATE DATABASE accuracy' WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'accuracy')\gexec
SELECT 'CREATE DATABASE performance' WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'performance')\gexec
SQL
    for db in accuracy performance; do
        cat "${SCRIPT_DIR}/schema/common.sql" "${SCRIPT_DIR}/schema/${db}.sql" |
            docker exec -i -e PGOPTIONS='-c client_min_messages=warning' "${CONTAINER}" \
                psql -v ON_ERROR_STOP=1 -q -U "${DB_USER}" -d "${db}"
        log "schema applied to database: ${db}"
    done
    if [[ "${INSTALL_CRON}" == 1 ]]; then
        install_cron
    fi
    # Every backup chain starts from a base backup.
    backup
}

watchdog() {
    local state
    state="$(container_state)"
    case "${state}" in
        "running healthy" | "running starting") ;;
        "running unhealthy")
            log "container unhealthy, restarting"
            docker restart -t 120 "${CONTAINER}" >/dev/null ;;
        missing)
            log "container missing, starting the service"
            start ;;
        *)
            log "container state '${state}', starting it"
            docker start "${CONTAINER}" >/dev/null ;;
    esac
}

complete_bases() {
    find "${BACKUP_DIR}/base" -mindepth 1 -maxdepth 1 -type d -name '2*' | sort
}

# Results are kept for DATA_DAYS after the run started; the run row goes with its last result.
purge() {
    local db deleted
    for db in accuracy performance; do
        deleted="$(docker exec -i "${CONTAINER}" psql -v ON_ERROR_STOP=1 -Atq -U "${DB_USER}" -d "${db}" <<SQL
WITH old AS (
    DELETE FROM ${db} t USING run r
    WHERE t.run_id = r.id AND r.started_at < now() - make_interval(days => ${DATA_DAYS})
    RETURNING 1
)
SELECT count(*) FROM old;
DELETE FROM run r
WHERE r.started_at < now() - make_interval(days => ${DATA_DAYS})
  AND NOT EXISTS (SELECT 1 FROM ${db} t WHERE t.run_id = r.id);
SQL
)"
        if [[ "${deleted}" != 0 ]]; then
            log "${db}: purged ${deleted} rows of runs older than ${DATA_DAYS} days"
        fi
    done
}

backup() {
    mkdir -p "${BACKUP_DIR}/base" "${BACKUP_DIR}/wal"
    purge
    # Close the current WAL segment and wait until it is archived, so the backup (and the
    # offsite copy) contains everything written so far.
    local segment
    segment="$(docker exec "${CONTAINER}" psql -U "${DB_USER}" -d postgres -Atqc \
        'SELECT pg_walfile_name(pg_switch_wal())')"
    for _ in $(seq 1 120); do
        [[ -f "${BACKUP_DIR}/wal/${segment}.gz" ]] && break
        sleep 1
    done
    [[ -f "${BACKUP_DIR}/wal/${segment}.gz" ]] || log "WARNING: WAL ${segment} not archived yet; check: status"

    local latest
    latest="$(complete_bases | tail -n 1)"
    if [[ -z "${latest}" || -n "$(find "${latest}/.started" -mmin +$((BASE_DAYS * 1440)) 2>/dev/null)" ]]; then
        local ts started tmp
        ts="$(date -u +%Y%m%dT%H%M%SZ)"
        started="$(date +%s)"
        tmp="${BACKUP_DIR}/base/.tmp-${ts}"
        docker exec "${CONTAINER}" pg_basebackup -U "${DB_USER}" -D "/backup/base/.tmp-${ts}" -Ft -z -X stream -c fast
        touch -d "@${started}" "${tmp}/.started"
        mv "${tmp}" "${BACKUP_DIR}/base/${ts}"
        log "base backup ${ts} done"
    fi

    # Drop base backups older than RETENTION_DAYS (never the newest one) and the WAL before the
    # oldest base kept, so no backup data is older than the retention window.
    local bases oldest
    mapfile -t bases < <(complete_bases)
    (( ${#bases[@]} > 0 )) || die "no base backup in ${BACKUP_DIR}/base"
    for old in "${bases[@]:0:${#bases[@]}-1}"; do
        if [[ -n "$(find "${old}/.started" -mmin +$((RETENTION_DAYS * 1440)))" ]]; then
            rm -rf "${old}"
            log "pruned base backup $(basename "${old}") (older than ${RETENTION_DAYS} days)"
        fi
    done
    oldest="$(complete_bases | head -n 1)"
    [[ -n "${oldest}" ]] || die "no base backup in ${BACKUP_DIR}/base"
    find "${BACKUP_DIR}/wal" -type f -name '*.gz' ! -newer "${oldest}/.started" -delete
    find "${BACKUP_DIR}/base" -mindepth 1 -maxdepth 1 -name '.tmp-*' -mmin +1440 -exec rm -rf {} +

    if [[ -n "${BACKUP_REMOTE}" ]]; then
        rsync -a --delete "${BACKUP_DIR}/" "${BACKUP_REMOTE%/}/"
        log "backup synced offsite"
    fi
}

restore() {
    local src="${1:-${BACKUP_DIR}}" base
    src="$(cd "${src}" && pwd)"
    check_dir "restore source" "${src}"
    base="$(BACKUP_DIR="${src}" complete_bases | tail -n 1)"
    [[ -n "${base}" ]] || die "no base backup under ${src}/base"
    [[ "$(container_state)" == missing ]] || die "container ${CONTAINER} exists; remove it first"
    if [[ -d "${PGDATA_DIR}" && -n "$(ls -A "${PGDATA_DIR}")" ]]; then
        die "${PGDATA_DIR} is not empty; move it away first"
    fi

    log "restoring base backup $(basename "${base}") with WAL from ${src}/wal"
    mkdir -p "${PGDATA_DIR}"
    tar -xzf "${base}/base.tar.gz" -C "${PGDATA_DIR}"
    tar -xzf "${base}/pg_wal.tar.gz" -C "${PGDATA_DIR}/pg_wal"
    # Replays every archived WAL segment, then promotes to a normal read-write server.
    echo "restore_command = '${RESTORE_CMD}'" >> "${PGDATA_DIR}/postgresql.auto.conf"
    touch "${PGDATA_DIR}/recovery.signal"
    if [[ ! -f "${ENV_FILE}" ]]; then
        cp -p "${src}/postgres.env" "${ENV_FILE}"
    fi
    BACKUP_DIR="${src}"
    start
}

status() {
    log "container: $(container_state)"
    docker exec "${CONTAINER}" psql -U "${DB_USER}" -d postgres -Atc \
        "SELECT 'archived: ' || coalesce(last_archived_wal, '-') || ' at ' || coalesce(last_archived_time::text, '-')
                || ', failures: ' || failed_count FROM pg_stat_archiver" 2>/dev/null || true
    log "base backups: $(complete_bases | xargs -r -n1 basename | tr '\n' ' ')"
    log "archived WAL: $(find "${BACKUP_DIR}/wal" -name '*.gz' | wc -l) files, $(du -sh "${BACKUP_DIR}" | cut -f1) total"
}

case "${1:-}" in
    start | watchdog | backup | purge | restore | status) ;;
    *) sed -n '2,11p' "$0"; exit 1 ;;
esac

# Serialize commands: cron jobs must not run while start/restore recreates the container.
if [[ "$1" != status ]]; then
    if [[ "$1" == start || "$1" == restore ]]; then
        ensure_dirs
    fi
    mkdir -p "${DB_DIR}"
    exec 9>"${DB_DIR}/.lock"
    case "$1" in
        watchdog | backup | purge) flock -n 9 || { log "another db_service.sh command is running, skip $1"; exit 0; } ;;
        start | restore) flock 9 ;;
    esac
fi

case "$1" in
    start) start ;;
    watchdog) watchdog ;;
    backup) backup ;;
    purge) purge ;;
    restore) restore "${2:-}" ;;
    status) status ;;
esac
