#!/bin/bash
# =============================================================================
# Database Backup Script for DevOps AI Assistant
# =============================================================================
# Backs up PostgreSQL, Qdrant, and Redis data with rotation and verification.
#
# Usage:
#   ./scripts/backup_databases.sh              # Full backup
#   ./scripts/backup_databases.sh --postgres   # PostgreSQL only
#   ./scripts/backup_databases.sh --qdrant     # Qdrant only
#   ./scripts/backup_databases.sh --redis      # Redis only
#   ./scripts/backup_databases.sh --verify     # Verify latest backup
#
# Environment variables (optional):
#   BACKUP_DIR          - Base backup directory (default: /data/backups)
#   BACKUP_RETENTION    - Days to keep backups (default: 7)
#   POSTGRES_CONTAINER  - PostgreSQL container name (default: postgres)
#   QDRANT_HOST         - Qdrant host (default: localhost)
#   QDRANT_PORT         - Qdrant port (default: 6333)
#   REDIS_CONTAINER     - Redis container name (default: redis)
# =============================================================================

set -euo pipefail

# Configuration with defaults
BACKUP_DIR="${BACKUP_DIR:-/data/backups}"
BACKUP_RETENTION="${BACKUP_RETENTION:-7}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-postgres}"
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
REDIS_CONTAINER="${REDIS_CONTAINER:-redis}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-devops_docs}"

# Timestamp for this backup run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="${BACKUP_DIR}/${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create backup directory
create_backup_dir() {
    mkdir -p "${BACKUP_PATH}"
    log_info "Created backup directory: ${BACKUP_PATH}"
}

# Backup PostgreSQL
backup_postgres() {
    log_info "Backing up PostgreSQL..."

    if ! docker ps --format '{{.Names}}' | grep -q "^${POSTGRES_CONTAINER}$"; then
        log_error "PostgreSQL container '${POSTGRES_CONTAINER}' is not running"
        return 1
    fi

    # Get database credentials from container environment
    local pg_user=$(docker exec ${POSTGRES_CONTAINER} printenv POSTGRES_USER 2>/dev/null || echo "raguser")
    local pg_db=$(docker exec ${POSTGRES_CONTAINER} printenv POSTGRES_DB 2>/dev/null || echo "ragdb")

    # Create compressed dump with custom format (supports parallel restore)
    if docker exec ${POSTGRES_CONTAINER} pg_dump -U "${pg_user}" -Fc "${pg_db}" > "${BACKUP_PATH}/postgres.dump"; then
        local size=$(du -h "${BACKUP_PATH}/postgres.dump" | cut -f1)
        log_info "PostgreSQL backup complete: ${BACKUP_PATH}/postgres.dump (${size})"

        # Create plain SQL backup as well for portability
        docker exec ${POSTGRES_CONTAINER} pg_dump -U "${pg_user}" "${pg_db}" | gzip > "${BACKUP_PATH}/postgres.sql.gz"
        log_info "PostgreSQL SQL backup: ${BACKUP_PATH}/postgres.sql.gz"
        return 0
    else
        log_error "PostgreSQL backup failed"
        return 1
    fi
}

# Backup Qdrant vector database
backup_qdrant() {
    log_info "Backing up Qdrant collection '${QDRANT_COLLECTION}'..."

    # Create snapshot via Qdrant API
    local snapshot_response=$(curl -s -X POST "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${QDRANT_COLLECTION}/snapshots" 2>/dev/null)

    if echo "${snapshot_response}" | grep -q '"status":"ok"\|"result"'; then
        local snapshot_name=$(echo "${snapshot_response}" | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)

        if [ -n "${snapshot_name}" ]; then
            log_info "Qdrant snapshot created: ${snapshot_name}"

            # Download the snapshot
            if curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${QDRANT_COLLECTION}/snapshots/${snapshot_name}" \
                -o "${BACKUP_PATH}/qdrant_${QDRANT_COLLECTION}.snapshot"; then
                local size=$(du -h "${BACKUP_PATH}/qdrant_${QDRANT_COLLECTION}.snapshot" | cut -f1)
                log_info "Qdrant backup complete: ${BACKUP_PATH}/qdrant_${QDRANT_COLLECTION}.snapshot (${size})"

                # Store snapshot metadata
                echo "${snapshot_response}" > "${BACKUP_PATH}/qdrant_snapshot_info.json"
                return 0
            fi
        fi
    fi

    log_error "Qdrant backup failed. Response: ${snapshot_response}"
    return 1
}

# Backup Redis
backup_redis() {
    log_info "Backing up Redis..."

    if ! docker ps --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER}$"; then
        log_error "Redis container '${REDIS_CONTAINER}' is not running"
        return 1
    fi

    # Trigger background save
    docker exec ${REDIS_CONTAINER} redis-cli BGSAVE > /dev/null 2>&1

    # Wait for save to complete (max 30 seconds)
    local count=0
    while [ $count -lt 30 ]; do
        local status=$(docker exec ${REDIS_CONTAINER} redis-cli LASTSAVE 2>/dev/null)
        sleep 1
        local new_status=$(docker exec ${REDIS_CONTAINER} redis-cli LASTSAVE 2>/dev/null)
        if [ "${status}" != "${new_status}" ] || [ $count -eq 0 ]; then
            break
        fi
        count=$((count + 1))
    done

    # Copy RDB file from container
    if docker cp "${REDIS_CONTAINER}:/data/dump.rdb" "${BACKUP_PATH}/redis.rdb" 2>/dev/null; then
        local size=$(du -h "${BACKUP_PATH}/redis.rdb" | cut -f1)
        log_info "Redis backup complete: ${BACKUP_PATH}/redis.rdb (${size})"

        # Also backup AOF if it exists
        if docker cp "${REDIS_CONTAINER}:/data/appendonly.aof" "${BACKUP_PATH}/redis_appendonly.aof" 2>/dev/null; then
            log_info "Redis AOF backup: ${BACKUP_PATH}/redis_appendonly.aof"
        fi
        return 0
    else
        log_error "Redis backup failed"
        return 1
    fi
}

# Verify backup integrity
verify_backup() {
    local backup_to_verify="${1:-$(ls -td ${BACKUP_DIR}/*/ 2>/dev/null | head -1)}"

    if [ -z "${backup_to_verify}" ] || [ ! -d "${backup_to_verify}" ]; then
        log_error "No backup found to verify"
        return 1
    fi

    log_info "Verifying backup: ${backup_to_verify}"
    local errors=0

    # Verify PostgreSQL dump
    if [ -f "${backup_to_verify}/postgres.dump" ]; then
        if pg_restore --list "${backup_to_verify}/postgres.dump" > /dev/null 2>&1; then
            log_info "PostgreSQL dump: VALID"
        else
            log_error "PostgreSQL dump: INVALID"
            errors=$((errors + 1))
        fi
    fi

    # Verify Qdrant snapshot (check file size > 0)
    if [ -f "${backup_to_verify}/qdrant_${QDRANT_COLLECTION}.snapshot" ]; then
        local size=$(stat -f%z "${backup_to_verify}/qdrant_${QDRANT_COLLECTION}.snapshot" 2>/dev/null || \
                     stat -c%s "${backup_to_verify}/qdrant_${QDRANT_COLLECTION}.snapshot" 2>/dev/null || echo "0")
        if [ "${size}" -gt 1000 ]; then
            log_info "Qdrant snapshot: VALID (${size} bytes)"
        else
            log_error "Qdrant snapshot: INVALID (too small)"
            errors=$((errors + 1))
        fi
    fi

    # Verify Redis RDB
    if [ -f "${backup_to_verify}/redis.rdb" ]; then
        # Check for Redis RDB magic number (REDIS)
        if head -c 5 "${backup_to_verify}/redis.rdb" | grep -q "REDIS"; then
            log_info "Redis RDB: VALID"
        else
            log_error "Redis RDB: INVALID (bad magic number)"
            errors=$((errors + 1))
        fi
    fi

    if [ $errors -eq 0 ]; then
        log_info "Backup verification: ALL PASSED"
        return 0
    else
        log_error "Backup verification: ${errors} FAILURES"
        return 1
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than ${BACKUP_RETENTION} days..."

    local count=$(find "${BACKUP_DIR}" -maxdepth 1 -type d -mtime +${BACKUP_RETENTION} 2>/dev/null | wc -l)

    if [ "${count}" -gt 0 ]; then
        find "${BACKUP_DIR}" -maxdepth 1 -type d -mtime +${BACKUP_RETENTION} -exec rm -rf {} \;
        log_info "Removed ${count} old backup(s)"
    else
        log_info "No old backups to clean up"
    fi
}

# Create backup manifest
create_manifest() {
    cat > "${BACKUP_PATH}/manifest.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "created_at": "$(date -Iseconds)",
    "components": {
        "postgres": $([ -f "${BACKUP_PATH}/postgres.dump" ] && echo "true" || echo "false"),
        "qdrant": $([ -f "${BACKUP_PATH}/qdrant_${QDRANT_COLLECTION}.snapshot" ] && echo "true" || echo "false"),
        "redis": $([ -f "${BACKUP_PATH}/redis.rdb" ] && echo "true" || echo "false")
    },
    "sizes": {
        "postgres": "$(du -h "${BACKUP_PATH}/postgres.dump" 2>/dev/null | cut -f1 || echo "N/A")",
        "qdrant": "$(du -h "${BACKUP_PATH}/qdrant_${QDRANT_COLLECTION}.snapshot" 2>/dev/null | cut -f1 || echo "N/A")",
        "redis": "$(du -h "${BACKUP_PATH}/redis.rdb" 2>/dev/null | cut -f1 || echo "N/A")"
    },
    "retention_days": ${BACKUP_RETENTION}
}
EOF
    log_info "Created backup manifest: ${BACKUP_PATH}/manifest.json"
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --postgres    Backup PostgreSQL only"
    echo "  --qdrant      Backup Qdrant only"
    echo "  --redis       Backup Redis only"
    echo "  --verify      Verify the latest backup"
    echo "  --cleanup     Only run cleanup of old backups"
    echo "  --help        Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  BACKUP_DIR          Base backup directory (default: /data/backups)"
    echo "  BACKUP_RETENTION    Days to keep backups (default: 7)"
}

# Main execution
main() {
    local do_postgres=true
    local do_qdrant=true
    local do_redis=true
    local do_verify=false
    local do_cleanup_only=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --postgres)
                do_qdrant=false
                do_redis=false
                shift
                ;;
            --qdrant)
                do_postgres=false
                do_redis=false
                shift
                ;;
            --redis)
                do_postgres=false
                do_qdrant=false
                shift
                ;;
            --verify)
                do_verify=true
                do_postgres=false
                do_qdrant=false
                do_redis=false
                shift
                ;;
            --cleanup)
                do_cleanup_only=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Verify only mode
    if [ "${do_verify}" = true ]; then
        verify_backup
        exit $?
    fi

    # Cleanup only mode
    if [ "${do_cleanup_only}" = true ]; then
        cleanup_old_backups
        exit 0
    fi

    log_info "Starting backup at $(date)"
    log_info "Backup directory: ${BACKUP_PATH}"

    create_backup_dir

    local success=true

    # Run backups
    if [ "${do_postgres}" = true ]; then
        backup_postgres || success=false
    fi

    if [ "${do_qdrant}" = true ]; then
        backup_qdrant || success=false
    fi

    if [ "${do_redis}" = true ]; then
        backup_redis || success=false
    fi

    # Create manifest and cleanup
    create_manifest
    cleanup_old_backups

    # Final status
    if [ "${success}" = true ]; then
        log_info "Backup completed successfully!"
        log_info "Location: ${BACKUP_PATH}"

        # Verify the backup we just created
        verify_backup "${BACKUP_PATH}"
    else
        log_warn "Backup completed with some errors"
        exit 1
    fi
}

main "$@"
