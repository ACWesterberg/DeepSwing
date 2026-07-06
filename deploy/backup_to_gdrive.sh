#!/bin/bash
#
# Offsite backup of DeepSwing's durable state to a cloud remote via rclone.
# Runs independently of the app (systemd timer) so backups survive an app crash
# or an SD-card failure — the local data/backups/ snapshots live on the same
# card and do NOT protect against card death; this does.
#
# One-time setup on the Pi (as the service user):
#   sudo apt install -y rclone sqlite3
#   rclone config                      # create a Google Drive remote named "gdrive"
#   rclone mkdir gdrive:DeepSwingBackups
# Then enable the timer (see systemd/deepswing-backup.timer).
#
# Config via environment (or /etc/default/deepswing-backup):
#   RCLONE_REMOTE   destination, e.g. gdrive:DeepSwingBackups   (required)
#   BACKUP_KEEP     how many archives to retain remotely         (default 14)
#   BACKUP_INCLUDE_ENV  include .env (API keys) in the archive   (default true)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

REMOTE="${RCLONE_REMOTE:-}"
KEEP="${BACKUP_KEEP:-14}"
INCLUDE_ENV="${BACKUP_INCLUDE_ENV:-true}"

if [ -z "$REMOTE" ]; then
    echo "ERROR: RCLONE_REMOTE is not set (e.g. gdrive:DeepSwingBackups)" >&2
    exit 1
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# 1. Consistent SQLite snapshot (online backup API — safe during a live write).
DB="data/deepswing.db"
if [ -f "$DB" ]; then
    if command -v sqlite3 >/dev/null 2>&1; then
        sqlite3 "$DB" ".backup '$STAGE/deepswing.db'"
    else
        # Fallback: plain copy. Slight torn-write risk, still better than nothing.
        cp "$DB" "$STAGE/deepswing.db"
    fi
fi

# 2. Learned state: heuristics + compiled MIPRO programs (small JSON trees).
[ -d heuristics ] && cp -r heuristics "$STAGE/heuristics"
[ -d compiled ]   && cp -r compiled   "$STAGE/compiled"

# 3. Config/secrets — optional. Lets a rebuild skip re-entering API keys.
if [ "$INCLUDE_ENV" = "true" ] && [ -f .env ]; then
    cp .env "$STAGE/.env"
fi

# 4. Single compressed archive, then push to the remote.
ARCHIVE="deepswing_backup_${STAMP}.tar.gz"
tar -czf "$STAGE/$ARCHIVE" -C "$STAGE" \
    $([ -f "$STAGE/deepswing.db" ] && echo deepswing.db) \
    $([ -d "$STAGE/heuristics" ]   && echo heuristics) \
    $([ -d "$STAGE/compiled" ]     && echo compiled) \
    $([ -f "$STAGE/.env" ]         && echo .env)

echo "Uploading $ARCHIVE to $REMOTE ..."
rclone copyto "$STAGE/$ARCHIVE" "$REMOTE/$ARCHIVE" --no-traverse

# 5. Rotate: keep the newest $KEEP archives (names sort chronologically).
mapfile -t OLD < <(rclone lsf --files-only "$REMOTE" 2>/dev/null \
    | grep '^deepswing_backup_.*\.tar\.gz$' | sort -r | tail -n "+$((KEEP + 1))")
for f in "${OLD[@]:-}"; do
    [ -n "$f" ] || continue
    echo "Pruning old backup $f"
    rclone deletefile "$REMOTE/$f" || true
done

echo "Backup complete: $ARCHIVE ($(du -h "$STAGE/$ARCHIVE" | cut -f1))"
