#!/bin/bash
#
# Restore DeepSwing state from the newest (or a named) offsite backup archive.
# Run on a fresh Pi after cloning the repo and creating the venv, BEFORE
# starting the service. Stops the service if it's running.
#
#   RCLONE_REMOTE=gdrive:DeepSwingBackups ./deploy/restore_from_gdrive.sh
#   RCLONE_REMOTE=gdrive:DeepSwingBackups ./deploy/restore_from_gdrive.sh deepswing_backup_20260702_233000.tar.gz

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

REMOTE="${RCLONE_REMOTE:-}"
if [ -z "$REMOTE" ]; then
    echo "ERROR: RCLONE_REMOTE is not set (e.g. gdrive:DeepSwingBackups)" >&2
    exit 1
fi

ARCHIVE="${1:-}"
if [ -z "$ARCHIVE" ]; then
    ARCHIVE="$(rclone lsf --files-only "$REMOTE" \
        | grep '^deepswing_backup_.*\.tar\.gz$' | sort -r | head -n 1)"
    [ -n "$ARCHIVE" ] || { echo "ERROR: no backups found in $REMOTE" >&2; exit 1; }
fi
echo "Restoring from $ARCHIVE"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
rclone copyto "$REMOTE/$ARCHIVE" "$STAGE/$ARCHIVE" --no-traverse
tar -xzf "$STAGE/$ARCHIVE" -C "$STAGE"

sudo systemctl stop deepswing 2>/dev/null || true

mkdir -p data
[ -f "$STAGE/deepswing.db" ] && cp "$STAGE/deepswing.db" data/deepswing.db && echo "  restored data/deepswing.db"
[ -d "$STAGE/heuristics" ]   && { rm -rf heuristics; cp -r "$STAGE/heuristics" heuristics; echo "  restored heuristics/"; }
[ -d "$STAGE/compiled" ]     && { rm -rf compiled;   cp -r "$STAGE/compiled" compiled;     echo "  restored compiled/"; }
if [ -f "$STAGE/.env" ] && [ ! -f .env ]; then
    cp "$STAGE/.env" .env && echo "  restored .env"
elif [ -f "$STAGE/.env" ]; then
    echo "  .env already present — left untouched (backup copy in archive)"
fi

echo "Restore complete. Start the service:  sudo systemctl start deepswing"
