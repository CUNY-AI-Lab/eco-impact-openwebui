#!/bin/bash
# install-registry.example.sh — EXAMPLE, adapt to your infrastructure.
#
# The Eco Impact Estimator filter reads its model registry from a JSON file on
# disk inside the Open WebUI container, default path:
#     /app/backend/data/eco_models.json
# (/app/backend/data is Open WebUI's persistent data dir, so the file survives
# restarts.) The filter re-reads the file whenever its mtime changes — no
# restart needed — and falls back to a small embedded registry if it is absent.
#
# "Installing the registry" therefore just means: get eco_models.json (built by
# sync-eco-models.py) onto that path in your deployment. HOW you copy a file in
# depends entirely on where/how you run Open WebUI. A few common shapes:
#
# ── Docker / docker compose ────────────────────────────────────────────────
#   docker cp eco_models.json <container>:/app/backend/data/eco_models.json
#
# ── Kubernetes ─────────────────────────────────────────────────────────────
#   kubectl cp eco_models.json <namespace>/<pod>:/app/backend/data/eco_models.json
#
# ── A mounted volume / bind mount ──────────────────────────────────────────
#   cp eco_models.json /path/to/your/openwebui/data/eco_models.json
#
# ── AWS ECS on Fargate (no shell access; file must be written from inside a
#    running task via ECS Exec, in base64 chunks because one exec command
#    cannot carry the whole file) ─────────────────────────────────────────
#   The original authors deploy this way; a reference implementation of the
#   chunked-ECS-Exec approach lives in the project this was extracted from.
#   It is intentionally NOT included here because it is specific to one
#   cluster/service/task layout. If you run on ECS Fargate and want that
#   script, open an issue.
#
# Whatever the transport, the only requirement is: the JSON ends up readable at
# the filter's `registry_path` valve (default above), and its mtime changes so
# the filter reloads it.

set -euo pipefail

FILE="${1:-eco_models.json}"
DEST="${DEST:-/app/backend/data/eco_models.json}"

[ -f "${FILE}" ] || { echo "No such file: ${FILE}"; exit 1; }
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "${FILE}" \
  || { echo "Invalid JSON: ${FILE}"; exit 1; }

echo "Validated ${FILE} ($(wc -c < "${FILE}") bytes)."
echo "Now copy it to ${DEST} inside your Open WebUI container using whichever"
echo "of the methods above fits your deployment (see comments in this file)."
