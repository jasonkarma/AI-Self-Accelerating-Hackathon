#!/bin/sh
# Fail loudly, and before envsubst runs, if the backend origin is missing.
# Without this the container exits on an nginx config error that names a line
# number but not the variable the reader forgot to set.
set -eu

if [ -z "${API_ORIGIN:-}" ]; then
  cat >&2 <<'MSG'
[fatal] API_ORIGIN is not set.

This image proxies the browser's /api/* calls to the FastAPI backend, so it
needs that backend's public origin at start-up:

    API_ORIGIN=https://<your-backend>.up.railway.app

On Railway: open this service → Variables → add API_ORIGIN, then redeploy.
MSG
  exit 1
fi

case "${API_ORIGIN}" in
  http://*|https://*) : ;;
  *)
    echo "[fatal] API_ORIGIN must start with http:// or https:// (got: ${API_ORIGIN})" >&2
    exit 1
    ;;
esac
