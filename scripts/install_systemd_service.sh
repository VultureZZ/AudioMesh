#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install AudioMesh as a systemd service (Linux).

This writes system services under /etc/systemd/system and (optionally) an env file under /etc/audiomesh/.
Run this script from the repo (it will use the current repo path as WorkingDirectory).

Usage:
  sudo ./scripts/install_systemd_service.sh [--with-web] [--web-mode preview|dev]

Options:
  --with-web            Also install/enable the web UI service (Node/Vite)
  --web-mode <mode>     "preview" (default) runs Vite preview (requires build)
                        "dev" runs Vite dev server (HMR; not recommended for production)

What it installs:
  - /etc/systemd/system/audiomesh-api.service
  - /etc/systemd/system/audiomesh-web.service   (only if --with-web)
  - /etc/audiomesh/audiomesh.env                (created if missing)

After install:
  systemctl status audiomesh-api
  journalctl -u audiomesh-api -f
EOF
}

WITH_WEB=0
WEB_MODE="preview"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-web)
      WITH_WEB=1
      shift
      ;;
    --web-mode)
      WEB_MODE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "This installer must run as root (it writes to /etc/systemd/system)." >&2
  echo "Re-run with sudo:" >&2
  echo "  sudo $0 ${WITH_WEB:+--with-web} --web-mode ${WEB_MODE}" >&2
  exit 1
fi

if [[ "${WEB_MODE}" != "preview" && "${WEB_MODE}" != "dev" ]]; then
  echo "Invalid --web-mode: ${WEB_MODE} (expected: preview|dev)" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-root}"

VENV_PY="${REPO_DIR}/.venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing Python venv at ${VENV_PY}" >&2
  echo "Create it first (from repo root):" >&2
  echo "  python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

run_as_run_user() {
  # Avoid creating root-owned node_modules/dist when this script is run with sudo.
  if [[ "${RUN_USER}" == "root" ]]; then
    bash -lc "$1"
    return
  fi
  sudo -u "${RUN_USER}" -H bash -lc "$1"
}

ENV_DIR="/etc/audiomesh"
ENV_FILE="${ENV_DIR}/audiomesh.env"

mkdir -p "${ENV_DIR}"
if [[ ! -f "${ENV_FILE}" ]]; then
  cat >"${ENV_FILE}" <<'EOF'
# AudioMesh service environment
#
# Server
# HOST=0.0.0.0
# PORT=8000
#
# Auth / rate limiting
# API_KEY=
# RATE_LIMIT_PER_MINUTE=100
#
# Paths (relative paths are resolved from the repo root)
# MODEL_PATH=models/VibeVoice-1.5B
# CUSTOM_VOICES_DIR=custom_voices
# OUTPUT_DIR=outputs
# PODCASTS_DIR=podcasts
# VIBEVOICE_REPO_DIR=VibeVoice
#
# Ollama (optional but recommended)
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2
#
# Realtime TTS (optional)
# REALTIME_VIBEVOICE_REPO_DIR=VibeVoice-Microsoft
# REALTIME_MODEL_ID=microsoft/VibeVoice-Realtime-0.5B
# REALTIME_DEVICE=cuda
# REALTIME_HOST=127.0.0.1
# REALTIME_PORT=6767
# REALTIME_STARTUP_TIMEOUT_SECONDS=60
# REALTIME_SERVER_COMMAND=
#
# Web UI service (only used if audiomesh-web.service is enabled)
# AUDIOMESH_WEB_HOST=0.0.0.0
# AUDIOMESH_WEB_PORT=3001
#
# Vite host allow-list (recommended when accessing the UI via a real hostname)
# Example:
#   VITE_ALLOWED_HOSTS=your-hostname.example.com
# Or, to disable host checks entirely (less secure):
#   VITE_ALLOWED_HOSTS=all
EOF
  chmod 0644 "${ENV_FILE}"
  echo "Created ${ENV_FILE}"
fi

API_UNIT="/etc/systemd/system/audiomesh-api.service"
cat >"${API_UNIT}" <<EOF
[Unit]
Description=AudioMesh API (FastAPI)
After=network.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=${REPO_DIR}/src
EnvironmentFile=-${ENV_FILE}
# Use bash so HOST/PORT from EnvironmentFile can be used.
ExecStart=/usr/bin/env bash -lc '${VENV_PY} -m uvicorn vibevoice.main:app --host "\${HOST:-0.0.0.0}" --port "\${PORT:-8000}"'
Restart=on-failure
RestartSec=3
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

echo "Wrote ${API_UNIT}"

if [[ "${WITH_WEB}" -eq 1 ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm not found. Install Node.js + npm, then re-run with --with-web." >&2
    exit 1
  fi

  if [[ ! -d "${REPO_DIR}/frontend/node_modules" ]]; then
    echo "Installing frontend dependencies..."
    run_as_run_user "cd \"${REPO_DIR}/frontend\" && npm install"
  fi
  if [[ "${WEB_MODE}" == "preview" ]]; then
    echo "Building frontend for preview..."
    run_as_run_user "cd \"${REPO_DIR}/frontend\" && npm run build"
  fi

  WEB_UNIT="/etc/systemd/system/audiomesh-web.service"
  if [[ "${WEB_MODE}" == "preview" ]]; then
    WEB_CMD='npm run preview -- --host "${AUDIOMESH_WEB_HOST:-0.0.0.0}" --port "${AUDIOMESH_WEB_PORT:-3001}"'
  else
    WEB_CMD='npm run dev -- --host "${AUDIOMESH_WEB_HOST:-0.0.0.0}" --port "${AUDIOMESH_WEB_PORT:-3001}"'
  fi

  cat >"${WEB_UNIT}" <<EOF
[Unit]
Description=AudioMesh Web UI (Vite ${WEB_MODE})
After=network.target audiomesh-api.service
Wants=audiomesh-api.service

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}/frontend
EnvironmentFile=-${ENV_FILE}
ExecStart=/usr/bin/env bash -lc '${WEB_CMD}'
Restart=on-failure
RestartSec=3
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

  echo "Wrote ${WEB_UNIT}"
fi

systemctl daemon-reload
systemctl enable --now audiomesh-api.service

if [[ "${WITH_WEB}" -eq 1 ]]; then
  systemctl enable --now audiomesh-web.service
fi

echo ""
echo "Done. Useful commands:"
echo "  systemctl status audiomesh-api"
echo "  journalctl -u audiomesh-api -f"
if [[ "${WITH_WEB}" -eq 1 ]]; then
  echo "  systemctl status audiomesh-web"
  echo "  journalctl -u audiomesh-web -f"
fi
