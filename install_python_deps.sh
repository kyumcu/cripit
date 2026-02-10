#!/usr/bin/env bash
set -euo pipefail

# Bootstraps pip (if missing) and installs Python deps.
#
# Usage:
#   ./install_python_deps.sh                 # creates ./py_cripit venv + installs requirements.txt
#   ./install_python_deps.sh --cli           # creates ./py_cripit venv + installs requirements-cli.txt if present
#   ./install_python_deps.sh --no-venv       # installs to user/site (or system if root)
#
# Notes:
# - When using --no-venv, installs with --user when not running as root.
# - This script only installs Python packages. Some packages (sounddevice, PyQt6,
#   pywhispercpp builds) may require OS-level libraries and build tools.

REQ_MODE="gui"
USE_VENV="1"
VENV_DIR="py_cripit"

for arg in "$@"; do
  case "$arg" in
    --cli)
      REQ_MODE="cli"
      ;;
    --gui)
      REQ_MODE="gui"
      ;;
    --no-venv)
      USE_VENV="0"
      ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

if command -v python3.13 >/dev/null 2>&1; then
  PY_SYS="python3.13"
elif command -v python3 >/dev/null 2>&1; then
  PY_SYS="python3"
else
  echo "python3 not found" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

bootstrap_pip() {
  local py="$1"

  if "$py" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  echo "pip not found; attempting bootstrap via ensurepip..." >&2
  if "$py" -m ensurepip --upgrade >/dev/null 2>&1; then
    return 0
  fi

  echo "ensurepip unavailable; attempting bootstrap via get-pip.py..." >&2
  TMPDIR="$(mktemp -d)"
  trap 'rm -rf "$TMPDIR"' EXIT

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$TMPDIR/get-pip.py" https://bootstrap.pypa.io/get-pip.py
  else
    echo "Neither ensurepip nor curl/wget available to install pip." >&2
    echo "Install pip using your OS package manager (example on Debian/Ubuntu):" >&2
    echo "  sudo apt-get update && sudo apt-get install -y python3-pip" >&2
    exit 1
  fi

  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    "$py" "$TMPDIR/get-pip.py"
  else
    "$py" "$TMPDIR/get-pip.py" --user
  fi
}

pick_requirements_file() {
  if [[ "$REQ_MODE" == "cli" ]]; then
    if [[ -f "requirements-cli.txt" ]]; then
      echo "requirements-cli.txt"
      return 0
    fi
    echo "requirements.txt"
    return 0
  fi

  echo "requirements.txt"
}

if [[ "$USE_VENV" == "1" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtualenv: $VENV_DIR" >&2
    if ! "$PY_SYS" -m venv "$VENV_DIR"; then
      echo "Failed to create venv. On Debian/Ubuntu you may need:" >&2
      echo "  sudo apt-get update && sudo apt-get install -y python3-venv" >&2
      exit 1
    fi
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PY="python"
  bootstrap_pip "$PY"
  PIP_USER_ARGS=()
else
  PY="$PY_SYS"
  bootstrap_pip "$PY"
  PIP_USER_ARGS=()
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    PIP_USER_ARGS=(--user)
  fi
fi

echo "Upgrading pip tooling..." >&2
"$PY" -m pip install "${PIP_USER_ARGS[@]}" --upgrade pip setuptools wheel

REQ_FILE="$(pick_requirements_file)"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Requirements file not found: $REQ_FILE" >&2
  exit 1
fi

echo "Installing Python packages from: $REQ_FILE" >&2
"$PY" -m pip install "${PIP_USER_ARGS[@]}" -r "$REQ_FILE"

echo "" >&2
echo "Done." >&2
if [[ "$USE_VENV" == "1" ]]; then
  echo "Activate the venv:" >&2
  echo "  source $VENV_DIR/bin/activate" >&2
fi
echo "Run (GUI):" >&2
echo "  python main.py" >&2
echo "Run (TUI, if present):" >&2
echo "  python terminal_app.py" >&2
