#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/manager/opencode/cripit"
CONDA_SH="/home/manager/miniforge3/etc/profile.d/conda.sh"
ENV_NAME="py_cripit"
CUDA_HOME_PATH="/usr/lib/cuda"

cd "$REPO_DIR"

# Stop any running CripIt instance (ignore if none)
pkill -f "python -u main.py" 2>/dev/null || true
pkill -f "python main.py" 2>/dev/null || true

# Activate conda env
if [ ! -f "$CONDA_SH" ]; then
  echo "Missing conda activation script: $CONDA_SH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

export CUDA_HOME="$CUDA_HOME_PATH"

cuda_ready() {
  python - <<'PY'
from utils.cuda_utils import CUDAManager
ok, msg = CUDAManager().validate_cuda_setup(verbose=False)
print(msg)
raise SystemExit(0 if ok else 1)
PY
}

fix_pywhispercpp_libcuda() {
  # Some pywhispercpp wheels bundle a copy of libcuda with a hashed filename.
  # That copy can fail at runtime because libcuda expects other driver libs
  # to be available relative to its install location.
  #
  # Workaround: replace the bundled libcuda with a symlink to the system libcuda.
  local sys_cuda="/lib/x86_64-linux-gnu/libcuda.so.1"
  if [ ! -e "$sys_cuda" ]; then
    echo "System libcuda not found at: $sys_cuda" >&2
    return 1
  fi

  local libs_dir
  libs_dir="$(python - <<'PY'
from pathlib import Path
import pywhispercpp
print(str(Path(pywhispercpp.__file__).resolve().parent.parent / 'pywhispercpp.libs'))
PY
  )"

  if [ ! -d "$libs_dir" ]; then
    echo "pywhispercpp libs dir not found: $libs_dir" >&2
    return 1
  fi

  shopt -s nullglob
  local changed=0
  local f
  for f in "$libs_dir"/libcuda-*.so.*; do
    # ignore backups
    case "$f" in
      *.bundled) continue ;;
    esac

    if [ -L "$f" ]; then
      continue
    fi

    mv "$f" "$f.bundled"
    ln -s "$sys_cuda" "$f"
    changed=1
  done
  shopt -u nullglob

  if [ "$changed" -eq 1 ]; then
    echo "Patched pywhispercpp to use system libcuda" 
  fi
}

# Build CUDA-enabled pywhispercpp only if needed
if cuda_ready >/dev/null 2>&1; then
  echo "CUDA + pywhispercpp CUDA support already OK; skipping rebuild"
else
  echo "CUDA not ready in current env; rebuilding pywhispercpp with CUDA"
  bash ./build_cuda.sh --force --verbose
fi

# Ensure runtime links against system libcuda
fix_pywhispercpp_libcuda || true

# Verify CUDA + pywhispercpp CUDA support (prints report)
python -c "from utils.cuda_utils import CUDAManager; CUDAManager().validate_cuda_setup(verbose=True)"

# Run app
mkdir -p logs
LOG_FILE="logs/run-$(date +%Y%m%d-%H%M%S).log"
echo "Starting CripIt; logging to: $LOG_FILE"
python -u main.py >"$LOG_FILE" 2>&1
