# CUDA Setup (GPU Acceleration)

CripIt can use GPU acceleration via CUDA when `pywhispercpp` is compiled with CUDA support.

## Quick Start

1. Confirm NVIDIA driver + GPU are working:

```bash
nvidia-smi
```

2. Confirm CUDA toolkit is installed:

```bash
nvcc --version
```

3. Activate the app environment:

```bash
conda activate py_cripit
```

4. Build/install `pywhispercpp` with CUDA enabled:

```bash
bash build_cuda.sh
```

5. Verify from Python:

```bash
python -c "from utils.cuda_utils import CUDAManager; CUDAManager().validate_cuda_setup(verbose=True)"
```

6. Run the app:

```bash
python main.py
```

Tip: use the helper script to run with logging and avoid unnecessary rebuilds:

```bash
./run_with_cuda.sh
```

## App Settings

CripIt stores CUDA preferences in `app_config.json` under `model`:

- `use_cuda`: try GPU when possible
- `cuda_device`: GPU index (0 for first GPU)
- `cuda_fallback_to_cpu`: auto-fallback if GPU init fails
- `cuda_warn_on_fallback`: warn when falling back

You can change these in the GUI: `Settings...` -> `GPU Acceleration (CUDA)`.

## Environment Variables

If CUDA is installed but not detected, set `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Then re-run `bash build_cuda.sh`.

## Troubleshooting

### CUDA detected but CripIt still uses CPU

- Make sure you built CUDA-enabled `pywhispercpp`:
  - Run `bash build_cuda.sh`
  - Then run the validation command in the Quick Start
- Restart your shell / Python environment after the build.

### Out-of-memory (GPU)

- Use a smaller model (e.g. `small` or `medium`)
- Close other GPU-heavy applications
- Keep `cuda_fallback_to_cpu` enabled so transcription still works

### No `nvidia-smi`

- Install NVIDIA drivers for your GPU.

## Notes

- GPU acceleration requires a compatible NVIDIA GPU + driver.
- CUDA setup differs by OS; use your distro/vendor instructions for installing the CUDA toolkit.
