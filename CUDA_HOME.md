# CUDA Home

- `CUDA_HOME`: `/usr/lib/cuda`
- `nvcc`: `/usr/bin/nvcc` (CUDA 12.0)
- Headers: `/usr/include` (e.g. `/usr/include/cuda_runtime.h`)
- Libraries: `/usr/lib/x86_64-linux-gnu` (e.g. `libcudart.so*`)

If you need `CUDA_HOME` set for builds:

```sh
export CUDA_HOME=/usr/lib/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
```

Note:
- Some environments (including conda) may set `CUDA_HOME=/usr/local/cuda` even when that path does not exist.
- CripIt's CUDA detection ignores a non-existent `CUDA_HOME` and falls back to common install paths.
