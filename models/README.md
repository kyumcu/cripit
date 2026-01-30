# Model Files Directory

This directory stores GGML model files for whisper.cpp.

## Download Models

### Option 1: Using whisper.cpp's download script

```bash
cd ../..  # Go to project root
# Download Whisper Large V3 Turbo (RECOMMENDED - 809M)
./models/download-ggml-model.sh large-v3-turbo

# Or download other models:
./models/download-ggml-model.sh large-v3      # 1.55B - Higher quality
./models/download-ggml-model.sh small         # 466M - Faster
./models/download-ggml-model.sh base          # 142M - Testing
```

### Option 2: Manual Download

Download from: https://huggingface.co/ggerganov/whisper.cpp

Place `.bin` files in this directory.

## Recommended Model

**Whisper Large V3 Turbo** (809M parameters)
- File: `ggml-large-v3-turbo.bin`
- Speed: 6x faster than Large V3
- Quality: ~7.75% WER
- Languages: 99+
- VRAM: ~6GB

## Model Files

| Model | Size | File | Speed | Quality |
|-------|------|------|-------|---------|
| Large V3 Turbo | ~1.5GB | `ggml-large-v3-turbo.bin` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Large V3 | ~3GB | `ggml-large-v3.bin` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Small | ~500MB | `ggml-small.bin` | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Base | ~150MB | `ggml-base.bin` | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## Note

Model files are NOT tracked in git due to large size (~1-3GB each).
The app will auto-download models if not present.
