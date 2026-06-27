# onyx

RVC voice conversion inference library and CLI. Runs on ONNX Runtime with CPU, CUDA, or DirectML.

## Install

```bash
pip install onyx[cpu]       # ONNX Runtime CPU + FAISS CPU
pip install onyx[cuda]      # ONNX Runtime CUDA + FAISS GPU
pip install onyx[dml]       # ONNX Runtime DirectML + FAISS CPU
```

## Quick start

```bash
# Convert audio
onyx infer -i input.wav -o output.wav --model model.rvc

# Package model into .rvc
onyx pack --model model.onnx -o model.rvc

# Verify container integrity
onyx verify model.rvc
```

## Pipeline

1. Load + resample audio to 16kHz → high-pass filter (48Hz)
2. Extract ContentVec features via ONNX (50Hz → 100Hz)
3. Extract F0 via RMVPE ONNX or autocorrelation
4. Retrieve similar features from FAISS index (optional)
5. Synthesize audio via RVC ONNX model
6. Output WAV at model-native sample rate

Provider priority (auto): `CUDA` → `DirectML` → `CPU`

## Documentation

- [CLI reference](docs/cli.md) — all subcommands and options
- [Library API](docs/library.md) — using onyx from Python
- [Container format](docs/format.md) — .rvc file specification
