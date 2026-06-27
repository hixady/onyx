# onyx

RVC voice conversion inference library and CLI. Runs on ONNX Runtime with CPU, CUDA, or DirectML.

## Install

```bash
pip install onyx[cpu]       # ONNX Runtime CPU + FAISS CPU
pip install onyx[cuda]      # ONNX Runtime CUDA + FAISS GPU
pip install onyx[dml]       # ONNX Runtime DirectML + FAISS CPU
```

## Usage

### Inference

```bash
onyx infer -i input.wav -o output.wav --model model.rvc
```

| Option | Description |
|--------|-------------|
| `-i, --input` | Input audio file |
| `-o, --output` | Output WAV path |
| `--model` | `.rvc` container or `.onnx` model |
| `--rmvpe` / `--cv` | Override bundled RMVPE/ContentVec models |
| `--index` | FAISS index for retrieval-augmented synthesis |
| `--chunk-sec` | Chunk length in seconds (0 = auto) |
| `--speaker-id` | Speaker ID (default: 0) |
| `--f0-method` | `rmvpe` or `autocorr` |
| `--index-rate` | Retrieval blend weight (0-1) |
| `--no-verify` | Skip sha256 check on `.rvc` |

### Container commands

```bash
# Pack .onnx → .rvc
onyx pack --model model.onnx -o model.rvc [--index model.index]

# Verify integrity
onyx verify model.rvc

# Extract contents
onyx unpack -i model.rvc -o output_dir/
```

## Pipeline

1. Load + resample audio to 16kHz → high-pass filter (48Hz, Butterworth 5)
2. Extract ContentVec features via ONNX (50Hz → 100Hz)
3. Extract F0 via RMVPE ONNX or autocorrelation
4. Retrieve similar features from FAISS index (optional)
5. Synthesize audio via RVC ONNX model
6. Output WAV at model-native sample rate (40kHz / 48kHz)

## `.rvc` format

Standard zip (DEFLATE) containing:

| Entry | Required | Description |
|-------|----------|-------------|
| `model.onnx` | Yes | RVC ONNX model |
| `metadata.json` | Auto | sha256 + sample_rate + phone_dim |
| `model.index` | No | FAISS index |
| `rmvpe.onnx` | No | RMVPE model |
| `contentvec.onnx` | No | ContentVec model |

## Provider selection

Auto-selects: `CUDAExecutionProvider` → `DmlExecutionProvider` → `CPUExecutionProvider`
