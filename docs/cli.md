# CLI reference

## Global options

```bash
onyx {infer,pack,unpack,verify} ...
```

## onyx infer

Run voice conversion on an audio file.

```bash
onyx infer -i input.wav -o output.wav --model model.rvc [options]
```

### Required

| Argument | Description |
|----------|-------------|
| `-i, --input` | Input audio file (any format soundfile supports) |
| `--model` | `.rvc` container or `.onnx` model path |

### Model files

| Argument | Description |
|----------|-------------|
| `--rmvpe` | RMVPE ONNX model (override bundled or CWD fallback) |
| `--cv` | ContentVec ONNX model (override bundled or CWD fallback) |
| `--index` | FAISS index for retrieval-augmented synthesis |

If RMVPE/ContentVec are not specified, they are resolved in order:
1. Explicit `--rmvpe`/`--cv` argument
2. Bundled inside `.rvc` container
3. `./rmvpe.onnx` / `./contentvec.onnx` in current directory

### Output

| Argument | Default | Description |
|----------|---------|-------------|
| `-o, --output` | `tmp/output.wav` | Output WAV path |

### Conversion parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--speaker-id` | `0` | Speaker ID for multi-speaker models |
| `--f0-method` | `rmvpe` | F0 extraction: `rmvpe` (GPU, accurate) or `autocorr` (CPU) |
| `--index-rate` | `0.5` | Blend between retrieved and original features (0–1) |

### Chunking

| Argument | Default | Description |
|----------|---------|-------------|
| `--chunk-sec` | `10` | Chunk size in seconds (`0` = auto-detect from VRAM) |
| `--overlap-sec` | `0.5` | Crossfade overlap between chunks |

### Verification

| Argument | Description |
|----------|-------------|
| `--no-verify` | Skip SHA-256 integrity check on `.rvc` (faster loading) |

### Examples

```bash
# Basic usage with .rvc container
onyx infer -i speech.wav -o output.wav --model MyModel.rvc

# With separate .onnx files
onyx infer -i speech.wav -o output.wav --model model.onnx \
    --rmvpe rmvpe.onnx --cv contentvec.onnx

# With FAISS index retrieval
onyx infer -i speech.wav -o output.wav --model model.rvc \
    --index model.index --index-rate 0.7

# Large file with small chunks (low VRAM)
onyx infer -i long_song.mp3 -o output.wav --model model.rvc \
    --chunk-sec 5 --overlap-sec 0.3

# CPU-only
onyx infer -i speech.wav -o output.wav --model model.rvc \
    CUDAExecutionProvider
```

## onyx pack

Bundle an ONNX model and optional files into a `.rvc` container.

```bash
onyx pack --model model.onnx -o model.rvc [options]
```

| Argument | Description |
|----------|-------------|
| `--model` | Input `.onnx` model (required) |
| `-o, --output` | Output `.rvc` path (required) |
| `--index` | FAISS index to bundle |
| `--rmvpe` | RMVPE model to bundle |
| `--cv` | ContentVec model to bundle |
| `--demo` | Demo audio to bundle |
| `--icon` | Icon image to bundle |
| `--name` | Model display name |
| `--author` | Author name |
| `--notes` | Notes |
| `--tags` | Comma-separated tags |

### Example

```bash
onyx pack --model model.onnx -o model.rvc \
    --index model.index --rmvpe rmvpe.onnx \
    --name "My Voice" --tags "japanese,anime"
```

## onyx unpack

Extract all files from a `.rvc` container to a directory.

```bash
onyx unpack model.rvc -o output_dir/
```

| Argument | Description |
|----------|-------------|
| `model` | `.rvc` file (positional) |
| `-o, --output` | Output directory |

## onyx verify

Check `.rvc` container integrity (SHA-256 + required files).

```bash
onyx verify model.rvc
```

| Argument | Description |
|----------|-------------|
| `model` | `.rvc` file (positional) |

Prints metadata and any integrity errors.

## Model resolution order

For RMVPE, ContentVec, and FAISS index:

```
1. Explicit CLI argument
2. Bundled in .rvc container
3. CWD fallback (rmvpe.onnx / contentvec.onnx)
4. Error (required models only)
```

## Provider override

The CLI does not expose a `--providers` flag. To force a specific provider, set it via Python:

```python
from onyx import RVCModel
model = RVCModel("model.rvc", providers=["CPUExecutionProvider"])
model.convert("input.wav", "output.wav")
```

Provider priority with default (`None`):
1. `CUDAExecutionProvider`
2. `DmlExecutionProvider`
3. `CPUExecutionProvider`
