# .rvc container format

`.rvc` is a standard ZIP archive (no compression — `ZIP_STORED`) that bundles an RVC ONNX model with optional auxiliary files and metadata.

## Structure

```
model.rvc
├── model.onnx         # Required — the RVC synthesizer ONNX model
├── metadata.json      # Auto-generated — container manifest
├── rmvpe.onnx         # Optional — RMVPE pitch extraction model
├── contentvec.onnx    # Optional — ContentVec feature extractor
├── model.index        # Optional — FAISS index for retrieval
├── demo.wav           # Optional — demo audio
└── icon.png           # Optional — model icon
```

Archive paths are arbitrary and specified in `metadata.json` via the `files` mapping. The above shows the default naming convention.

## metadata.json schema

```json
{
    "version": 2,
    "files": {
        "rvc": "model.onnx",
        "pitch": "rmvpe.onnx",
        "embedding": "contentvec.onnx",
        "index": "model.index"
    },
    "sha256": {
        "model.onnx": "e3b0c442...",
        "rmvpe.onnx": "d7a8fbb3..."
    },
    "sample_rate": 48000,
    "phone_dim": 768,
    "name": "MyModel",
    "author": "...",
    "notes": "...",
    "tags": ["tag1", "tag2"]
}
```

### version

Format version integer. Current: `1`.

### files

Maps logical roles to archive paths inside the zip:

| Role | Required | Description |
|------|----------|-------------|
| `rvc` | Yes | RVC ONNX model |
| `pitch` | No | RMVPE pitch extraction model |
| `embedding` | No | ContentVec feature extractor |
| `index` | No | FAISS index |
| `demo` | No | Demo audio |
| `icon` | No | Model icon |

The presence of a role key in `files` indicates the container has that capability.
To check if a container is pitch-capable, check `"pitch" in metadata["files"]`.

### sha256

SHA-256 hex digests of every file in the archive, keyed by archive path.
Used by `onyx verify` to check integrity of all files.

### sample_rate

Output sample rate, auto-detected from the model's `ConvTranspose` node strides:

```python
ratio = 1
for node in model.graph.node:
    if node.op_type == "ConvTranspose":
        for attr in node.attribute:
            if attr.name == "strides":
                ratio *= attr.ints[0]
sample_rate = ratio * 100
```

Common: 400× upsampling → 40 kHz, 480× → 48 kHz.

### phone_dim

Phone embedding dimension, read from the `phone` input tensor (axis 2):
- **768** — v2 model
- **256** — v1 model

## Modeling

```python
from onyx import RVCModel

# From ONNX file
RVCModel.pack("model.rvc", model="model.onnx", index="model.index")

# From ONNX bytes (e.g. after conversion)
with open("model.onnx", "rb") as f:
    RVCModel.pack("model.rvc", model=f.read(), rmvpe="rmvpe.onnx")
```

## Capability detection

The `files` mapping doubles as the capability manifest:

```python
with RVCContainer("model.rvc") as c:
    meta = c.read_metadata()
    files = meta.get("files", {})
    has_pitch = "pitch" in files
    has_embedding = "embedding" in files
    has_index = "index" in files
```

## Backward compatibility

Old format (v1, missing `files` and `version`) is detected automatically:

- `has_*` properties fall back to checking hardcoded filenames
- `sha256` string → treated as hash of `model.onnx` only
- All old `.rvc` files remain readable

## Integrity verification

`onyx verify model.rvc` checks:
1. `metadata.json` exists
2. Every file listed in `sha256` has a matching hash
3. All files listed in `files` mapping exist in the archive
4. `model.onnx` is present (backward compat fallback)
