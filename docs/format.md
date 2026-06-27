# .rvc container format

`.rvc` is a standard ZIP archive (DEFLATE compression) that bundles an RVC ONNX model with optional auxiliary files and metadata.

## Structure

```
model.rvc
├── model.onnx        # Required — the RVC synthesizer ONNX model
├── metadata.json     # Auto-generated — container metadata (see below)
├── model.index       # Optional — FAISS index for retrieval-augmented synthesis
├── rmvpe.onnx        # Optional — RMVPE pitch extraction model
├── contentvec.onnx   # Optional — ContentVec feature extractor (v2, 768-dim)
├── demo.wav          # Optional — demo audio sample
└── icon.png           # Optional — model icon
```

## metadata.json schema

```json
{
  "sha256":       "abc...",       // SHA-256 hex digest of model.onnx
  "sample_rate":  48000,          // Output sample rate (auto-detected)
  "phone_dim":    768,            // Phone embedding dimension (auto-detected)
  "name":         "MyModel",      // Optional — model display name
  "author":       "...",          // Optional
  "notes":        "...",          // Optional
  "tags":         ["tag1", "tag2"] // Optional
}
```

### sha256

Computed from `model.onnx` at pack time. Used by `onyx verify` and `--no-verify` to skip the check during inference.

### sample_rate auto-detection

Detected from the model's `ConvTranspose` node strides:

```python
ratio = 1
for node in model.graph.node:
    if node.op_type == "ConvTranspose":
        for attr in node.attribute:
            if attr.name == "strides":
                ratio *= attr.ints[0]
sample_rate = ratio * 100
```

Common values:
- 400× upsampling → **40 kHz**
- 480× upsampling → **48 kHz**

### phone_dim auto-detection

Read from the `phone` input tensor shape (dimension 2):

- **768** — v2 model (768-dim phone embeddings)
- **256** — v1 model (256-dim phone embeddings)

## Creating .rvc files

```python
from onyx import RVCModel

# From ONNX file
RVCModel.pack("model.rvc", model="model.onnx", index="model.index")

# From ONNX bytes (e.g. after conversion)
with open("model.onnx", "rb") as f:
    RVCModel.pack("model.rvc", model=f.read(), index="model.index")

# With extra metadata
RVCModel.pack("model.rvc", model="model.onnx",
              name="My Voice", author="User", tags="cool,test")
```

## Model bundling priority

When loading a `.rvc` container, the model resolution order is:

1. **Explicit CLI argument** (`--rmvpe`, `--cv`, `--index`) — highest priority
2. **Bundled inside .rvc** — used if the explicit argument is not provided
3. **CWD fallback** — `./rmvpe.onnx`, `./contentvec.onnx` checked if not found above
4. **Error** — if RMVPE or ContentVec are still missing

## Version compatibility

| Field | v1 (256-dim) | v2 (768-dim) |
|-------|-------------|-------------|
| Phone dim | 256 | 768 |
| ContentVec | contentvec_256l9 | contentvec_768l12 |
| Status | Untested (code present) | Fully tested |

## Integrity verification

`onyx verify model.rvc` checks:
1. `metadata.json` exists
2. SHA-256 of `model.onnx` matches the hash in `metadata.json`
3. All required entries (`model.onnx`) are present
