# `.onyx` Container Format

`.onyx` is a ZIP-based container (stored, no compression) that bundles ONNX models with supporting files.

## Structure

```
model.onyx
├── metadata.json      # JSON manifest (required)
├── *.onnx / *.wav     # Model and support files
└── ...
```

## metadata.json

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | int | yes | Format version (currently `1`) |
| `type` | string | yes | Model architecture type (e.g. `"rvc"`, `"mdx"`) |
| `files` | object | yes | Logical roles → archive paths |
| `sha256` | object | yes | Per-file sha256 hex digests |
| `sample_rate` | int | no | Output sample rate |
| `phone_dim` | int | no | Embedding dimension (RVC) |
| `config` | object | no | Arch-specific config (e.g. MDX source params) |

## `files` roles (RVC)

| Role | Archive path | Type | Description |
|------|-------------|------|-------------|
| `model` | `model.onnx` | required | Main ONNX model |
| `pitch` | `rmvpe.onnx` | optional | Pitch extraction model |
| `embedding` | `contentvec.onnx` | optional | Embedding/extractor model |
| `index` | `model.index` | optional | FAISS index for feature retrieval |
| `demo` | `demo.wav` | optional | Demo audio |
| `icon` | `icon.png` | optional | Model icon |

## `files` roles (MDX)

| Role | Archive path | Type | Description |
|------|-------------|------|-------------|
| `model_vocals` | `vocals.onnx` | required | Vocals separator |
| `model_bass` | `bass.onnx` | required | Bass separator |
| `model_drums` | `drums.onnx` | required | Drums separator |
| `model_other` | `other.onnx` | required | Other separator |
| `mixer` | `mixer.onnx` | optional | Post-processing mixer |

Roles are defined per-architecture by the converter tool (`onyx-conv`) and are not hardcoded in the container format.

## Example (RVC)

```json
{
    "version": 1,
    "type": "rvc",
    "files": {
        "model": "model.onnx",
        "pitch": "rmvpe.onnx",
        "embedding": "contentvec.onnx",
        "index": "model.index"
    },
    "sha256": {
        "model.onnx": "abc...",
        "rmvpe.onnx": "def..."
    },
    "sample_rate": 48000,
    "phone_dim": 768
}
```

## Example (MDX)

```json
{
    "version": 1,
    "type": "mdx",
    "files": {
        "model_vocals": "vocals.onnx",
        "model_bass": "bass.onnx",
        "model_drums": "drums.onnx",
        "model_other": "other.onnx",
        "mixer": "mixer.onnx"
    },
    "sha256": {
        "vocals.onnx": "abc...",
        "bass.onnx": "def..."
    },
    "sample_rate": 44100,
    "config": {
        "hop_length": 1024,
        "dim_t": 512,
        "dim_c": 4,
        "sources": {
            "vocals": { "n_fft": 6144, "dim_f": 2048 },
            "bass": { "n_fft": 16384, "dim_f": 2048 },
            "drums": { "n_fft": 4096, "dim_f": 2048 },
            "other": { "n_fft": 8192, "dim_f": 2048 }
        }
    }
}
```
