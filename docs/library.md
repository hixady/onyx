# Using onyx as a Python library

## Installation

```bash
pip install onyx[cpu]    # or [cuda] or [dml]
```

## RVCModel — high-level API

The main entry point. Loads a model from a `.rvc` container or separate `.onnx` files and runs conversion.

```python
from onyx import RVCModel

# From .rvc container (recommended)
model = RVCModel("model.rvc")
model.convert("input.wav", "output.wav")

# From separate files
model = RVCModel("model.onnx", rmvpe="rmvpe.onnx", cv="contentvec.onnx")
model.convert("input.wav", "output.wav")

# With custom settings
model = RVCModel(
    model="model.rvc",
    chunk_sec=10,           # seconds per chunk (0 = auto)
    overlap_sec=0.5,        # crossfade overlap
    verify=True,            # verify sha256 on load
    providers=None,         # auto-select CUDA → DML → CPU
)
```

### RVCModel()

```python
RVCModel(
    model: str | Path,         # Path to .rvc or .onnx
    rmvpe: str | None = None,  # Override RMVPE path
    cv: str | None = None,     # Override ContentVec path
    index: str | None = None,  # Override index path
    providers: list[str] | None = None,  # ORT provider list
    chunk_sec: float = 10,     # Chunk size in seconds
    overlap_sec: float = 0.5,  # Crossfade overlap
    verify: bool = True,       # Verify .rvc sha256
)
```

### .convert()

```python
model.convert(
    input_path: str,       # Input audio file
    output_path: str,      # Output WAV path
    speaker_id: int = 0,   # Speaker ID
    f0_method: str = "rmvpe",  # "rmvpe" or "autocorr"
    index_rate: float = 0.5,   # Retrieval blend (0 = none, 1 = full)
    trim_silence: int = 0,     # Trim edge silence (seconds)
) -> dict
```

Returns a result dict:
```python
{
    "output_path": "output.wav",
    "sr": 48000,
    "duration": 237.2,     # Input duration in seconds
    "elapsed": 19.4,       # Processing time in seconds
    "rtf": 12.3,           # Real-time factor
}
```

### .pack()

```python
RVCModel.pack(
    output_path: str,
    model: str | bytes,      # ONNX path or bytes
    index: str | None = None,
    rmvpe: str | None = None,
    cv: str | None = None,
    name: str | None = None,
    author: str | None = None,
    notes: str | None = None,
    tags: str | None = None,
    demo: str | None = None,
    icon: str | None = None,
    metadata: dict | None = None,
)
```

### .verify() / .unpack()

```python
errors = RVCModel.verify("model.rvc")  # Returns list of error strings
RVCModel.unpack("model.rvc", "output_dir/")
```

### Context manager

`RVCModel` supports `with` for automatic cleanup:

```python
with RVCModel("model.rvc") as model:
    result = model.convert("input.wav", "output.wav")
    print(f"{result['duration']:.1f}s in {result['elapsed']:.1f}s")
```

### Properties

```python
model.name        # Model name (from metadata or filename)
model.version     # "v1" or "v2"
model.sample_rate # Output sample rate (e.g. 40000, 48000)
model.metadata    # Dict of metadata.json contents
```

## Pipeline — lower-level API

For more control over individual pipeline stages:

```python
from onyx.pipeline import Pipeline
from onyx.audio import load_audio, highpass_filter, get_chunk_ranges, stitch_chunks

pipeline = Pipeline(
    cv="contentvec.onnx",   # ContentVec ONNX path or bytes
    rmvpe="rmvpe.onnx",     # RMVPE ONNX path or bytes
    model="model.onnx",     # Synthesizer ONNX path or bytes
    index="model.index",    # FAISS index path or bytes (optional)
    providers=None,         # Auto-select providers
    chunk_sec=10,
    overlap_sec=0.5,
)

# Run conversion
result = pipeline.convert("input.wav", "output.wav")

# Access individual components
pipeline.cv       # ContentVecExtractor
pipeline.f0       # F0Extractor
pipeline.synth    # Synthesizer
pipeline.index    # IndexManager
```

## Individual components

### ContentVecExtractor

```python
from onyx.contentvec import ContentVecExtractor

extractor = ContentVecExtractor("contentvec.onnx")
features = extractor.extract(audio_16khz)  # Returns (T, 768) array
```

### F0Extractor

```python
from onyx.f0 import F0Extractor

f0 = F0Extractor("rmvpe.onnx")
f0_rmvpe = f0.extract_rmvpe(audio_16khz, n_frames)   # GPU RMVPE
f0_autocorr = f0.extract_autocorr(audio_16khz, n_frames)  # CPU autocorrelation
```

### Synthesizer

```python
from onyx.synthesis import Synthesizer

synth = Synthesizer("model.onnx")
wav = synth.synthesize(features, f0, speaker_id=0)
print(synth.sr_out)    # Output sample rate
print(synth.phone_dim) # Phone embedding dimension
print(synth.version)   # "v1" or "v2"
```

### IndexManager

```python
from onyx.index import IndexManager

manager = IndexManager("model.index")  # Path or bytes
retrieved = manager.retrieve(features, index_rate=0.5)
```

Set `index=None` to disable retrieval.

### RVCContainer

```python
from onyx.container import RVCContainer, create_rvc_package

# Read
with RVCContainer("model.rvc") as c:
    model_bytes = c.read("model.onnx")
    meta = c.read_metadata()
    c.has_rmvpe  # bool

# Write
create_rvc_package("model.rvc", onnx_path="model.onnx")
```

## Provider selection

If `providers=None`, onyx auto-selects:

1. `CUDAExecutionProvider` — NVIDIA GPU (SM 7.0+)
2. `DmlExecutionProvider` — DirectML (any GPU via DirectX 12)
3. `CPUExecutionProvider` — fallback

To force a specific provider:

```python
model = RVCModel("model.rvc", providers=["CPUExecutionProvider"])
```

## Chunked inference

Long audio is split into chunks, processed independently, and stitched with Hann² crossfade:

```python
model = RVCModel("model.rvc", chunk_sec=10, overlap_sec=0.5)
```

- `chunk_sec=0` — auto-calculates chunk size from estimated free VRAM
- Larger chunks = less overhead but more VRAM
- Crossfade prevents audible seams between chunks
