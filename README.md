# onyx

Multi-arch ONNX inference framework and container format. Runs voice conversion (RVC), source separation (MDX-Net), and more on ONNX Runtime with CPU, CUDA, or DirectML.

## Install

```bash
pip install onyx              # core only (no runtime)
pip install onyx[cpu]         # ONNX Runtime CPU
pip install onyx[cuda]        # ONNX Runtime CUDA
pip install onyx[dml]         # ONNX Runtime DirectML
```

## Quick start

```bash
# Voice conversion (RVC)
onyx run rvc -i input.wav -o output.wav --model model.onyx

# Source separation (MDX-Net)
onyx run mdx -i mix.wav -o stems/ --model mdx.onyx

# Single stem extraction
onyx run mdx -i mix.wav -o vocals.wav --model mdx.onyx --only vocals

# Package files into .onyx
onyx pack -o model.onyx --type rvc --model model.onnx

# Verify container integrity
onyx verify model.onyx
```

Provider priority (auto): `CUDA` → `DirectML` → `CPU`

## Architectures

| Type | Description |
|------|-------------|
| `rvc` | RVC voice conversion (ContentVec + RMVPE + Synthesizer) |
| `mdx` | MDX-Net music source separation (vocals, bass, drums, other) |

## Documentation

- [CLI reference](docs/cli.md) — all subcommands and options
- [Library API](docs/library.md) — using onyx from Python
- [Container format](docs/format.md) — `.onyx` file specification
