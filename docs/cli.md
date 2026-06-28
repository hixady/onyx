# `onyx` CLI Reference

## Usage

```bash
onyx <command> [options]
```

## Commands

### `onyx run <arch>`

Run inference with a registered model architecture.

```bash
onyx run rvc -i input.wav -o output.wav --model model.onyx \
    --rmvpe rmvpe.onnx --cv contentvec.onnx \
    --index model.index --speaker-id 0 --index-rate 0.5
```

```bash
onyx run mdx -i mix.wav -o stems/ --model mdx.onyx \
    --mixer mixer.onnx \
    --only vocals
```

**Core args:** `-i`, `-o`, `--model`, `--chunk-sec`, `--overlap-sec`

**Arch-specific args** (shown by `onyx run <arch> --help`).

### `onyx pack`

Build a `.onyx` container from source files.

```bash
onyx pack -o model.onyx --type rvc --model model.onnx \
    --rmvpe rmvpe.onnx --cv contentvec.onnx --index model.index
```

### `onyx unpack`

Extract a `.onyx` container to a directory.

```bash
onyx unpack model.onyx -o outdir/
```

### `onyx verify`

Verify `.onyx` container integrity (sha256 + all referenced files).

```bash
onyx verify model.onyx
```

### `onyx models`

List registered model types and cache directory.

```bash
onyx models
```
