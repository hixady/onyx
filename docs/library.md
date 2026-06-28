# `onyx` Library Reference

## Exports

```python
from onyx import run, register, lookup, list_types
```

### `run(input_path, output_path, model_path, arch_type, ...)`

Run inference. Orchestrates container loading, shared model resolution, chunking, and output writing.

```python
run("input.wav", "output.wav", "model.onyx", "rvc",
    speaker_id=0, index_rate=0.5)
```

```python
run("mix.wav", "stems/", "mdx.onyx", "mdx", only="vocals")
```

### `register(type_name)` / `lookup(type_name)` / `list_types()`

Plugin registry for `ModelArch` subclasses.

```python
@register("myarch")
class MyArch(ModelArch):
    ...
```

## Core Components

| Module | Description |
|--------|-------------|
| `core.arch.ModelArch` | Base class for architecture plugins |
| `core.registry` | Plugin registry with entry-point discovery |
| `core.container.Container` | Read/verify/extract `.onyx` files |
| `core.runner.run` | Main inference orchestrator |
| `core.resolver.resolve_shared` | Resolution: CLI → container → cache → error |

## Container API

```python
from onyx.core.container import Container, create_package

with Container("model.onyx") as c:
    meta = c.read_metadata()
    model_bytes = c.read_by_role("model")
    errors = c.verify()

# Generic packaging (roles defined by caller)
file_data = {"model.onnx": onnx_bytes, "rmvpe.onnx": rmvpe_bytes}
create_package(output_path="out.onyx", model_type="rvc",
               metadata={"files": {"model": "model.onnx", "pitch": "rmvpe.onnx"}},
               file_data=file_data)
```
