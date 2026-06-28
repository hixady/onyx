from pathlib import Path

import numpy as np
import onnxruntime as ort

from onyx.core.arch import ModelArch
from onyx.core.container import Container
from onyx.core.registry import register
from onyx.models.mdx.separator import MDXSeparator, apply_mixer

SOURCES = ["bass", "drums", "other", "vocals"]


@register("mdx")
class MDXArch(ModelArch):
    type = "mdx"
    description = "MDX-Net music source separation (vocals, bass, drums, other)"
    input_sr = 44100
    shared_models = {}
    chunkable = False
    supports = SOURCES

    @staticmethod
    def add_args(p):
        p.add_argument("--mixer", default=None,
                       help="Path to mixer ONNX model (optional)")
        p.add_argument("--only", default=None,
                       help="Extract only a single stem (default: all)")

    def validate(self, container: Container):
        meta = container.read_metadata()
        if "config" not in meta:
            raise ValueError("Missing 'config' in metadata")
        cfg = meta["config"]
        for key in ("hop_length", "dim_t", "sources"):
            if key not in cfg:
                raise ValueError(f"Missing config key: {key}")

    def _available_sources(self, container: Container) -> list[str]:
        files = container.read_metadata().get("files", {})
        return [k.removeprefix("model_") for k in files if k.startswith("model_")]

    def run(self, audio, sr, container, shared, **kwargs):
        providers = self._select_providers()
        meta = container.read_metadata()
        cfg = meta["config"]
        hop = cfg["hop_length"]
        available = self._available_sources(container)

        only = kwargs.get("only")
        if only and only not in available:
            raise ValueError(f"Stem '{only}' not in container (available: {available})")
        targets = [only] if only else available

        sources_cfg = cfg["sources"]
        separators = {}
        for name in targets:
            role = f"model_{name}"
            onnx_data = container.read_by_role(role)
            src_cfg = sources_cfg.get(name, {})
            n_fft = src_cfg.get("n_fft", 6144)
            dim_f = src_cfg.get("dim_f", 2048)
            separators[name] = MDXSeparator(
                onnx_data, n_fft=n_fft, dim_f=dim_f,
                hop_length=hop, providers=providers
            )

        if audio.ndim == 1:
            mix = np.stack([audio, audio], axis=0)
        elif audio.shape[0] == 1:
            mix = np.vstack([audio, audio])
        elif audio.shape[0] > 2:
            mix = audio[:2]
        else:
            mix = audio

        outputs = {}
        for name, sep in separators.items():
            outputs[name] = sep.separate(mix)

        if len(outputs) >= 2 and (kwargs.get("mixer") or "mixer" in container.get_files()):
            try:
                mixer_path = kwargs.get("mixer")
                if mixer_path:
                    mixer_data = Path(mixer_path).read_bytes()
                else:
                    mixer_data = container.read_by_role("mixer")
                outputs = apply_mixer(mixer_data, outputs, mix, providers)
            except Exception:
                pass

        return outputs

    @staticmethod
    def _select_providers():
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider"]
        return [p for p in preferred if p in available] + ["CPUExecutionProvider"]
