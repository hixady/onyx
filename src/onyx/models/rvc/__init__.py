import time

import numpy as np
import onnxruntime as ort

from onyx.audio import load_audio, highpass_filter
from onyx.core.arch import ModelArch
from onyx.core.container import Container
from onyx.core.registry import register
from onyx.models.rvc.contentvec import ContentVecExtractor
from onyx.models.rvc.f0 import F0Extractor
from onyx.models.rvc.index import IndexManager
from onyx.models.rvc.synthesis import Synthesizer


@register("rvc")
class RVCArch(ModelArch):
    type = "rvc"
    description = "RVC voice conversion"
    input_sr = 16000
    shared_models = {
        "embedding": ["contentvec_768l12.onnx", "contentvec.onnx"],
        "pitch": ["rmvpe_fp32.onnx", "rmvpe.onnx"],
    }
    chunkable = True
    supports = ["audio"]

    @staticmethod
    def add_args(p):
        p.add_argument("--rmvpe", default=None)
        p.add_argument("--cv", default=None)
        p.add_argument("--index", default=None)
        p.add_argument("--speaker-id", type=int, default=0)
        p.add_argument("--f0-method", choices=["rmvpe", "autocorr"], default="rmvpe")
        p.add_argument("--index-rate", type=float, default=0.5)

    def validate(self, container: Container):
        meta = container.read_metadata()
        for key in ("sample_rate", "phone_dim"):
            if key not in meta:
                raise ValueError(f"Missing required metadata: {key}")

    def run(self, audio: np.ndarray, sr: int, container: Container,
            shared: dict[str, bytes], **kwargs) -> dict[str, np.ndarray]:
        providers = self._select_providers()

        model_bytes = container.read_by_role("model")
        rmvpe_data = shared.get("pitch")
        cv_data = shared.get("embedding")

        cv = ContentVecExtractor(cv_data, providers)
        f0 = F0Extractor(rmvpe_data, providers)
        synth = Synthesizer(model_bytes, providers)
        idx_mgr = IndexManager(container.read_by_role("index")
                               if kwargs.get("index") is None
                               and "index" in container.get_files()
                               else kwargs.get("index"))

        if cv.output_dim != synth.phone_dim:
            print(f"  WARNING: ContentVec dim ({cv.output_dim}) != "
                  f"synthesizer phone dim ({synth.phone_dim})")

        speaker_id = kwargs.get("speaker_id", 0)
        f0_method = kwargs.get("f0_method", "rmvpe")
        index_rate = kwargs.get("index_rate", 0.5)

        features = cv.extract(audio)
        if idx_mgr is not None and index_rate > 0:
            features = idx_mgr.retrieve(features, index_rate)
        if f0_method == "rmvpe":
            f0_vals = f0.extract_rmvpe(audio, features.shape[0])
        else:
            f0_vals = f0.extract_autocorr(audio, features.shape[0])
        wav = synth.synthesize(features, f0_vals, speaker_id)
        return {"audio": wav}

    @staticmethod
    def _select_providers():
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider"]
        return [p for p in preferred if p in available] + ["CPUExecutionProvider"]
