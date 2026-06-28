import numpy as np
import onnx
import onnxruntime as ort

from onyx.models.rvc.f0 import f0_to_coarse


class Synthesizer:
    def __init__(self, model, providers=None):
        if isinstance(model, bytes):
            self.sess = ort.InferenceSession(model, providers=providers)
            self.sr_out = self._detect_sr_from_bytes(model)
        else:
            self.sess = ort.InferenceSession(model, providers=providers)
            self.sr_out = self._detect_sr_from_bytes(open(model, "rb").read())
        self.phone_dim = self._detect_phone_dim()

    @staticmethod
    def _detect_sr_from_bytes(data):
        try:
            model = onnx.load_model_from_string(data)
            ratio = 1
            for node in model.graph.node:
                if node.op_type == "ConvTranspose":
                    for attr in node.attribute:
                        if attr.name == "strides":
                            ratio *= attr.ints[0]
            return ratio * 100
        except Exception:
            pass
        return 40000

    def _detect_phone_dim(self):
        for inp in self.sess.get_inputs():
            if inp.name == "phone":
                shape = inp.shape
                if len(shape) == 3:
                    dim = shape[2]
                    if isinstance(dim, int):
                        return dim
        return 768

    @property
    def version(self):
        return "v1" if self.phone_dim == 256 else "v2"

    def synthesize(self, features, f0, speaker_id=0):
        T = features.shape[0]
        phone = features[np.newaxis, :, :].astype(np.float32)
        phone_lengths = np.array([T], dtype=np.int64)
        f0_coarse = f0_to_coarse(f0)
        pitch = f0_coarse[np.newaxis, :].astype(np.int64)
        pitchf = f0[np.newaxis, :].astype(np.float32)
        ds = np.array([speaker_id], dtype=np.int64)
        rnd = np.random.randn(1, 192, T).astype(np.float32)
        out = self.sess.run(None, {
            "phone": phone, "phone_lengths": phone_lengths,
            "pitch": pitch, "pitchf": pitchf,
            "ds": ds, "rnd": rnd,
        })
        return out[0][0, 0]
