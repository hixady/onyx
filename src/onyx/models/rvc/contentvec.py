import numpy as np
import onnxruntime as ort


class ContentVecExtractor:
    def __init__(self, model, providers=None):
        self.sess = ort.InferenceSession(model, providers=providers)
        inp = self.sess.get_inputs()
        self.input_names = [i.name for i in inp]
        self.first_input = self.input_names[0]
        self.use_attention_mask = "attention_mask" in self.input_names
        self.output_dim = self._detect_output_dim()

    def _detect_output_dim(self):
        out = self.sess.get_outputs()[0]
        shape = out.shape
        if len(shape) == 3:
            for d in shape[1:]:
                if isinstance(d, int):
                    return d
        return 768

    def extract(self, audio):
        x = audio[np.newaxis, :].astype(np.float32)
        if self.first_input == "source":
            x = x[:, np.newaxis, :]

        feed = {self.first_input: x}
        if self.use_attention_mask:
            feed["attention_mask"] = np.ones((1, x.shape[1]), dtype=np.int64)

        out = self.sess.run(None, feed)
        hidden = out[0]
        if hidden.ndim == 3 and hidden.shape[1] == self.output_dim:
            feats_50hz = hidden[0]
        elif hidden.ndim == 3 and hidden.shape[2] == self.output_dim:
            feats_50hz = hidden[0]
        else:
            feats_50hz = hidden[0]
            if feats_50hz.shape[0] == self.output_dim:
                feats_50hz = feats_50hz.T

        feats = np.repeat(feats_50hz, 2, axis=0)
        return feats
