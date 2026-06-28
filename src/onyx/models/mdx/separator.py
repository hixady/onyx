import numpy as np
import onnxruntime as ort

from onyx.models.mdx.stft import mdx_stft, mdx_istft, hann_window


class MDXSeparator:
    def __init__(self, onnx_data: bytes, n_fft: int, dim_f: int, hop_length: int,
                 dim_c: int = 4, providers=None):
        self.n_fft = n_fft
        self.dim_f = dim_f
        self.hop_length = hop_length
        self.dim_c = dim_c
        self.trim = n_fft // 2
        self.window = hann_window(n_fft, periodic=True)

        self.sess = ort.InferenceSession(onnx_data, providers=providers)
        inp = self.sess.get_inputs()[0]
        out = self.sess.get_outputs()[0]
        self.dim_t = inp.shape[3]
        self.chunk_size = hop_length * (self.dim_t - 1)

    def separate(self, mix: np.ndarray) -> np.ndarray:
        n_sample = mix.shape[1]
        gen_size = self.chunk_size - 2 * self.trim
        pad = (gen_size - n_sample % gen_size) % gen_size

        mix_p = np.zeros((2, self.trim + n_sample + pad + self.trim), dtype=np.float32)
        mix_p[:, self.trim:self.trim + n_sample] = mix[:, :n_sample]

        n_chunks = (n_sample + pad + gen_size - 1) // gen_size
        chunks = np.zeros((n_chunks, 2, self.chunk_size), dtype=np.float32)
        for i in range(n_chunks):
            offset = i * gen_size
            chunks[i] = mix_p[:, offset:offset + self.chunk_size]

        spec = mdx_stft(chunks, self.n_fft, self.hop_length, self.dim_f, window=self.window)
        out_spec = self.sess.run(None, {"input": spec})[0]
        tar_waves = mdx_istft(out_spec, self.n_fft, self.hop_length,
                              self.dim_f, self.dim_c, window=self.window)

        tar_signal = tar_waves[:, :, self.trim:-self.trim].transpose(1, 0, 2).reshape(2, -1)
        if pad > 0:
            tar_signal = tar_signal[:, :-pad]
        return tar_signal


def apply_mixer(mixer_onnx: bytes, sources: dict[str, np.ndarray],
                mix: np.ndarray, providers=None):
    sess = ort.InferenceSession(mixer_onnx, providers=providers)
    dim_s = len(sources)
    names = list(sources.keys())
    x = np.stack([sources[n] for n in names] + [mix], axis=0)
    x = x.reshape(1, (dim_s + 1) * 2, -1).transpose(0, 2, 1)
    out = sess.run(None, {"input": x.astype(np.float32)})[0]
    out = out.transpose(0, 2, 1).reshape(dim_s, 2, -1)
    return {names[i]: out[i] for i in range(dim_s)}
