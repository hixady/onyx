import numpy as np


def hann_window(n, periodic=True):
    if periodic:
        return np.hanning(n + 1)[:n].astype(np.float32)
    return np.hanning(n).astype(np.float32)


def stft(audio, n_fft, hop_length, window=None, center=True):
    """numpy STFT matching torch.stft(center=True, return_complex=False).
    audio: (batch, T) or (T,)
    returns: (batch, n_fft//2+1, n_frames, 2) or (n_fft//2+1, n_frames, 2)
    """
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    batch, T = audio.shape

    if window is None:
        window = hann_window(n_fft)
    n_bins = n_fft // 2 + 1

    if center:
        pad = n_fft // 2
        audio = np.pad(audio, [(0, 0), (pad, pad)], mode="reflect")

    n_frames = 1 + (audio.shape[1] - n_fft) // hop_length

    result = np.zeros((batch, n_bins, n_frames, 2), dtype=np.float32)
    for b in range(batch):
        for t in range(n_frames):
            start = t * hop_length
            frame = audio[b, start:start + n_fft] * window
            spec = np.fft.rfft(frame.astype(np.float64)).astype(np.complex64)
            result[b, :, t, 0] = spec.real
            result[b, :, t, 1] = spec.imag
    return result


def istft(spec, n_fft, hop_length, window=None, center=True, length=None):
    """numpy ISTFT matching torch.istft(center=True).
    spec: (batch, n_bins, n_frames, 2)
    returns: (batch, T)
    """
    batch, n_bins, n_frames, _ = spec.shape

    if window is None:
        window = hann_window(n_fft)

    expected_bins = n_fft // 2 + 1
    if n_bins != expected_bins:
        raise ValueError(f"expected {expected_bins} bins, got {n_bins}")

    out_len = (n_frames - 1) * hop_length + n_fft
    audio = np.zeros((batch, out_len), dtype=np.float32)
    window_sum = np.zeros(out_len, dtype=np.float32)

    for b in range(batch):
        for t in range(n_frames):
            start = t * hop_length
            complex_spec = spec[b, :, t, 0] + 1j * spec[b, :, t, 1]
            frame = np.fft.irfft(complex_spec.astype(np.complex64), n=n_fft).real.astype(np.float32)
            audio[b, start:start + n_fft] += frame * window
            window_sum[start:start + n_fft] += window

    window_sum = np.maximum(window_sum, 1e-10)
    audio /= window_sum

    if center:
        pad = n_fft // 2
        audio = audio[:, pad:-pad] if out_len > 2 * pad else audio[:, :0]

    if length is not None and audio.shape[1] > length:
        audio = audio[:, :length]

    return audio


def mdx_stft(audio, n_fft, hop_length, dim_f, window=None):
    """MDX-style STFT: (N, 2, T) -> (N, 4, dim_f, dim_t).
    Matches Conv_TDF_net_trim.stft() behavior.
    """
    dim_b = audio.shape[0]
    x = audio.reshape(dim_b * 2, -1)
    s = stft(x, n_fft, hop_length, window=window, center=True)
    # s: (dim_b*2, n_bins, dim_t, 2)
    s = s.transpose(0, 3, 1, 2)  # (dim_b*2, 2, n_bins, dim_t)
    n_bins = s.shape[2]
    dim_t = s.shape[3]
    s = s.reshape(dim_b, 2, 2, n_bins, dim_t).reshape(dim_b, 4, n_bins, dim_t)
    return s[:, :, :dim_f, :].copy()


def mdx_istft(spec, n_fft, hop_length, dim_f, dim_c=4, window=None):
    """MDX-style ISTFT: (N, 4, dim_f, dim_t) -> (N, 2, chunk_size).
    Matches Conv_TDF_net_trim.istft() behavior.
    """
    dim_b = spec.shape[0]
    n_bins = n_fft // 2 + 1
    dim_t = spec.shape[3]
    freq_pad = np.zeros((1, dim_c, n_bins - dim_f, dim_t), dtype=np.float32)
    x = np.concatenate([spec, freq_pad.repeat(dim_b, axis=0)], axis=2)
    x = x.reshape(dim_b, 2, 2, n_bins, dim_t).reshape(dim_b * 2, 2, n_bins, dim_t)
    x = x.transpose(0, 2, 3, 1)  # (dim_b*2, n_bins, dim_t, 2)
    out = istft(x, n_fft, hop_length, window=window, center=True)
    dim_b_2, T = out.shape
    return out.reshape(dim_b, 2, T)
