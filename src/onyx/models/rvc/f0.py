import numpy as np
import onnxruntime as ort

_RVC_INPUT_SR = 16000
_RMVPE_N_MEL = 128
_RMVPE_N_FFT = 1024
_RMVPE_HOP = 160
_RMVPE_WIN = 1024
_RMVPE_FMIN = 30.0
_RMVPE_FMAX = 8000.0


def mel_filterbank(n_fft, n_mel, sr, fmin, fmax):
    n_fft_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_fft_bins)
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mel + 2)
    freq_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    fb = np.zeros((n_mel, n_fft_bins), dtype=np.float32)
    for m in range(n_mel):
        lo, center, hi = freq_points[m], freq_points[m + 1], freq_points[m + 2]
        for k, f in enumerate(fft_freqs):
            if lo <= f <= center:
                fb[m, k] = (f - lo) / (center - lo)
            elif center < f <= hi:
                fb[m, k] = (hi - f) / (hi - center)
    return fb


def stft_mag(audio, n_fft, hop, win):
    window = np.hanning(win).astype(np.float32)
    pad = n_fft // 2
    audio_padded = np.pad(audio, (pad, pad), mode="reflect")
    n_frames = 1 + (len(audio_padded) - win) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio_padded,
        shape=(n_frames, win),
        strides=(audio_padded.strides[0] * hop, audio_padded.strides[0]),
    ).copy()
    windowed = frames * window[np.newaxis, :]
    spec = np.fft.rfft(windowed, n=n_fft, axis=1)
    return np.abs(spec).T.astype(np.float32)


def audio_to_rmvpe_mel(audio):
    mag = stft_mag(audio, n_fft=_RMVPE_N_FFT, hop=_RMVPE_HOP, win=_RMVPE_WIN)
    fb = mel_filterbank(_RMVPE_N_FFT, _RMVPE_N_MEL, _RVC_INPUT_SR, _RMVPE_FMIN, _RMVPE_FMAX)
    mel = fb @ mag
    mel = np.log(np.clip(mel, 1e-5, None))
    return mel[np.newaxis, :, :].astype(np.float32)


def rmvpe_decode(raw, thred=0.03):
    cents_mapping = 20.0 * np.arange(360, dtype=np.float32) + 1997.3794084376191
    cents_mapping_padded = np.pad(cents_mapping, (4, 4))
    n_frames = raw.shape[0]
    f0 = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        salience = raw[i]
        center = np.argmax(salience)
        if salience[center] <= thred:
            continue
        start = max(0, center - 4)
        end = min(360, center + 5)
        local_sal = salience[start:end]
        local_cents = cents_mapping_padded[4 + start : 4 + end]
        weight = local_sal.sum()
        if weight > 0:
            cents = (local_sal * local_cents).sum() / weight
            f0[i] = 10.0 * (2.0 ** (cents / 1200.0))
    return f0


def f0_to_coarse(f0, f0_min=50.0, f0_max=1100.0):
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(np.int64)


def resample_f0_to_length(f0, src_hop, tgt_len):
    if tgt_len == 0 or len(f0) == 0:
        return np.zeros(tgt_len, dtype=np.float32)
    src_t = np.arange(len(f0)) * src_hop
    tgt_t = np.arange(tgt_len) * (src_hop * len(f0) / tgt_len)
    return np.interp(tgt_t, src_t, f0).astype(np.float32)


class F0Extractor:
    def __init__(self, rmvpe_model=None, providers=None):
        self.rmvpe_sess = None
        if rmvpe_model:
            self.rmvpe_sess = ort.InferenceSession(rmvpe_model, providers=providers)

    def extract_rmvpe(self, audio, n_feature_frames):
        if self.rmvpe_sess is None:
            raise RuntimeError("RMVPE model not loaded")
        mel = audio_to_rmvpe_mel(audio)
        T_mel = mel.shape[2]
        pad_to = ((T_mel + 31) // 32) * 32
        if pad_to > T_mel:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, pad_to - T_mel)), mode="reflect")
        out = self.rmvpe_sess.run(None, {"input": mel})
        raw = out[0][0, :T_mel, :]
        f0 = rmvpe_decode(raw)
        return resample_f0_to_length(f0, src_hop=_RMVPE_HOP, tgt_len=n_feature_frames)

    def extract_autocorr(self, audio, n_feature_frames, fmin=50, fmax=1100):
        frame_len = 1024
        hop = 160
        frames = []
        for i in range(0, len(audio) - frame_len, hop):
            frame = audio[i : i + frame_len]
            frame = frame - frame.mean()
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]
            min_lag = int(_RVC_INPUT_SR / fmax)
            max_lag = min(int(_RVC_INPUT_SR / fmin), len(corr) - 1)
            if max_lag <= min_lag:
                frames.append(0)
                continue
            seg = corr[min_lag : max_lag + 1]
            peak_lag = min_lag + np.argmax(seg)
            if corr[peak_lag] > 0.3 * corr[0]:
                frames.append(_RVC_INPUT_SR / peak_lag)
            else:
                frames.append(0)
        f0 = np.array(frames, dtype=np.float32)
        return resample_f0_to_length(f0, src_hop=hop, tgt_len=n_feature_frames)
