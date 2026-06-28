import numpy as np
import soundfile as sf
from scipy import signal as scipy_signal

_RVC_INPUT_SR = 16000


def load_audio(path, target_sr=None):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        n_out = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        return audio, target_sr
    return audio, sr


def highpass_filter(audio, cutoff=48, order=5, sr=_RVC_INPUT_SR):
    bh, ah = scipy_signal.butter(N=order, Wn=cutoff, btype="high", fs=sr)
    return scipy_signal.filtfilt(bh, ah, audio).astype(np.float32)


def get_chunk_ranges(total_len, chunk_size, overlap):
    hop = chunk_size - overlap
    ranges = []
    start = 0
    while start < total_len:
        end = min(start + chunk_size, total_len)
        ranges.append((start, end))
        if end == total_len:
            break
        start += hop
    return ranges


def stitch_chunks(chunks, ranges, total_input_len, sr_out, overlap_sec, sr_in=16000):
    output_sr_ratio = sr_out / sr_in
    total_output_len = int(total_input_len * output_sr_ratio)
    overlap_out = int(overlap_sec * sr_out) if len(ranges) > 1 else 0

    if overlap_out <= 0 or len(chunks) <= 1:
        return np.concatenate(chunks)[:total_output_len]

    fade = np.sin(np.linspace(0, np.pi / 2, overlap_out)) ** 2
    result = np.zeros(total_output_len, dtype=np.float32)
    wsum = np.zeros(total_output_len, dtype=np.float32)

    for i, (chunk, (start_in, _)) in enumerate(zip(chunks, ranges)):
        offset_out = int(start_in * output_sr_ratio)
        clen = len(chunk)
        end_out = min(offset_out + clen, total_output_len)
        actual_clen = end_out - offset_out
        ol = min(overlap_out, actual_clen)

        if i == 0:
            win = np.ones(actual_clen, dtype=np.float32)
            if len(chunks) > 1:
                win[-ol:] = 1 - fade[:ol]
        elif i == len(chunks) - 1:
            win = np.ones(actual_clen, dtype=np.float32)
            win[:ol] = fade[:ol]
        else:
            win = np.ones(actual_clen, dtype=np.float32)
            win[:ol] = fade[:ol]
            win[-ol:] = 1 - fade[:ol]

        result[offset_out:end_out] += chunk[:actual_clen] * win[:actual_clen]
        wsum[offset_out:end_out] += win[:actual_clen]

    wsum = np.maximum(wsum, 1e-8)
    result /= wsum
    return result
