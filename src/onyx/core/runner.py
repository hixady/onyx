import time
from pathlib import Path

import numpy as np
import soundfile as sf

from onyx.audio import load_audio, get_chunk_ranges, stitch_chunks
from onyx.core.container import Container
from onyx.core.registry import lookup
from onyx.core.resolver import resolve_shared


def _normalize(wav):
    if wav.ndim == 1:
        peak = np.max(np.abs(wav)) + 1e-8
        return (wav / peak * 0.9).astype(np.float32)
    else:
        peak = np.max(np.abs(wav)) + 1e-8
        return (wav / peak * 0.9).astype(np.float32)


def write_outputs(output_path, outputs, sr):
    names = list(outputs.keys())
    if len(names) == 1:
        p = Path(output_path)
        if not p.suffix:
            p = p.with_suffix(".wav")
        wav = _normalize(outputs[names[0]])
        if wav.ndim == 2:
            wav = wav.T
        sf.write(str(p), wav, sr, subtype="PCM_16")
        return

    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        wav = _normalize(outputs[name])
        if wav.ndim == 2:
            wav = wav.T
        sf.write(dst / f"{name}.wav", wav, sr, subtype="PCM_16")


def run(input_path, output_path, model_path, arch_type,
        chunk_sec=0, overlap_sec=0.5, **arch_kwargs):
    t0 = time.time()

    container = Container(model_path)
    arch_cls = lookup(arch_type)
    arch = arch_cls()
    arch.validate(container)

    shared = resolve_shared(arch, container, arch_kwargs)
    sr_out = container.read_metadata().get("sample_rate", 44100)
    target_sr = getattr(arch, "input_sr", None)
    audio, sr = load_audio(input_path, target_sr=target_sr)
    dur = len(audio) / sr

    if arch.chunkable and chunk_sec > 0:
        chunk_samps = int(chunk_sec * sr)
        overlap_samps = int(overlap_sec * sr)
        ranges = get_chunk_ranges(len(audio), chunk_samps, overlap_samps)
        outputs_list = []
        for start, end in ranges:
            chunk = audio[start:end]
            outputs_list.append(arch.run(chunk, sr, container, shared, **arch_kwargs))
        key = arch.supports[0]
        merged = stitch_chunks(
            [o[key] for o in outputs_list], ranges, len(audio), sr_out, overlap_sec, sr_in=sr
        )
        outputs = {key: merged}
    else:
        outputs = arch.run(audio, sr, container, shared, **arch_kwargs)

    write_outputs(output_path, outputs, sr_out)

    elapsed = time.time() - t0
    print(f"Saved {output_path} ({dur:.1f}s in {elapsed:.1f}s, {dur/max(elapsed,0.001):.1f}x real-time)")
    return outputs
