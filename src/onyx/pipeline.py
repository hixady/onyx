import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

from onyx.audio import load_audio, highpass_filter, get_chunk_ranges, stitch_chunks
from onyx.contentvec import ContentVecExtractor
from onyx.f0 import F0Extractor
from onyx.index import IndexManager
from onyx.synthesis import Synthesizer
from onyx.utils import auto_chunk_sec

_RVC_INPUT_SR = 16000


class Pipeline:
    def __init__(self, cv, rmvpe, model, index=None,
                 providers=None, chunk_sec=0, overlap_sec=0.5):
        if providers is None:
            available = ort.get_available_providers()
            preferred = ["CUDAExecutionProvider", "DmlExecutionProvider"]
            providers = [p for p in preferred if p in available] + ["CPUExecutionProvider"]

        self.cv = ContentVecExtractor(cv, providers)
        self.f0 = F0Extractor(rmvpe, providers)
        self.synth = Synthesizer(model, providers)
        self.index = IndexManager(index)
        self.overlap_sec = overlap_sec

        if chunk_sec <= 0:
            chunk_sec = auto_chunk_sec(self.synth.sr_out)
        self.chunk_sec = chunk_sec

        if self.cv.output_dim != self.synth.phone_dim:
            print(f"  WARNING: ContentVec dim ({self.cv.output_dim}) != "
                  f"synthesizer phone dim ({self.synth.phone_dim})")

    def convert(self, input_path, output_path, speaker_id=0,
                f0_method="rmvpe", index_rate=0.5, trim_silence=0):
        t0 = time.time()

        audio = load_audio(input_path)
        audio = highpass_filter(audio)
        dur = len(audio) / _RVC_INPUT_SR

        if trim_silence > 0:
            edge = int(trim_silence * _RVC_INPUT_SR)
            if 2 * edge < len(audio):
                audio = audio[edge:-edge]

        chunk_samps = int(self.chunk_sec * _RVC_INPUT_SR)
        overlap_samps = int(self.overlap_sec * _RVC_INPUT_SR)
        ranges = get_chunk_ranges(len(audio), chunk_samps, overlap_samps)

        audio_outs = []
        for ci, (start_in, end_in) in enumerate(ranges):
            chunk = audio[start_in:end_in]
            features = self.cv.extract(chunk)

            if self.index is not None and index_rate > 0:
                features = self.index.retrieve(features, index_rate)

            if f0_method == "rmvpe":
                f0 = self.f0.extract_rmvpe(chunk, features.shape[0])
            else:
                f0 = self.f0.extract_autocorr(chunk, features.shape[0])

            wav = self.synth.synthesize(features, f0, speaker_id)
            audio_outs.append(wav)

        waveform = stitch_chunks(audio_outs, ranges, len(audio),
                                 self.synth.sr_out, self.overlap_sec)
        peak = np.max(np.abs(waveform)) + 1e-8
        waveform = waveform / peak * 0.9
        audio_int16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(output_path, audio_int16, self.synth.sr_out, subtype="PCM_16")

        elapsed = time.time() - t0
        return {
            "output_path": output_path,
            "sr": self.synth.sr_out,
            "duration": dur,
            "elapsed": elapsed,
            "rtf": dur / max(elapsed, 0.001),
        }
