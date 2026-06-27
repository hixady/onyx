import os


def detect_free_vram_mb():
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
        from pynvml import nvmlDeviceGetMemoryInfo, nvmlShutdown
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()
        return info.free / (1024 * 1024)
    except Exception:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            free = float(result.stdout.strip())
            return free
    except Exception:
        pass

    return None


def auto_chunk_sec(synth_sr_out, prefer_sec=10):
    free_mb = detect_free_vram_mb()
    if free_mb is None:
        return prefer_sec

    overhead_mb = 300
    mb_per_frame = 2
    safe = free_mb - overhead_mb
    if safe <= 0:
        return 0.5

    max_frames = int(safe / mb_per_frame)
    max_frames = max(20, min(1000, max_frames))
    return max_frames / 100
