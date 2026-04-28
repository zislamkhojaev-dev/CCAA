"""Speech gating before STT: Silero VAD (ONNX) or WebRTC VAD — replaces RMS energy gate."""

from __future__ import annotations

import os
import urllib.request
from collections.abc import AsyncIterator, Callable
from typing import Literal

import numpy as np

from apps.backend.config import get_settings
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)

VadBackend = Literal["silero_onnx", "webrtc"]


def _silero_cache_path() -> str:
    s = get_settings()
    base = (s.silero_vad_cache_dir or "").strip() or os.path.join(
        os.path.expanduser("~"), ".cache", "voiceagent"
    )
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "silero_vad.onnx")


def _ensure_silero_onnx(url: str) -> str:
    path = _silero_cache_path()
    if os.path.isfile(path) and os.path.getsize(path) > 10_000:
        return path
    log.info("silero_vad_downloading", url=url, path=path)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


class _SileroOnnxStream:
    """Streaming Silero VAD (v5.1.2 ONNX) — numpy + onnxruntime, 512-sample chunks @ 16 kHz."""

    def __init__(self, onnx_path: str) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        provs = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            onnx_path, providers=provs, sess_options=opts
        )
        self._sr = np.array(16_000, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def step(self, window_float32: np.ndarray) -> float:
        """window_float32 shape (1, 512) float32 in [-1, 1]. Returns speech probability."""
        if window_float32.shape != (1, 512):
            raise ValueError(f"expected (1, 512), got {window_float32.shape}")
        x = np.concatenate([self._context, window_float32], axis=1).astype(np.float32)
        out, state_n = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._state = state_n.astype(np.float32)
        self._context = x[:, -64:].astype(np.float32)
        return float(out.reshape(-1)[0])


class _WebRtcStream:
    def __init__(self, aggressiveness: int) -> None:
        import webrtcvad

        self._vad = webrtcvad.Vad(max(0, min(3, aggressiveness)))
        self._frame_samples = 480  # 30 ms @ 16 kHz
        self._frame_bytes = self._frame_samples * 2

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    def is_speech(self, frame_pcm16_mono: bytes) -> bool:
        return bool(self._vad.is_speech(frame_pcm16_mono, 16_000))


async def pcm_vad_gate_stream(
    audio_in: AsyncIterator[bytes],
    *,
    sample_rate: int,
    backend: VadBackend,
    runtime_get: Callable[[], dict],
) -> AsyncIterator[bytes]:
    """Pass voiced audio (plus short tails) to STT; drop idle noise."""
    if sample_rate != 16_000:
        log.warning("vad_resample_not_implemented_pass_through", sample_rate=sample_rate)
        async for chunk in audio_in:
            if chunk:
                yield chunk
        return

    rt = runtime_get()
    min_voiced_sec = max(0.05, float(rt.get("stt_min_voiced_seconds", 0.2)))
    prebuffer_sec = max(0.05, float(rt.get("vad_prebuffer_sec", 0.3)))
    hangover_sec = max(0.05, float(rt.get("vad_hangover_sec", 0.25)))
    silero_thr = float(rt.get("vad_silero_speech_threshold", 0.45))
    webrtc_aggr = int(rt.get("vad_webrtc_aggressiveness", 2))

    if backend == "silero_onnx":
        path = _ensure_silero_onnx(get_settings().silero_vad_onnx_url)
        engine = _SileroOnnxStream(path)
        frame_bytes = 1024  # 512 samples * 2
        prob_for_chunk: Callable[[bytes], float] = lambda b: engine.step(
            np.frombuffer(b, dtype=np.int16).astype(np.float32).reshape(1, -1) / 32768.0
        )
    else:
        wr = _WebRtcStream(webrtc_aggr)
        frame_bytes = wr.frame_bytes

        def prob_for_chunk(b: bytes) -> float:
            return 1.0 if wr.is_speech(b) else 0.0

    raw = bytearray()
    min_voiced_bytes = int(sample_rate * 2 * min_voiced_sec)
    prebuffer_max_bytes = int(sample_rate * 2 * prebuffer_sec)
    hangover_max_bytes = int(sample_rate * 2 * hangover_sec)
    prebuffer = bytearray()
    voiced_run = 0
    non_speech_run = 0
    in_speech = False

    async for chunk in audio_in:
        if not chunk:
            continue
        raw.extend(chunk)
        while len(raw) >= frame_bytes:
            frame = bytes(raw[:frame_bytes])
            del raw[:frame_bytes]
            p = prob_for_chunk(frame)
            speech = p >= silero_thr if backend == "silero_onnx" else p >= 0.5
            if speech:
                voiced_run += len(frame)
                non_speech_run = 0
                if not in_speech:
                    prebuffer.extend(frame)
                    if len(prebuffer) > prebuffer_max_bytes:
                        del prebuffer[: len(prebuffer) - prebuffer_max_bytes]
                    if voiced_run >= min_voiced_bytes:
                        in_speech = True
                        if prebuffer:
                            yield bytes(prebuffer)
                            prebuffer.clear()
                else:
                    yield frame
                continue
            if in_speech:
                if non_speech_run < hangover_max_bytes:
                    yield frame
                non_speech_run += len(frame)
                if non_speech_run >= hangover_max_bytes:
                    in_speech = False
                    voiced_run = 0
                    non_speech_run = 0
                    prebuffer.clear()
            else:
                prebuffer.clear()
                voiced_run = 0

    if in_speech and raw:
        yield bytes(raw)
