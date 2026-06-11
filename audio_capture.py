import threading
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
MIN_DURATION_SAMPLES = int(SAMPLE_RATE * 0.5)  # 0.5 seconds


class AudioRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        with self._lock:
            self._frames.append(indata.copy())

    def drain(self) -> np.ndarray | None:
        with self._lock:
            frames, self._frames = self._frames, []
        if not frames:
            return None
        audio = np.concatenate(frames, axis=0).flatten()
        if len(audio) < MIN_DURATION_SAMPLES:
            return None
        return audio

    def stop(self) -> np.ndarray | None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            frames, self._frames = self._frames, []
        if not frames:
            return None
        audio = np.concatenate(frames, axis=0).flatten()
        if len(audio) < MIN_DURATION_SAMPLES:
            return None
        return audio
