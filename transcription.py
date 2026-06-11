import numpy as np
from transformers import pipeline

SAMPLE_RATE = 16000
MODEL_ID = "openai/whisper-small"


class Transcriber:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._pipe = None

    def transcribe(self, audio: np.ndarray) -> str:
        self._ensure_loaded()
        attention_mask = np.ones(len(audio), dtype=np.int32)
        result = self._pipe({"array": audio, "sampling_rate": SAMPLE_RATE, "attention_mask": attention_mask})
        return result["text"].strip()

    def _ensure_loaded(self) -> None:
        if self._pipe is None:
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device="cpu",
            )
