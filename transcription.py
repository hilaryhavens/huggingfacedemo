import threading
import warnings
import numpy as np
from transformers import pipeline

# Suppress transformers internals unfixable from user code
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*attention_mask.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")

SAMPLE_RATE = 16000
MODEL_ID = "openai/whisper-small"


class Transcriber:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self._pipe = None
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray) -> str:
        self._ensure_loaded()
        with self._lock:
            result = self._pipe({"array": audio, "sampling_rate": SAMPLE_RATE})
        return result["text"].strip()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._pipe is None:
                self._pipe = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_id,
                    device="cpu",
                )
                forced = self._pipe.tokenizer.get_decoder_prompt_ids(
                    language="en", task="transcribe"
                )
                self._pipe.model.config.forced_decoder_ids = forced
