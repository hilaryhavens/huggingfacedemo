import numpy as np
from unittest.mock import patch, MagicMock


def test_transcribe_returns_stripped_text():
    from transcription import Transcriber
    transcriber = Transcriber()
    transcriber._pipe = MagicMock(return_value={"text": "  hello world  "})
    audio = np.zeros(16000, dtype="float32")
    result = transcriber.transcribe(audio)
    assert result == "hello world"


def test_pipeline_loaded_lazily():
    from transcription import Transcriber
    transcriber = Transcriber()
    assert transcriber._pipe is None


def test_pipeline_loaded_only_once():
    from transcription import Transcriber
    transcriber = Transcriber()
    mock_pipe = MagicMock(return_value={"text": "test"})
    with patch("transcription.pipeline", return_value=mock_pipe) as mock_factory:
        audio = np.zeros(16000, dtype="float32")
        transcriber.transcribe(audio)
        transcriber.transcribe(audio)
        mock_factory.assert_called_once()


def test_transcribe_passes_correct_sampling_rate():
    from transcription import Transcriber, SAMPLE_RATE
    transcriber = Transcriber()
    mock_pipe = MagicMock(return_value={"text": "hi"})
    transcriber._pipe = mock_pipe
    audio = np.zeros(16000, dtype="float32")
    transcriber.transcribe(audio)
    call_arg = mock_pipe.call_args[0][0]
    assert call_arg["sampling_rate"] == SAMPLE_RATE
