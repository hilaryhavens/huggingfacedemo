import numpy as np
from unittest.mock import MagicMock, patch


def test_stop_with_no_frames_returns_none():
    from audio_capture import AudioRecorder
    recorder = AudioRecorder()
    recorder._stream = MagicMock()
    result = recorder.stop()
    assert result is None


def test_stop_with_short_audio_returns_none():
    from audio_capture import AudioRecorder
    recorder = AudioRecorder()
    recorder._stream = MagicMock()
    # 4000 samples < 0.5s at 16kHz (8000 minimum)
    recorder._frames = [np.zeros((4000, 1), dtype="float32")]
    result = recorder.stop()
    assert result is None


def test_stop_with_valid_audio_returns_array():
    from audio_capture import AudioRecorder
    recorder = AudioRecorder()
    recorder._stream = MagicMock()
    # 16000 samples = 1 second at 16kHz
    recorder._frames = [np.ones((16000, 1), dtype="float32")]
    result = recorder.stop()
    assert result is not None
    assert len(result) == 16000
    assert result.dtype == np.float32


def test_callback_appends_frames():
    from audio_capture import AudioRecorder
    recorder = AudioRecorder()
    chunk = np.ones((1024, 1), dtype="float32")
    recorder._callback(chunk, 1024, None, None)
    assert len(recorder._frames) == 1
    np.testing.assert_array_equal(recorder._frames[0], chunk)


def test_drain_with_no_frames_returns_none():
    from audio_capture import AudioRecorder
    recorder = AudioRecorder()
    result = recorder.drain()
    assert result is None


def test_drain_with_short_audio_returns_none():
    from audio_capture import AudioRecorder
    recorder = AudioRecorder()
    recorder._frames = [np.zeros((4000, 1), dtype="float32")]
    result = recorder.drain()
    assert result is None


def test_drain_with_valid_audio_returns_array_and_clears_frames():
    from audio_capture import AudioRecorder
    recorder = AudioRecorder()
    recorder._frames = [np.ones((16000, 1), dtype="float32")]
    result = recorder.drain()
    assert result is not None
    assert len(result) == 16000
    assert result.dtype == np.float32
    assert recorder._frames == []


def test_drain_then_stop_only_returns_frames_after_drain():
    from audio_capture import AudioRecorder
    recorder = AudioRecorder()
    recorder._stream = MagicMock()
    recorder._frames = [np.ones((16000, 1), dtype="float32")]
    recorder.drain()
    # Add new frames after drain
    recorder._frames = [np.ones((8000, 1), dtype="float32")]
    result = recorder.stop()
    assert result is not None
    assert len(result) == 8000


def test_start_raises_on_no_device():
    from audio_capture import AudioRecorder
    import sounddevice as sd
    recorder = AudioRecorder()
    with patch("sounddevice.InputStream", side_effect=sd.PortAudioError("no device")):
        try:
            recorder.start()
            raised = False
        except sd.PortAudioError:
            raised = True
    assert raised
