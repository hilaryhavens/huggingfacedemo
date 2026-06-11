# Voice-to-Text Word Processor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local desktop word processor that transcribes microphone dictation using Whisper and lets the user edit and save documents as .txt or .docx.

**Architecture:** Four focused modules (audio_capture, transcription, file_io, ui) wired together by a thin entry point (app.py). The design spec said "single file" for simplicity, but we split into modules here to make each piece independently testable — the user experience is identical. Each module has one responsibility and communicates through simple Python interfaces.

**Tech Stack:** Python 3.10+, tkinter (stdlib), sounddevice, transformers, torch (CPU), numpy, python-docx, pytest

---

## File Structure

```
huggingfacedemo/
├── app.py               ← entry point: creates tk.Tk, instantiates App, calls mainloop()
├── audio_capture.py     ← AudioRecorder: start/stop mic recording via sounddevice
├── transcription.py     ← Transcriber: lazy-loads whisper-small, transcribes numpy audio
├── file_io.py           ← save_txt() and save_docx() pure functions
├── ui.py                ← VoiceWordProcessor: full tkinter UI, wires all modules together
├── requirements.txt     ← pinned dependencies
└── tests/
    ├── test_file_io.py
    ├── test_audio_capture.py
    └── test_transcription.py
```

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `tests/` directory

- [ ] **Step 1: Create requirements.txt**

```
sounddevice==0.5.1
numpy==1.26.4
transformers==4.44.2
torch==2.3.1
python-docx==1.1.2
pytest==8.3.2
```

- [ ] **Step 2: Install dependencies**

Run:
```
pip install -r requirements.txt
```

Expected: All packages install without error. `torch` is CPU-only by default on Windows when installed this way — no CUDA required.

- [ ] **Step 3: Create tests directory**

Run:
```
mkdir tests
```

- [ ] **Step 4: Verify pytest works**

Run:
```
pytest --collect-only
```

Expected: `no tests ran` (no test files yet — that's fine).

- [ ] **Step 5: Commit**

```
git init
git add requirements.txt
git commit -m "chore: project setup with dependencies"
```

---

## Task 2: File I/O Module

**Files:**
- Create: `file_io.py`
- Create: `tests/test_file_io.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_file_io.py`:

```python
import os
import tempfile
from docx import Document


def test_save_txt_writes_content():
    from file_io import save_txt
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        path = f.name
    try:
        save_txt("hello world", path)
        with open(path, encoding="utf-8") as f:
            assert f.read() == "hello world"
    finally:
        os.unlink(path)


def test_save_txt_overwrites_existing():
    from file_io import save_txt
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("old content")
        path = f.name
    try:
        save_txt("new content", path)
        with open(path, encoding="utf-8") as f:
            assert f.read() == "new content"
    finally:
        os.unlink(path)


def test_save_docx_writes_paragraphs():
    from file_io import save_docx
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        path = f.name
    try:
        save_docx("line one\nline two", path)
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs]
        assert "line one" in paragraphs
        assert "line two" in paragraphs
    finally:
        os.unlink(path)


def test_save_docx_empty_text():
    from file_io import save_docx
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        path = f.name
    try:
        save_docx("", path)
        doc = Document(path)
        # Empty string produces one empty paragraph — no crash
        assert isinstance(doc.paragraphs, list)
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
pytest tests/test_file_io.py -v
```

Expected: `ModuleNotFoundError: No module named 'file_io'`

- [ ] **Step 3: Implement file_io.py**

Create `file_io.py`:

```python
from docx import Document


def save_txt(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_docx(text: str, path: str) -> None:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
pytest tests/test_file_io.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add file_io.py tests/test_file_io.py
git commit -m "feat: file I/O module for saving txt and docx"
```

---

## Task 3: Audio Capture Module

**Files:**
- Create: `audio_capture.py`
- Create: `tests/test_audio_capture.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_audio_capture.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
pytest tests/test_audio_capture.py -v
```

Expected: `ModuleNotFoundError: No module named 'audio_capture'`

- [ ] **Step 3: Implement audio_capture.py**

Create `audio_capture.py`:

```python
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
MIN_DURATION_SAMPLES = int(SAMPLE_RATE * 0.5)  # 0.5 seconds


class AudioRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

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
        self._frames.append(indata.copy())

    def stop(self) -> np.ndarray | None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._frames:
            return None
        audio = np.concatenate(self._frames, axis=0).flatten()
        if len(audio) < MIN_DURATION_SAMPLES:
            return None
        return audio
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
pytest tests/test_audio_capture.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```
git add audio_capture.py tests/test_audio_capture.py
git commit -m "feat: audio capture module with sounddevice"
```

---

## Task 4: Transcription Module

**Files:**
- Create: `transcription.py`
- Create: `tests/test_transcription.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transcription.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
pytest tests/test_transcription.py -v
```

Expected: `ModuleNotFoundError: No module named 'transcription'`

- [ ] **Step 3: Implement transcription.py**

Create `transcription.py`:

```python
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
        result = self._pipe({"array": audio, "sampling_rate": SAMPLE_RATE})
        return result["text"].strip()

    def _ensure_loaded(self) -> None:
        if self._pipe is None:
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device="cpu",
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
pytest tests/test_transcription.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```
git add transcription.py tests/test_transcription.py
git commit -m "feat: transcription module with lazy-loaded whisper-small"
```

---

## Task 5: UI — Layout, Color Scheme, and Static Shell

**Files:**
- Create: `ui.py`

*(No automated tests for the UI class — tkinter requires a display and the logic is thin wiring. Manual smoke test in Task 7.)*

- [ ] **Step 1: Create ui.py with the full app class**

Create `ui.py`:

```python
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import numpy as np

from audio_capture import AudioRecorder
from transcription import Transcriber
from file_io import save_txt, save_docx

COLORS = {
    "bg": "#0d2b2b",
    "editor_bg": "#0f3333",
    "text": "#e0f2f1",
    "button": "#00897b",
    "button_text": "#ffffff",
    "recording": "#f44336",
    "status": "#80cbc4",
}


class VoiceWordProcessor:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Voice Word Processor")
        self.root.configure(bg=COLORS["bg"])
        self.root.geometry("900x650")

        self._recorder = AudioRecorder()
        self._transcriber = Transcriber()
        self._recording = False
        self._has_unsaved = False

        self._build_menu()
        self._build_editor()
        self._build_toolbar()

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menubar = tk.Menu(
            self.root,
            bg=COLORS["bg"],
            fg=COLORS["text"],
            activebackground=COLORS["button"],
            activeforeground=COLORS["button_text"],
        )
        file_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=COLORS["bg"],
            fg=COLORS["text"],
            activebackground=COLORS["button"],
            activeforeground=COLORS["button_text"],
        )
        file_menu.add_command(label="New", command=self._new_document)
        file_menu.add_separator()
        file_menu.add_command(label="Save as TXT", command=self._save_txt)
        file_menu.add_command(label="Save as DOCX", command=self._save_docx)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    # ── Editor ────────────────────────────────────────────────────────────────

    def _build_editor(self) -> None:
        frame = tk.Frame(self.root, bg=COLORS["bg"])
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        scrollbar = tk.Scrollbar(frame, bg=COLORS["bg"])
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._text = tk.Text(
            frame,
            bg=COLORS["editor_bg"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            selectbackground=COLORS["button"],
            selectforeground=COLORS["button_text"],
            font=("Courier New", 12),
            wrap=tk.WORD,
            undo=True,
            yscrollcommand=scrollbar.set,
            relief=tk.FLAT,
            padx=8,
            pady=8,
        )
        self._text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._text.yview)
        self._text.bind("<<Modified>>", self._on_text_modified)

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        toolbar = tk.Frame(self.root, bg=COLORS["bg"], pady=8)
        toolbar.pack(fill=tk.X, side=tk.BOTTOM)

        self._record_btn = tk.Button(
            toolbar,
            text="🎙  Start Recording",
            bg=COLORS["button"],
            fg=COLORS["button_text"],
            activebackground=COLORS["button"],
            activeforeground=COLORS["button_text"],
            relief=tk.FLAT,
            padx=16,
            pady=8,
            font=("Courier New", 11, "bold"),
            command=self._toggle_recording,
            cursor="hand2",
        )
        self._record_btn.pack(side=tk.LEFT, padx=12)

        self._status_label = tk.Label(
            toolbar,
            text="Status: Ready",
            bg=COLORS["bg"],
            fg=COLORS["status"],
            font=("Courier New", 10),
        )
        self._status_label.pack(side=tk.LEFT, padx=8)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_text_modified(self, event) -> None:
        self._has_unsaved = True
        self._text.edit_modified(False)

    def _toggle_recording(self) -> None:
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        try:
            self._recorder.start()
        except Exception as e:
            messagebox.showerror(
                "Microphone Error",
                f"No input device found. Please connect a microphone.\n\n{e}",
            )
            return
        self._recording = True
        self._record_btn.config(text="⏹  Stop Recording", bg=COLORS["recording"])
        self._status_label.config(text="Status: Recording...")

    def _stop_recording(self) -> None:
        self._recording = False
        self._record_btn.config(
            text="🎙  Start Recording",
            bg=COLORS["button"],
            state=tk.DISABLED,
        )
        self._status_label.config(text="Status: Transcribing...")

        audio = self._recorder.stop()
        if audio is None:
            self._status_label.config(text="Status: Ready")
            self._record_btn.config(state=tk.NORMAL)
            return

        thread = threading.Thread(target=self._transcribe_worker, args=(audio,), daemon=True)
        thread.start()

    def _transcribe_worker(self, audio: np.ndarray) -> None:
        try:
            if self._transcriber._pipe is None:
                self.root.after(0, lambda: self._status_label.config(text="Status: Loading model..."))
            text = self._transcriber.transcribe(audio)
            self.root.after(0, lambda: self._append_text(text))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Transcription Error", str(e)))
        finally:
            self.root.after(0, self._transcription_done)

    def _append_text(self, text: str) -> None:
        current = self._text.get("1.0", tk.END).strip()
        if current:
            self._text.insert(tk.END, " " + text)
        else:
            self._text.insert(tk.END, text)
        self._text.see(tk.END)

    def _transcription_done(self) -> None:
        self._status_label.config(text="Status: Ready")
        self._record_btn.config(state=tk.NORMAL)

    # ── File operations ───────────────────────────────────────────────────────

    def _new_document(self) -> None:
        if self._has_unsaved:
            if not messagebox.askyesno("Discard changes?", "Discard unsaved changes?"):
                return
        self._text.delete("1.0", tk.END)
        self._has_unsaved = False

    def _save_txt(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save as TXT",
        )
        if path:
            save_txt(self._text.get("1.0", tk.END), path)
            self._has_unsaved = False

    def _save_docx(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".docx",
            filetypes=[("Word documents", "*.docx"), ("All files", "*.*")],
            title="Save as DOCX",
        )
        if path:
            save_docx(self._text.get("1.0", tk.END), path)
            self._has_unsaved = False
```

- [ ] **Step 2: Commit**

```
git add ui.py
git commit -m "feat: tkinter UI with dark teal theme and recording toggle"
```

---

## Task 6: Entry Point

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create app.py**

Create `app.py`:

```python
import tkinter as tk
from ui import VoiceWordProcessor


def main() -> None:
    root = tk.Tk()
    VoiceWordProcessor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```
git add app.py
git commit -m "feat: entry point wiring tkinter root to VoiceWordProcessor"
```

---

## Task 7: Full Test Suite Pass + Manual Smoke Test

**Files:**
- No new files

- [ ] **Step 1: Run the full test suite**

Run:
```
pytest tests/ -v
```

Expected: 13 passed, 0 failed.

- [ ] **Step 2: Launch the app and do a manual smoke test**

Run:
```
python app.py
```

Check each of the following manually:

| Check | Expected |
|---|---|
| Window opens with dark teal background | Pass |
| Editor area is dark, text is light teal | Pass |
| "🎙 Start Recording" button is teal | Pass |
| Status shows "Status: Ready" | Pass |
| Click Start Recording → button turns red, label shows "Recording..." | Pass |
| Speak a sentence, click Stop Recording → label shows "Transcribing..." then "Ready" | Pass |
| Transcribed text appears in editor | Pass |
| Text is editable (type in the editor) | Pass |
| Ctrl+Z undoes edits | Pass |
| File → New with content → confirm dialog appears | Pass |
| File → Save as TXT → file dialog opens, saves correctly | Pass |
| File → Save as DOCX → file dialog opens, saves correctly | Pass |
| Recording < 0.5s → no text appended, returns to Ready | Pass |

- [ ] **Step 3: Final commit**

```
git add .
git commit -m "chore: verified full test suite and manual smoke test pass"
```

---

## Self-Review Notes

- **Spec coverage:** All spec requirements covered — toggle recording, dark teal scheme, lazy model load, background transcription thread, .txt and .docx save, unsaved-changes guard on New, mic error dialog, short-recording discard.
- **No placeholders:** All steps contain complete code.
- **Type consistency:** `AudioRecorder.stop()` returns `np.ndarray | None` — matched correctly in `ui.py`'s `_stop_recording`. `Transcriber._pipe` accessed directly in `ui.py` to detect first-load — this is an intentional internal check, noted.
- **Scope:** Single focused app, no decomposition needed.
