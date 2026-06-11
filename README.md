# Voice Word Processor

A desktop word processor that transcribes speech to text in real time using [OpenAI Whisper](https://huggingface.co/openai/whisper-small) (whisper-small) running locally via Hugging Face Transformers.

![dark teal UI with recording toggle and text editor]

## Features

- **Real-time transcription** — text appears in the editor every ~3 seconds as you speak
- **Local model** — whisper-small runs on CPU, no API key or internet connection required after first download
- **Dark teal theme** — easy on the eyes for long sessions
- **Save as TXT or DOCX** — File menu export to plain text or Word document
- **Undo support** — standard Ctrl+Z undo in the editor

## Requirements

- Python 3.12+
- A microphone

## Installation

```bash
pip install -r requirements.txt
```

The whisper-small model (~460 MB) downloads automatically from Hugging Face on first use.

## Usage

```bash
python app.py
```

1. Click **Start Recording** to begin — the button turns red
2. Speak; transcribed text appears in the editor every few seconds
3. Click **Stop Recording** when done — any remaining audio is transcribed
4. Edit freely, then save via **File → Save as TXT** or **File → Save as DOCX**

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Entry point — wires tkinter root to `VoiceWordProcessor` |
| `ui.py` | Main UI class — layout, recording toggle, chunk timer |
| `audio_capture.py` | `AudioRecorder` — sounddevice stream with thread-safe frame drain |
| `transcription.py` | `Transcriber` — lazy-loaded Whisper pipeline wrapper |
| `file_io.py` | TXT and DOCX save helpers |

## Running Tests

```bash
python -m pytest -v
```

17 tests covering audio capture, transcription, and file I/O.
