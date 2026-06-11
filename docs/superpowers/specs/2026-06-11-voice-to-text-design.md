# Voice-to-Text Word Processor — Design Spec

**Date:** 2026-06-11
**Status:** Approved

---

## Overview

A local desktop word processor application that lets the user dictate long documents using their microphone. Speech is transcribed using OpenAI Whisper (small model) and appended to an editable text area. Documents can be saved as `.txt` or `.docx`. The app runs entirely on CPU — no GPU required.

---

## Architecture

Single Python file (`app.py`) with four logical components:

### 1. Audio Capture
- Library: `sounddevice`
- Toggle-based recording: one click to start, one click to stop
- Audio is collected into a numpy array buffer while recording is active
- On stop, the buffer is passed to the transcription component

### 2. Transcription
- Model: `openai/whisper-small` via `transformers` pipeline
- Model is loaded lazily on first recording stop (not at startup)
- Status label shows `Loading model...` during initial load (cached locally after)
- Transcription runs on a background thread to keep UI responsive
- Minimum audio threshold: 0.5 seconds — shorter recordings are silently discarded

### 3. Text Editor
- Widget: `tkinter.Text`
- Scrollable, editable, word-wrap enabled, monospaced font, undo enabled (`undo=True`)
- Transcribed text is appended at end of document with a space separator
- If document is empty, text starts at the beginning
- User can freely type, edit, cut, copy, and paste at any time

### 4. File I/O
- Save as `.txt`: Python built-in `open()` with native file-save dialog
- Save as `.docx`: `python-docx` library with native file-save dialog
- "New" clears the editor; if unsaved content exists, a confirm dialog asks first

---

## UI Layout

```
┌─────────────────────────────────────────┐
│  File                                   │  ← menu bar
├─────────────────────────────────────────┤
│                                         │
│                                         │
│         [Text editor area]              │
│                                         │
│                                         │
├─────────────────────────────────────────┤
│  🎙 Start Recording    Status: Ready    │  ← bottom toolbar
└─────────────────────────────────────────┘
```

### Menu Bar
- **File → New**: clears editor (with unsaved-changes confirmation if needed)
- **File → Save as TXT**: opens native save dialog, saves `.txt`
- **File → Save as DOCX**: opens native save dialog, saves `.docx`

### Bottom Toolbar
- **Toggle button**: label switches between `🎙 Start Recording` and `⏹ Stop Recording`
- Button turns red while recording is active
- **Status label**: displays one of: `Ready` / `Recording...` / `Loading model...` / `Transcribing...`

---

## Color Scheme — Dark Teal

| Element | Color |
|---|---|
| Window / frame background | `#0d2b2b` |
| Text editor background | `#0f3333` |
| Editor text | `#e0f2f1` |
| Buttons (idle) | `#00897b` |
| Button text | `#ffffff` |
| Recording active button | `#f44336` |
| Status label text | `#80cbc4` |
| Menu bar background | `#0d2b2b` |
| Menu bar text | `#e0f2f1` |

---

## Error Handling

| Scenario | Behavior |
|---|---|
| No microphone detected | `messagebox.showerror` dialog: "No input device found. Please connect a microphone." |
| Recording < 0.5 seconds | Silently discarded, status returns to `Ready` |
| Unsaved content on New | Confirm dialog: "Discard unsaved changes?" — Cancel keeps content |
| Transcription in progress | Button disabled until transcription completes |

---

## Dependencies

| Package | Purpose |
|---|---|
| `transformers` | Whisper model pipeline |
| `torch` | Required by transformers (CPU build) |
| `sounddevice` | Microphone capture |
| `numpy` | Audio buffer handling |
| `python-docx` | `.docx` file export |

All standard library (`tkinter`, `threading`, `tkinter.filedialog`, `tkinter.messagebox`) — no additional installs needed for UI.

---

## Data Flow

```
Mic → sounddevice buffer
           ↓
    (stop recording)
           ↓
    whisper-small pipeline  ←  background thread
           ↓
      text string
           ↓
    append to Text widget
           ↓
    File → Save as TXT / DOCX
```

---

## Out of Scope

- Real-time streaming transcription (transcription happens after stop, not word-by-word)
- Speaker diarization
- Punctuation correction / post-processing
- Cloud/API-based transcription
- Undo history beyond tkinter's built-in undo
