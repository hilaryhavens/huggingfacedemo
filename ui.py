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
        self._chunk_timer_id: str | None = None

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
        self._schedule_chunk()

    def _schedule_chunk(self) -> None:
        self._chunk_timer_id = self.root.after(3000, self._chunk_transcribe)

    def _chunk_transcribe(self) -> None:
        audio = self._recorder.drain()
        if audio is not None:
            thread = threading.Thread(
                target=self._chunk_worker, args=(audio,), daemon=True
            )
            thread.start()
        if self._recording:
            self._schedule_chunk()

    def _chunk_worker(self, audio: np.ndarray) -> None:
        try:
            loading = self._transcriber._pipe is None
            if loading:
                self.root.after(0, lambda: self._status_label.config(text="Status: Loading model..."))
            text = self._transcriber.transcribe(audio)
            if loading:
                self.root.after(0, lambda: self._status_label.config(text="Status: Recording..."))
            if text:
                self.root.after(0, lambda t=text: self._append_text(t))
        except Exception:
            pass

    def _stop_recording(self) -> None:
        if self._chunk_timer_id is not None:
            self.root.after_cancel(self._chunk_timer_id)
            self._chunk_timer_id = None
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
