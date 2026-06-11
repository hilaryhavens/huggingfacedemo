import tkinter as tk
from ui import VoiceWordProcessor


def main() -> None:
    root = tk.Tk()
    VoiceWordProcessor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
