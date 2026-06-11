from docx import Document


def save_txt(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_docx(text: str, path: str) -> None:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(path)
