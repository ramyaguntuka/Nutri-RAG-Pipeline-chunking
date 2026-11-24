"""Load and split the local nutrition notes into titled, section-aware chunks."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from src.config import DATA_DIR


@dataclass(frozen=True)
class DocumentChunk:
    document: str
    section: str
    text: str

    @property
    def label(self) -> str:
        return f"{self.document} — {self.section}"


def _read_sections(path: Path) -> Iterable[tuple[str, str]]:
    content = path.read_text(encoding="utf-8")
    current_section = "Overview"
    paragraphs: list[str] = []

    for line in content.splitlines():
        heading = re.match(r"^#{1,3}\s+(.+?)\s*$", line)
        if heading:
            if paragraphs:
                yield current_section, "\n".join(paragraphs).strip()
                paragraphs = []
            current_section = heading.group(1).strip()
        elif line.strip():
            paragraphs.append(line.strip())
        elif paragraphs and paragraphs[-1] != "":
            paragraphs.append("")

    if paragraphs:
        yield current_section, "\n".join(paragraphs).strip()


def _split_text(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - min(25, max_words // 4)
    return chunks


def load_chunks(data_dir: Path = DATA_DIR, max_words: int = 150) -> list[DocumentChunk]:
    """Load every text document and split each section into manageable chunks."""
    if max_words < 1:
        raise ValueError("max_words must be at least 1.")
    if not data_dir.exists():
        raise FileNotFoundError(f"Knowledge-base directory was not found: {data_dir}")
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt knowledge-base documents found in {data_dir}")

    chunks: list[DocumentChunk] = []
    for path in files:
        title = path.stem.replace("_", " ").title()
        for section, text in _read_sections(path):
            chunks.extend(
                DocumentChunk(title, section, part)
                for part in _split_text(text, max_words)
                if part.strip()
            )
    return chunks
