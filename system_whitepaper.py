from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _paginate_markdown(text: str, max_chars: int = 3200) -> list[str]:
    cleaned = [line.rstrip() for line in text.splitlines()]
    pages = []
    buf = []
    size = 0
    for line in cleaned:
        line_size = len(line) + 1
        if size + line_size > max_chars and buf:
            pages.append("\n".join(buf).strip())
            buf = []
            size = 0
        buf.append(line)
        size += line_size
    if buf:
        pages.append("\n".join(buf).strip())
    return pages or [text]


def generate_system_whitepaper(source_path: Path | str,
                               output_path: Path | str) -> Path:
    source = Path(source_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    text = source.read_text(encoding="utf-8")
    pages = _paginate_markdown(text)

    with PdfPages(output) as pdf:
        for i, page_text in enumerate(pages, start=1):
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
            ax.axis("off")
            wrapped = "\n".join(
                textwrap.fill(line, width=95, break_long_words=False)
                for line in page_text.splitlines()
            )
            ax.text(
                0.0,
                1.0,
                wrapped,
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                color="#111111",
            )
            fig.suptitle(
                f"Quant Analyzer System Whitepaper — Page {i}",
                fontsize=12,
                fontweight="bold",
                y=0.985,
            )
            pdf.savefig(fig)
            plt.close(fig)
    return output


__all__ = ["generate_system_whitepaper"]
