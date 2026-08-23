"""Assemble manuscript/manuscript.md from the individual section files, with the house
title/author/word-count header (matches the companion attribution study's convention)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"

TITLE = "Reliability-Gated Fusion of a Language-Model Prior with a P300 Neural Posterior"

AUTHOR_BLOCK = """**Authors:** Alon Gorenshtein, MD^1,2^; Mahmud Omar^1,3^; Eric Jia^1,4^; Yosef Adiniaev^1^; Oved Daniel, MD^5^; Jonathan Kruskal^6^; Muneeb Ahmed^6^; Olga Brook^6^; Yiftach Barash^1,6^; Eyal Klang, MD^1,6^

**Affiliations:** ^1^ BRIDGE GenAI Lab, Beth Israel Deaconess Medical Center, Boston, MA, USA. ^2^ Department of Neurology, Beth Israel Deaconess Medical Center, Harvard Medical School, Boston, MA, USA. ^3^ Icahn School of Medicine at Mount Sinai, New York, NY, USA. ^4^ Northwestern University Feinberg School of Medicine, Chicago, IL, USA. ^5^ Neurology Division, Tel Aviv Sourasky University Medical Center, Tel Aviv, Israel. ^6^ Department of Radiology, Beth Israel Deaconess Medical Center, Harvard Medical School, Boston, MA, USA.

**Corresponding author:** Alon Gorenshtein, MD, Department of Neurology, Beth Israel Deaconess Medical Center, Harvard Medical School, Boston, MA, USA (agorensh@bidmc.harvard.edu)"""

# Nature Biomedical Engineering Article convention: main text is Introduction + Results +
# Discussion only (Methods runs after Discussion as "Online Methods" and is excluded from the
# word cap, along with the abstract, references, and figure/table legends).
MAIN_TEXT_SECTIONS = ["introduction", "results", "discussion"]
SECTION_ORDER = ["abstract", "introduction", "results", "discussion", "methods", "references"]


def word_count(path: Path, strip_heading: str | None = None) -> int:
    text = path.read_text()
    if strip_heading:
        text = text.replace(strip_heading, "")
    return len(text.split())


def main():
    abstract_words = word_count(MS / "abstract.md", "## Abstract")
    main_text_words = sum(word_count(MS / f"{name}.md") for name in MAIN_TEXT_SECTIONS)
    methods_words = word_count(MS / "methods.md")

    parts = [
        f"# {TITLE}",
        "",
        AUTHOR_BLOCK,
        "",
        f"**Word counts:** main text {main_text_words:,} (of 3,500 cap, excludes Methods per "
        f"Nature Biomedical Engineering Article convention); Methods {methods_words:,} "
        f"(excluded from cap); abstract {abstract_words} (of 175 cap)",
        "",
        "---",
        "",
    ]
    for name in SECTION_ORDER:
        section_text = (MS / f"{name}.md").read_text().strip()
        parts.append(section_text)
        parts.append("")

    (MS / "manuscript.md").write_text("\n".join(parts))
    print(f"Wrote manuscript.md: main text {main_text_words} words (methods {methods_words} excluded), abstract {abstract_words} words")


if __name__ == "__main__":
    main()
