"""Download the evaluation corpus: 14 sections of *Dive into Deep Learning*.

    python -m eval.fetch_corpus

The text is CC BY-SA 4.0 (Zhang, Lipton, Li & Smola, https://d2l.ai), so it is
downloaded at a pinned commit and verified by checksum rather than vendored into this
MIT-licensed repository. Code blocks, framework tabs, exercises and Sphinx markup are
stripped. What remains reads like the textbook chapters a student would upload.

The two short lecture notes in ``eval/corpus/`` (written for this project) are kept as
well, so the index mixes small hand-written notes with long, partly overlapping
textbook chapters. That overlap creates realistic distractors for retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

COMMIT = "23d7a5aecceee57d1292c56e90cce307f183bb0a"
BASE = f"https://raw.githubusercontent.com/d2l-ai/d2l-en/{COMMIT}/"
OUT_DIR = Path(__file__).parent / "corpus" / "d2l"
ATTRIBUTION = (
    "Source: Dive into Deep Learning (Zhang, Lipton, Li, Smola), https://d2l.ai, "
    f"CC BY-SA 4.0, commit {COMMIT[:7]}. Code, exercises and markup removed."
)

# path in the d2l repo -> sha256 of the raw file at COMMIT
FILES: dict[str, str] = {
    "chapter_linear-regression/generalization.md": "2090b8a2dc6a7b9d8d6f42f0aeaaad9ba76c59680ada6fb6ecef8160ac5122d6",
    "chapter_linear-regression/weight-decay.md": "081edd89fc50afb5049df925e82d7a33ea485f4cf1ee6ed28c78f30829b5f388",
    "chapter_linear-classification/generalization-classification.md": "e525245cc636ceb978f077ad289a5a5d4e4d8e1e0ae0b47a86ed5966f63432a3",
    "chapter_linear-classification/environment-and-distribution-shift.md": "524c1ae014be3b2ba046eea1dd22cc180120dafed13285f0f5dd6f6223cb4e57",
    "chapter_multilayer-perceptrons/generalization-deep.md": "25463e3c34bcc764baf3f6f44a8ff98b638d2576be4e22761b4f3f7babe528a0",
    "chapter_multilayer-perceptrons/dropout.md": "e5fd93ac9cca9ad6a563f216df28af485ea885b07a4f8e0ce874c5045a619cd3",
    "chapter_optimization/gd.md": "c614912211219b60e4cc91b1385b92ac5eb7cac77ca26b4e8c52fa02db810f76",
    "chapter_optimization/sgd.md": "b473add6ca7ddd7c46595bfa8f456f41870301b4f1ba8a718240b79f533cf952",
    "chapter_optimization/minibatch-sgd.md": "5a7f74531bb28744080a5edc41e1435cfb5457cc5aab4bd71eb43e396acf86bb",
    "chapter_optimization/momentum.md": "10a04b83ed4fc93a204c22b54a726cd7abb4ff3ef4c1a964490f94a7cc46c991",
    "chapter_optimization/adam.md": "b028c34fa6b0271df8f76311f255a9ca1cd213a9097d53e2e7c234b557c7ccb6",
    "chapter_convolutional-modern/batch-norm.md": "a82dc20e9a07dcee5e88ff4e094720bf524aab8ace7d9a6aca47f9827b4e9a33",
    "chapter_recurrent-modern/lstm.md": "66a0dc190a4641d4e17bb14f8ad57cb066a46f1ee4f472a45a3138c588008614",
    "chapter_attention-mechanisms-and-transformers/transformer.md": "085f66181f137620d27b61eabfeefe3927258abaabdf2693a8887bb186415c82",
}


def clean(markdown: str) -> str:
    text = re.sub(r"```.*?```", "", markdown, flags=re.S)  # code (all frameworks)
    text = re.sub(r":begin_tab:`[^`]*`.*?:end_tab:", "", text, flags=re.S)  # framework notes
    text = re.split(r"\n## Exercises\b|\n\[Discussions\]", text)[0]
    text = re.sub(r"^:(label|eqlabel|width|numref|bibliography):.*$", "", text, flags=re.M)
    text = re.sub(r"^!\[.*$", "", text, flags=re.M)  # figures
    text = re.sub(r":(?:cite|citet|citep):`[^`]*`", "", text)
    text = re.sub(r":(?:numref|eqref):`[^`]*`", "the corresponding section", text)
    text = re.sub(r"\(\*\*|\*\*\)|\[\*\*|\*\*\]", "", text)  # d2l highlight markers
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Unwrap hard-wrapped paragraphs (keep headings, lists and display math on their own lines).
    text = re.sub(r"(?<![\n$])\n(?![\n#*\-$\\]|\d+\. )", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def output_name(path: str) -> str:
    return path.split("/")[-1]


def fetch(path: str) -> bytes:
    with urllib.request.urlopen(BASE + path, timeout=30) as response:  # noqa: S310 (fixed https host)
        return response.read()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--print-checksums", action="store_true", help="print sha256 of each raw file and exit")
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for path, expected in FILES.items():
        raw = fetch(path)
        digest = hashlib.sha256(raw).hexdigest()
        if args.print_checksums:
            print(f'    "{path}": "{digest}",')
            continue
        if expected and digest != expected:
            print(f"checksum mismatch for {path}: {digest}", file=sys.stderr)
            failed = True
            continue
        text = clean(raw.decode("utf-8"))
        (OUT_DIR / output_name(path)).write_text(f"<!-- {ATTRIBUTION} -->\n\n{text}", encoding="utf-8")
        print(f"{output_name(path):45s} {len(text):>7,d} chars")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
