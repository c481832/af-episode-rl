"""Build rl_framework_paper_v6.pdf from rl_framework_paper_v6.md.

Run: ../.venv/bin/python scripts/build_paper_pdf.py

Math is written in the markdown as $...$ / $$...$$ LaTeX; KaTeX (loaded from
the jsDelivr CDN, so the build needs network) typesets it and headless Chrome
prints the page. Figures and tables are the PNGs in plots/, placed after the
paragraph that cites them as (`figure_xx_name`) or (`table_xx_name`). The
render_*.py scripts next to this one write those PNGs.

Needs: pypandoc_binary (uv pip install pypandoc_binary), google-chrome, and
pdffonts (poppler-utils) for the final check that the math was typeset.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "rl_framework_paper_v6.md"
PLOTS = ROOT / "plots"
OUT_PDF = ROOT / "rl_framework_paper_v6.pdf"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

# pandoc --katex leaves the LaTeX source in span.math; this renders it when the page loads.
KATEX_HEAD = f"""<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script src="{KATEX}/katex.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function () {{
  for (const span of document.querySelectorAll("span.math")) {{
    katex.render(span.textContent, span, {{ displayMode: span.classList.contains("display"), throwOnError: true }});
  }}
}});
</script>"""

# Captions are markdown, so they can carry $...$ math.
CAPTIONS = {
    "figure_01_pipeline_comparison": "Figure 1. The two decision stages on the same frozen-detector output $p(t)$ and the same test window (SHDB-AF). (a) Tuned post-processing: fixed rules applied feed-forward, with $\\theta$ and $g$ the same for every second and recording, and the 30-second rule applied as a filter afterwards. Here it adds a false episode and splits the onset across a gap longer than $g$. (b) The controller: a learned policy whose decision at each second $t$ uses its own past actions and the next 31 s of $p$ (shaded: the five forward-context buckets), with the 30-second rule enforced inside the loop as an exit mask. The zoom shows $t$ at the dip where post-processing splits the episode: $p$ is below 0.5, but the controller is already in AF and $p$ rises again within the forward context; it stays in AF. The window was chosen to illustrate the episode-formation errors of §2.3; on most isolated reference episodes the two methods agree.",
    "figure_02_rl_pipeline": "Figure 6. Phase 2 in detail: the eight-value observation, the policy and value networks, the exit mask, and PPO training with the three-term reward.",
    "figure_03_boundary_matching_examples": r"Figure 7. Boundary-matching examples under tolerance $\delta = 5$ s (one-to-one, fragmentation, merging, and missed/false episodes).",
    "figure_04_boundary_tolerance_sweep": "Figure 2. Boundary metrics against matching tolerance on the test set (ECGFounder). (a–c) Boundary F1, Se, and +P over all test sources, each the mean of the onset and offset values; (d–f) Boundary F1 per test source.",
    "figure_05_boundary_error_attribution": r"Figure 3. Boundary errors at $\delta = 5$ s attributed to the overlap groups of §4.6 (ECGFounder; onsets and offsets pooled).",
    "figure_06_sensitivity_and_ablation": "Figure 4. Controller robustness on the ECGFounder test set. (a) One-at-a-time hyperparameter sweep around the reported configuration; (b) feature ablation, one observation group removed at a time. Both are compared with the sweep re-run of the reported configuration.",
    "figure_07_policy_behavior_last_state": "Figure 5. Controller action probability vs. current detector probability, conditioned on the preceding action.",
    "table_01_test_cohort": "Table 2. Description of held-out test data.",
    "table_02_method_comparison": "Table 1. Method comparison on the test set for both detectors.",
    "table_03_episode_level": "Table 3. Episode-level metrics (ECGFounder).",
}

# Assets the text cites without a (`name`) marker: placed after the paragraph containing the anchor.
ANCHORED = {}

MARKER = re.compile(r" ?\(`((?:figure|table)_[a-z0-9_]+)`\)")

CSS = """
@page { size: letter; margin: 0.75in; }
html { font-size: 11pt; }
body {
  font-family: "Times New Roman", Times, "Liberation Serif", serif;
  line-height: 1.45;
  color: #111;
  max-width: 7.0in;
  margin: 0 auto;
}
h1 { font-size: 16pt; text-align: center; margin: 0 0 0.8em; line-height: 1.25; }
h2 { font-size: 13pt; margin: 1.4em 0 0.55em; border-bottom: 1px solid #ccc; padding-bottom: 0.15em; }
h3 { font-size: 11.5pt; margin: 1.1em 0 0.4em; }
p { margin: 0.55em 0; text-align: justify; hyphens: auto; }
li { margin: 0.25em 0; text-align: justify; }
ol[type="a"] { list-style: none; counter-reset: case; }
ol[type="a"] > li { counter-increment: case; }
ol[type="a"] > li::before { content: "(" counter(case, lower-alpha) ") "; }
strong { font-weight: 700; }
code { font-family: "Courier New", monospace; font-size: 0.92em; }
body .katex { font-size: 1em; }
figure { margin: 1.1em auto; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
figcaption { font-size: 9.5pt; margin-top: 0.45em; text-align: left; line-height: 1.35; }
.footnotes { font-size: 9.5pt; }
table { border-collapse: collapse; margin: 1.1em auto; font-size: 9.5pt; page-break-inside: avoid; }
th, td { padding: 0.25em 0.6em; }
thead tr { border-top: 1.5px solid #111; border-bottom: 1px solid #111; }
tbody tr:last-child { border-bottom: 1.5px solid #111; }
caption { caption-side: bottom; font-size: 9.5pt; margin-top: 0.45em; text-align: left; line-height: 1.35; }
"""


def figure_markdown(stem):
    png = PLOTS / f"{stem}.png"
    if not png.exists():
        raise FileNotFoundError(png)
    return f"![{CAPTIONS[stem]}]({png.as_uri()}){{#{stem}}}"


def place_assets(md):
    """Drop the (`name`) markers and put each figure after the paragraph citing it."""
    placed = []
    blocks = []
    for block in md.split("\n\n"):
        stems = MARKER.findall(block)
        stems += [stem for stem, anchor in ANCHORED.items() if anchor in block]
        blocks.append(MARKER.sub("", block))
        blocks += [figure_markdown(stem) for stem in stems]
        placed += stems

    missing = set(CAPTIONS) - set(placed)
    if missing:
        raise RuntimeError(f"assets never cited in the markdown: {sorted(missing)}")
    if len(placed) != len(set(placed)):
        raise RuntimeError(f"assets cited more than once: {placed}")
    return "\n\n".join(blocks)


def main():
    md = place_assets(MD_PATH.read_text(encoding="utf-8"))
    body = pypandoc.convert_text(md, "html5", format="markdown", extra_args=["--katex"])
    page = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>From ECG Window Predictions to AF Episodes</title>\n<style>{CSS}</style>\n{KATEX_HEAD}\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "paper.html"
        html_path.write_text(page, encoding="utf-8")
        subprocess.run(
            [
                "google-chrome",
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={OUT_PDF}",
                html_path.as_uri(),
            ],
            check=True,
        )
    fonts = subprocess.run(["pdffonts", OUT_PDF], capture_output=True, text=True, check=True).stdout
    if "KaTeX" not in fonts:
        raise RuntimeError("KaTeX did not load, so the PDF contains raw LaTeX; check the network and rebuild")
    print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
