"""Build a self-contained HTML gallery for HGE targeting diagnostics."""

from __future__ import annotations

import html
from pathlib import Path


SECTIONS = [
    (
        "Input catalogs",
        "On-sky coverage",
        "Source-density maps for the three input catalogs.",
        [
            ("tmass_skydensity", "2MASS", "Near-infrared source density"),
            ("virac2_skydensity", "VIRAC2", "Deep VVV astrometric catalog"),
            ("apoglimpse_skydensity", "APOGLIMPSE", "Mid-infrared source density"),
        ],
    ),
    (
        "Photometry",
        "Input color–magnitude diagrams",
        "Catalog photometry before merging and dereddening.",
        [
            ("tmass_cmd", "2MASS CMD", "H versus J − Ks"),
            ("virac2_cmd", "VIRAC2 CMD", "H versus J − Ks"),
            ("apoglimpse_cmd", "APOGLIMPSE CMD", "[4.5] versus [3.6] − [4.5]"),
        ],
    ),
    (
        "Catalog construction",
        "Merged and dereddened sample",
        "The 2MASS–VIRAC2 merge, RJCE dereddening, and photometric-quality cuts.",
        [
            ("tmassvirac2_cmd", "Merged NIR CMD", "2MASS-system photometry"),
            ("tmassvirac2_deredcmd", "RJCE dereddened CMD", "All matched sources"),
            ("tmassvirac2_deredcmd_qacuts", "After quality cuts", "Good NIR and GLIMPSE photometry"),
        ],
    ),
    (
        "Target selection",
        "RGB selection and neighbor rejection",
        "The RGB selection polygon and targets rejected because of contaminating neighbors.",
        [
            ("alltargets_deredcmd", "RGB selection", "Selection polygon and candidates"),
            ("targets_cmd_brtnei", "Bright neighbors", "Targets exceeding the contamination threshold"),
        ],
    ),
    (
        "Final catalog",
        "Selected HGE targets",
        "The final sample after magnitude, neighbor, and uniform-CMD prioritization.",
        [
            ("targets_cmd", "Observed CMD", "Final targets in observed color–magnitude space"),
            ("targets_deredcmd", "Dereddened CMD", "Final targets in intrinsic color–magnitude space"),
        ],
    ),
]


CSS = """
:root{--ink:#172126;--muted:#657177;--paper:#f4f1ea;--card:#fffdf8;--line:#d8d4ca;--red:#d94c3d;--blue:#214d63}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Arial,sans-serif}
header{padding:4.5rem 5vw;color:#f8f4e9;background:radial-gradient(circle at 84% 22%,rgba(217,76,61,.28),transparent 25%),linear-gradient(135deg,#122833,#214d63 56%,#183441)}
.kicker,.eyebrow{margin:0 0 1rem;color:#f7b2a8;font-size:.72rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase}
h1{max-width:1000px;margin:0;font:400 clamp(3rem,7vw,7rem)/.96 Georgia,serif;letter-spacing:-.05em}header p:last-child{max-width:720px;margin:2rem 0 0;color:#d9e2e5;line-height:1.6}
main{width:min(1440px,90vw);margin:auto}section{padding:6rem 0;border-bottom:1px solid var(--line)}.heading{max-width:680px;margin-bottom:2.5rem}.eyebrow{color:var(--red)}
h2{margin:0 0 1rem;font:400 clamp(2.2rem,4vw,4rem)/1 Georgia,serif;letter-spacing:-.035em}.heading>p:last-child{color:var(--muted);line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1.4rem}.grid.two{grid-template-columns:repeat(2,minmax(0,1fr))}
figure{min-width:0;margin:0}.frame{display:block;overflow:hidden;border:1px solid var(--line);background:white;cursor:zoom-in}.frame img{display:block;width:100%;aspect-ratio:1.15;object-fit:contain;transition:transform .3s}.frame:hover img{transform:scale(1.015)}
figcaption{padding:1rem .1rem}figcaption strong{display:block;font-size:.95rem}figcaption span{display:block;margin-top:.35rem;color:var(--muted);font-size:.76rem}
footer{padding:2.5rem 5vw;color:var(--muted);font-size:.72rem}.lightbox{display:none;position:fixed;z-index:10;inset:0;place-items:center;padding:3rem;background:rgba(10,18,22,.94)}.lightbox.open{display:grid}.lightbox img{max-width:92vw;max-height:88vh;background:white}.close{position:fixed;top:1rem;right:1.5rem;border:0;background:none;color:white;font-size:2.2rem;cursor:pointer}
@media(max-width:850px){.grid,.grid.two{grid-template-columns:1fr 1fr}}@media(max-width:560px){header{padding:4rem 1.25rem}main{width:calc(100% - 2.5rem)}section{padding:4rem 0}.grid,.grid.two{grid-template-columns:1fr}}
"""


JS = """
const box=document.querySelector('.lightbox'), full=box.querySelector('img');
document.querySelectorAll('.frame').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();full.src=a.href;full.alt=a.querySelector('img').alt;box.classList.add('open')}));
function closeBox(){box.classList.remove('open');full.src=''}
box.addEventListener('click',closeBox);document.querySelector('.close').addEventListener('click',closeBox);
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeBox()});
"""


def make_diagnostic_webpage(
    tag: str,
    *,
    output: str | Path | None = None,
    title: str = "HGE Test Field Diagnostics",
    field_label: str = "ℓ = 340°, b = 0°",
) -> Path:
    """Write an HTML gallery beside the PNG diagnostic plots.

    Parameters
    ----------
    tag
        Filename prefix used for the plots, e.g. ``hge_l340_b0.0_rad1.05``.
    output
        Output HTML filename. Defaults to ``{tag}_diagnostics.html``.
    title, field_label
        Text displayed in the page header.
    """
    output_path = Path(output or f"{tag}_diagnostics.html")
    cards = []
    number = 0
    for eyebrow, heading, description, plots in SECTIONS:
        number += 1
        figures = []
        for suffix, plot_title, caption in plots:
            filename = f"{tag}_{suffix}.png"
            if not Path(filename).is_file():
                print(f"WARNING: diagnostic plot not found: {filename}")
            figures.append(
                f'<figure><a class="frame" href="{html.escape(filename)}">'
                f'<img src="{html.escape(filename)}" loading="lazy" '
                f'alt="{html.escape(plot_title)}"></a><figcaption>'
                f'<strong>{html.escape(plot_title)}</strong>'
                f'<span>{html.escape(caption)}</span></figcaption></figure>'
            )
        grid_class = "grid two" if len(plots) == 2 else "grid"
        cards.append(
            f'<section><div class="heading"><p class="eyebrow">{number:02d} · '
            f'{html.escape(eyebrow)}</p><h2>{html.escape(heading)}</h2>'
            f'<p>{html.escape(description)}</p></div>'
            f'<div class="{grid_class}">{"".join(figures)}</div></section>'
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><header><p class="kicker">Hidden Galaxy Explorer · Test field diagnostics</p>
<h1>Target selection at {html.escape(field_label)}</h1>
<p>A visual audit trail from input catalogs through the final uniformly prioritized RGB target sample.</p>
</header><main>{"".join(cards)}</main>
<footer>HGE test-field targeting · Generated from {html.escape(tag)}</footer>
<div class="lightbox" role="dialog" aria-modal="true"><button class="close" aria-label="Close">×</button><img alt=""></div>
<script>{JS}</script></body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    make_diagnostic_webpage("hge_l340_b0.0_rad1.05")
