"""Surgical notebook markdown edits that preserve this repo's exact conventions.

Three conventions matter and all bite silently:
  * JSON is written json.dumps(indent=1, ensure_ascii=False) + "\n". Anything else
    reformats the whole file and makes the diff unreviewable.
  * Cell ids exist only at nbformat_minor >= 5. CH01 is minor 2 (no ids); CH02 is
    minor 5 (ids required). Inserting the wrong shape corrupts the notebook.
  * cell["source"] is a list of lines in every chapter notebook but a single plain
    string in TEMPLATE_CHAPTER.ipynb. Both are valid nbformat. A sweep that assumes
    a list iterates the string's characters instead, matches nothing, and reports
    success -- check isinstance(src, str) first and preserve whichever shape it has.

Code cells are never touched by anything in here.
"""
from __future__ import annotations

import json
import re
import shlex
import uuid
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path: str, nb: dict) -> None:
    Path(path).write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _as_source(text: str) -> list[str]:
    """nbformat stores source as a list of lines, each newline-terminated except the last."""
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def _new_id(nb: dict) -> str:
    taken = {c.get("id") for c in nb["cells"]}
    while True:
        cid = uuid.uuid4().hex[:8]
        if cid not in taken:
            return cid


def set_md(nb: dict, idx: int, text: str) -> None:
    """Replace the source of an existing markdown cell."""
    cell = nb["cells"][idx]
    assert cell["cell_type"] == "markdown", f"cell {idx} is {cell['cell_type']}, refusing"
    cell["source"] = _as_source(text)


def insert_md(nb: dict, idx: int, text: str) -> None:
    """Insert a new markdown cell at idx, matching the notebook's id convention."""
    cell: dict = {"cell_type": "markdown", "metadata": {}, "source": _as_source(text)}
    if nb.get("nbformat_minor", 0) >= 5:
        cell["id"] = _new_id(nb)
        cell = {"cell_type": "markdown", "id": cell["id"], "metadata": {}, "source": cell["source"]}
    nb["cells"].insert(idx, cell)


def delete_cells_containing(nb: dict, needle: str, kind: str = "code") -> int:
    """Remove every cell of `kind` whose source contains `needle`. Returns the count."""
    before = len(nb["cells"])
    nb["cells"] = [
        c for c in nb["cells"]
        if not (c["cell_type"] == kind and needle in "".join(c["source"]))
    ]
    return before - len(nb["cells"])


# Book-derived chapter summaries live in the repo root, one section per chapter.
# Only used when reformatting the 15 companion notebooks; self-contained bonus
# notebooks write their own Summary, so this file need not exist for them.
SUMMARIES = Path(__file__).resolve().parents[2] / "chapter_summaries.txt"


def book_summary(chapter_key: str) -> str:
    """Return the book's Summary for e.g. 'Ch 1' or 'Appendix C' as prose paragraphs.
    Faithful to the manuscript: the only change is normalizing the 2 banned em-dashes
    (the author's own style uses spaced en-dashes, which are kept)."""
    text = Path(SUMMARIES).read_text(encoding="utf-8")
    blocks = re.split(r'^#\s+(.+?)\s+Summary\s*$', text, flags=re.MULTILINE)
    # re.split with one capture -> [pre, key1, body1, key2, body2, ...]
    wanted = None
    for i in range(1, len(blocks), 2):
        if blocks[i].strip().lower() == chapter_key.strip().lower():
            wanted = blocks[i + 1]
            break
    if wanted is None:
        raise LookupError(f"no summary section for {chapter_key!r}")
    paras = [re.sub(r'\s+', ' ', ln).strip() for ln in wanted.split("\n") if ln.strip()]
    paras = [p.replace("—", ", ") for p in paras]   # normalize the 2 banned em-dashes only
    return "\n\n".join(paras)


def find(nb: dict, needle: str, kind: str = "markdown") -> int:
    """Index of the single cell whose source contains needle. Raises unless exactly one."""
    hits = [
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == kind and needle in "".join(c["source"])
    ]
    if len(hits) != 1:
        raise LookupError(f"{needle!r} matched {len(hits)} cells: {hits}")
    return hits[0]


def src(nb: dict, idx: int) -> str:
    return "".join(nb["cells"][idx]["source"])


# ---------------------------------------------------------------- Monograph blocks
MONO = "ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace"
SERIF = "Georgia,'Iowan Old Style','Palatino Linotype',serif"
CRIMSON, AMBER, TINT, BODY, MUTED, INK, HAIRLINE = (
    "#A20025", "#B4690E", "#FDF3F5", "#3d3436", "#6B6461", "#1A1416", "#E7E2DF")

# Book identity (retitle confirmed 2026-07-17). Cover + Colab URLs point at `main`,
# the publish branch, never the feature branch (deleted after merge -> would 404).
BOOK_TITLE = "Build AI Drug Discovery Pipelines"
BOOK_URL = "https://www.manning.com/books/build-ai-drug-discovery-pipelines"
REPO_URL = "https://github.com/nrflynn2/ml-drug-discovery"
COVER_URL = "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/assets/cover-sm.png"


def chapter_header(eyebrow: str, title: str) -> str:
    """Chapter opener: mono eyebrow + serif title + crimson rule (TEMPLATE.md §3.1).
    For a bonus notebook `eyebrow` is a topic/series label (e.g. "Bonus"), not "Chapter N"."""
    return (
        f'<div style="font-family:{MONO};font-size:11px;letter-spacing:.24em;'
        f'text-transform:uppercase;color:{CRIMSON};margin:0 0 12px;">{eyebrow}</div>\n\n'
        f'<h1 style="font-family:{SERIF};font-weight:500;font-size:33px;line-height:1.12;'
        f'margin:0 0 16px;color:{INK};letter-spacing:-.01em;">{title}</h1>\n\n'
        f'<hr style="border:0;border-top:1px solid {CRIMSON};opacity:.55;margin:0 0 4px;">'
    )


def section(n, title: str) -> str:
    """Numbered section header §N (TEMPLATE.md §3.3).

    The gutter is a literal em space inside the span, not margin-right: Colab's
    sanitizer drops layout CSS and the glyph would butt against the title.
    """
    return (f'<h2 style="font-family:{SERIF};font-weight:500;font-size:23px;margin:34px 0 10px;'
            f'color:{INK};"><span style="font-family:{MONO};font-size:13px;color:{CRIMSON};'
            f'font-weight:500;">§{n}&#8195;</span>{title}</h2>')


def backmatter(title: str) -> str:
    """Back-matter header (Summary / References): hairline rule + serif, no § (TEMPLATE.md §3.4)."""
    return (f'<hr style="border:0;border-top:1px solid {CRIMSON};opacity:.55;margin:34px 0 12px;">\n'
            f'<h2 style="font-family:{SERIF};font-weight:500;font-size:23px;margin:0 0 10px;'
            f'color:{INK};">{title}</h2>')


def subsection(title: str) -> str:
    """h3: serif, present, normal case (TEMPLATE.md §3.5)."""
    return (f'<h3 style="font-family:{SERIF};font-weight:600;font-size:17px;color:{INK};'
            f'margin:24px 0 8px;">{title}</h3>')


def subsub(title: str) -> str:
    """h4: mono, uppercase, crimson, no glyph (TEMPLATE.md §3.5)."""
    return (f'<h4 style="font-family:{MONO};font-weight:600;font-size:11.5px;letter-spacing:.1em;'
            f'text-transform:uppercase;color:{CRIMSON};margin:18px 0 6px;">{title}</h4>')


def masthead(eyebrow: str = "Companion notebook") -> str:
    """The book masthead: the first markdown cell of every notebook. Table layout +
    <img width> so it survives GitHub's style stripping. `eyebrow` labels the kind of
    notebook (bonus notebooks use "Bonus notebook")."""
    return (
        '<table style="border:none;border-collapse:collapse;margin:0 0 8px;width:100%;">\n'
        '<tr style="border:none;">\n'
        '  <td style="border:none;padding:0;vertical-align:middle;">\n'
        f'    <div style="font-family:{MONO};font-size:10.5px;letter-spacing:.2em;'
        f'text-transform:uppercase;color:{CRIMSON};margin:0 0 7px;">{eyebrow}</div>\n'
        f'    <div style="font-family:Georgia,\'Iowan Old Style\',serif;font-size:18px;'
        f'color:{INK};line-height:1.25;margin:0 0 7px;">\n'
        f'      <a href="{BOOK_URL}" style="color:{INK};text-decoration:none;'
        f'border-bottom:1px solid {CRIMSON};padding-bottom:1px;">{BOOK_TITLE}</a>\n'
        '    </div>\n'
        f'    <div style="font-family:{MONO};font-size:12px;color:{MUTED};line-height:1.7;">\n'
        '      Noah Flynn&nbsp;&middot;&nbsp;Manning Publications<br>\n'
        f'      Code&nbsp;&middot;&nbsp;<a href="{REPO_URL}" style="color:{CRIMSON};'
        'text-decoration:none;">github.com/nrflynn2/ml-drug-discovery</a>\n'
        '    </div>\n'
        '  </td>\n'
        '  <td style="border:none;padding:0 0 0 22px;vertical-align:middle;width:104px;text-align:right;">\n'
        f'    <a href="{BOOK_URL}">\n'
        f'      <img src="{COVER_URL}" width="88" alt="{BOOK_TITLE} cover" '
        f'style="border:1px solid {HAIRLINE};border-radius:3px;display:inline-block;">\n'
        '    </a>\n'
        '  </td>\n'
        '</tr>\n'
        '</table>\n'
        f'<hr style="border:0;border-top:1px solid {CRIMSON};opacity:.55;margin:0 0 22px;">'
    )


def note(n) -> str:
    """One-line opener note for a COMPANION chapter notebook (code lives here, prose in
    the book, read side by side)."""
    return f'<em style="color:{MUTED};font-size:14.5px;">Runnable code for Chapter {n}; read alongside the book.</em>'


def note_self_contained() -> str:
    """One-line opener note for a SELF-CONTAINED bonus notebook: unlike the 15 companion
    chapters, the write-up lives here too, so there is no 'read alongside the book'."""
    return (f'<em style="color:{MUTED};font-size:14.5px;">A self-contained bonus notebook: '
            'the explanation and the code both live here, and it runs top to bottom on its own.</em>')


def badge(nb_path: str) -> str:
    """Crimson Colab badge; its URL is per-notebook. nb_path is repo-relative."""
    url = f"https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/{nb_path}"
    img = "https://img.shields.io/badge/Open%20in-Colab-A20025?logo=googlecolab&logoColor=white"
    return (f'<a href="{url}" target="_blank"><img src="{img}" alt="Open in Colab" '
            'style="height:20px;vertical-align:middle;"></a>')


def caption(text: str) -> str:
    """Figure/Table/Listing caption text: muted, under the image, not a heading."""
    return f'<div style="font-size:12.5px;color:{MUTED};margin:6px 0 14px;">{text}</div>'


def summary_prose(paragraphs: list[str]) -> str:
    """Book-derived Summary: cohesive prose, not bullets (TEMPLATE.md 5)."""
    return "\n\n".join(paragraphs)


def intro(text: str) -> str:
    return f'<em style="color:{MUTED};font-size:14.5px;">{text}</em>'


def label(word: str, size: str = "11px", ls: str = ".16em", color: str = CRIMSON) -> str:
    return (
        f'<strong style="font-family:{MONO};font-size:{size};letter-spacing:{ls};'
        f'text-transform:uppercase;color:{color};">{word}</strong>'
    )


def objectives(items: list[str]) -> str:
    lis = "\n".join(f'<li style="margin:6px 0;">{i}</li>' for i in items)
    return (
        f"{label('Objectives')}\n"
        f'<ol style="margin:8px 0 0;padding-left:22px;font-size:15.5px;color:#241c1d;">\n'
        f"{lis}\n</ol>"
    )


def takeaways(items: list[tuple[str, str]]) -> str:
    """Key Takeaways block: mirrors Objectives, but <ul> with a bold lead per item."""
    lis = "\n".join(
        f'<li style="margin:6px 0;"><strong>{lead}</strong> {rest}</li>' for lead, rest in items
    )
    return (
        f"{label('Key Takeaways')}\n"
        f'<ul style="margin:8px 0 0;padding-left:22px;font-size:15.5px;color:#241c1d;">\n'
        f"{lis}\n</ul>"
    )


def callout(kind: str, text: str) -> str:
    """kind: 'Note' | 'Runtime' | 'Caution' | 'Key result' (TEMPLATE.md 3.6).
    Color signals severity: crimson Note, grey Runtime (timing only), amber
    Caution (real pitfalls only), crimson-on-tint Key result."""
    if kind == "Caution":
        rule, color, extra = AMBER, AMBER, ""
    elif kind == "Runtime":
        rule, color, extra = MUTED, MUTED, ""
    elif kind == "Key result":
        rule, color, extra = CRIMSON, CRIMSON, f"background:{TINT};"
    else:
        rule, color, extra = CRIMSON, CRIMSON, ""
    pad = "12px 18px" if kind == "Key result" else "2px 0 2px 18px"
    return (
        f'<div style="margin:0 0 16px;padding:{pad};border-left:2px solid {rule};{extra}">\n'
        f"{label(kind, size='10.5px', color=color)}<br>\n"
        f'<span style="color:{BODY};font-size:14.5px;">{text}</span>\n</div>'
    )


def callout_block(kind: str, html_body: str) -> str:
    """Like callout() but the body is block-level HTML (paragraphs, lists) for rich
    orientation boxes such as 'Before you start'."""
    if kind == "Runtime":
        rule, color = MUTED, MUTED
    elif kind == "Caution":
        rule, color = AMBER, AMBER
    else:
        rule, color = CRIMSON, CRIMSON
    lab = (f'<strong style="font-family:{MONO};font-size:10.5px;letter-spacing:.16em;'
           f'text-transform:uppercase;color:{color};">{kind}</strong>')
    return (f'<div style="margin:0 0 16px;padding:2px 0 2px 18px;border-left:2px solid {rule};">\n'
            f'{lab}\n<div style="color:{BODY};font-size:14.5px;">{html_body}</div>\n</div>')


def env_rows(local: str, colab: str) -> str:
    """The 0 Environment Local/Colab rows, copied from TEMPLATE_CHAPTER.ipynb cell-4.

    Each label carries two trailing em spaces. Where flex works they sit inside the
    96px column and are invisible; in Colab, which drops display/gap/min-width, they
    are the only thing keeping the label off the text ("ColabRun the install cell").
    """
    lab = (
        f'<strong style="font-family:{MONO};font-size:11px;letter-spacing:.06em;'
        f'text-transform:uppercase;color:{CRIMSON};min-width:96px;">'
    )
    return (
        '<div style="font-size:14px;margin:0 0 6px;">\n'
        '<div style="display:flex;gap:14px;padding:9px 0;border-bottom:1px solid #EFEBE8;">'
        f"{lab}Local&#8195;&#8195;</strong><span>{local}</span></div>\n"
        '<div style="display:flex;gap:14px;padding:9px 0;">'
        f"{lab}Colab&#8195;&#8195;</strong><span>{colab}</span></div>\n"
        "</div>"
    )


def md_to_html(md_text: str) -> str:
    """Convert markdown (numbered lists, nested bullets, bold, code, links) to HTML.
    Renders identically on GitHub and in Jupyter, unlike raw markdown inside <details>."""
    import markdown as _md
    html = _md.markdown(md_text.strip(), extensions=["extra", "sane_lists"])
    return html.replace("—", "&middot;")   # honor the em-dash ban in converted copy


def details_md(summary_text: str, md_text: str) -> str:
    """Collapsible exercise whose body is markdown, converted to HTML."""
    return details(summary_text, md_to_html(md_text))


def details(summary_text: str, inner: str) -> str:
    return (
        f'<details style="margin:0 0 14px;border-top:1px solid {HAIRLINE};border-bottom:1px solid {HAIRLINE};">\n'
        f'<summary style="cursor:pointer;padding:11px 0;font-family:{MONO};font-size:12px;'
        f'letter-spacing:.05em;text-transform:uppercase;color:{CRIMSON};">{summary_text}</summary>\n'
        f'<div style="padding:4px 0 14px;font-size:14.5px;color:{BODY};">\n{inner}\n</div>\n</details>'
    )


def references(items: list[str]) -> str:
    lis = "\n".join(f'<li style="margin:7px 0;">{i}</li>' for i in items)
    return f'<ol style="font-size:13.5px;color:{MUTED};padding-left:20px;">\n{lis}\n</ol>'


# ---------------------------------------------------------------- download swap
FETCH_DEF = (
    "from pathlib import Path\n"
    "import urllib.request\n"
    "\n"
    "def fetch(url: str, dest, force: bool = False) -> Path:\n"
    '    """Download `url` to `dest` if not already present. Cross-platform, no shell."""\n'
    "    dest = Path(dest)\n"
    "    dest.parent.mkdir(parents=True, exist_ok=True)\n"
    "    if force or not dest.exists():\n"
    "        urllib.request.urlretrieve(url, dest)\n"
    "    return dest\n"
)

_WGET = re.compile(
    r"""^(\s*)!\s*wget\s+(.*)$""", re.MULTILINE)


def _parse_wget(args: str):
    """Return (url, dest) from a wget argument string. dest=None -> basename."""
    toks = shlex.split(args)
    url = None
    dest = None
    it = iter(range(len(toks)))
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("-O", "--output-document"):
            dest = toks[i + 1]; i += 2; continue
        if t.startswith("-"):
            i += 1; continue
        if url is None:
            url = t
        i += 1
    if dest is None and url is not None:
        dest = url.rsplit("/", 1)[-1]
    return url, dest


def wget_to_fetch(src: str, define: bool = True) -> str:
    """Rewrite every '!wget ...' line in a code cell to a fetch() call. Preserves
    indentation and all non-wget lines (comments, os.makedirs, pip, condacolab)."""
    def repl(m):
        indent, args = m.group(1), m.group(2)
        url, dest = _parse_wget(args)
        if url is None:
            return m.group(0)
        # fetch() (urllib) handles the repo's own raw files. External hosts that urllib
        # can't follow (figshare returns HTTP 202 with an empty body; Drive needs gdown)
        # keep their working shell download, per TEMPLATE.md §8's case-by-case rule.
        if "raw.githubusercontent.com" not in url:
            return m.group(0)
        return f'{indent}fetch("{url}", "{dest}")'
    out = _WGET.sub(repl, src)
    if define and "def fetch(" not in out and "fetch(" in out:
        # Put the helper just above the first fetch() call, after any leading comments.
        lines = out.split("\n")
        first = next(i for i, ln in enumerate(lines) if "fetch(" in ln and "def fetch(" not in ln)
        # rewind past a contiguous comment/blank/makedirs preamble to keep it readable
        out = "\n".join(lines[:first]) + ("\n" if first else "") + FETCH_DEF + "\n" + "\n".join(lines[first:])
    return out
