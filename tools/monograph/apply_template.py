#!/usr/bin/env python3
"""
apply_template.py  —  ML4DD "Monograph" notebook template applier
==================================================================

Deterministically remaps the legacy header style

    # <b> <font color='#A20025'> 📚 Chapter 3: Ligand-based Screening ...
    ## <b> <font color='#A20025'> 2️⃣ Data Acquisition ...
    ### <b> <font color='#A20025'> Import Packages

onto the Monograph system (crimson mono eyebrow + serif title + hairline
rule; §-numbered serif section heads; mono-uppercase subheads). It ONLY
touches header lines in markdown cells. Code cells, body prose, images,
and outputs are never modified.

What it does NOT do (left to a coding agent + TEMPLATE.md, by design):
  - write Learning Objectives content
  - convert inline warnings into margin-note callouts
  - wrap exercises in <details>
These need judgment; the header remap is the deterministic 90%.

Usage
-----
  python apply_template.py CH03_FLYNN_ML4DD.ipynb            # -> CH03_..._monograph.ipynb
  python apply_template.py *.ipynb                           # batch
  python apply_template.py CH03_FLYNN_ML4DD.ipynb --in-place # edits file, writes .bak
  python apply_template.py CH03_FLYNN_ML4DD.ipynb --dry-run  # print change log only
  python apply_template.py *.ipynb --keep-emoji              # keep 📚/⚙️/keycaps as-is

Sources carrying no keycap numbering need one of:
  python apply_template.py APPENDIX_C_FLYNN_ML4DD.ipynb --in-place --autonumber
  python apply_template.py CH12_FLYNN_ML4DD/notebooks/*.ipynb --in-place --eyebrow "Chapter 12"

Idempotent: converted headers are no longer markdown "#" lines, so a second
run is a no-op.
"""
from __future__ import annotations
import argparse, json, re, sys, shutil
from pathlib import Path
from string import Template

# ----------------------------------------------------------------------
# PALETTE  (single source of truth — change here, re-run to restyle)
# ----------------------------------------------------------------------
CRIMSON   = "#A20025"
CRIMSON9  = "#6E0019"
INK       = "#1A1416"
MUTED     = "#6B6461"
HAIRLINE  = "#E7E2DF"
MONO   = "ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace"
SERIF  = "Georgia,'Iowan Old Style','Palatino Linotype',serif"

# ----------------------------------------------------------------------
# HTML BLOCKS  (string.Template -> no brace-escaping headaches)
# ----------------------------------------------------------------------
T_EYEBROW = Template(
'<div style="font-family:$MONO;font-size:11px;letter-spacing:.24em;'
'text-transform:uppercase;color:$CRIMSON;margin:0 0 12px;">$eyebrow</div>\n\n'
)

T_CHAPTER = Template(
'<h1 style="font-family:$SERIF;font-weight:500;font-size:33px;line-height:1.12;'
'margin:0 0 16px;color:$INK;letter-spacing:-.01em;">$title</h1>\n\n'
'<hr style="border:0;border-top:1px solid $CRIMSON;opacity:.55;margin:0 0 4px;">'
)

T_SEC_NUM = Template(  # gutter is a literal em space, not margin — Colab drops layout CSS
'<h2 style="font-family:$SERIF;font-weight:500;font-size:23px;margin:34px 0 10px;color:$INK;">'
'<span style="font-family:$MONO;font-size:13px;color:$CRIMSON;font-weight:500;">'
'§$sec&#8195;</span>$title</h2>'
)

T_SEC_PLAIN = Template(  # back-matter: Summary / References — thin crimson rule, serif, NO §
'<hr style="border:0;border-top:1px solid $CRIMSON;opacity:.55;margin:34px 0 12px;">\n'
'<h2 style="font-family:$SERIF;font-weight:500;font-size:23px;margin:0 0 10px;color:$INK;">$title</h2>'
)

T_H3 = Template(  # subsection: serif, present, normal case (loud enough to read as a heading)
'<h3 style="font-family:$SERIF;font-weight:600;font-size:17px;color:$INK;'
'margin:24px 0 8px;">$title</h3>'
)

T_H4 = Template(  # sub-subsection: mono, uppercase, crimson, heavier; NO leading glyph
'<h4 style="font-family:$MONO;font-weight:600;font-size:11.5px;letter-spacing:.1em;'
'text-transform:uppercase;color:$CRIMSON;margin:18px 0 6px;">$title</h4>'
)

def sub(t: Template, **kw) -> str:
    return t.substitute(MONO=MONO, SERIF=SERIF, CRIMSON=CRIMSON, CRIMSON9=CRIMSON9,
                        INK=INK, MUTED=MUTED, HAIRLINE=HAIRLINE, **kw)

# ----------------------------------------------------------------------
# PARSING
# ----------------------------------------------------------------------
KEYCAP = re.compile(r'([0-9])\uFE0F?\u20E3')
EMOJI  = re.compile(
    r'[\U0001F000-\U0001FAFF\u2600-\u27BF\u2190-\u21FF\u2B00-\u2BFF\uFE0F\u200D\u20E3]')
# CommonMark allows up to 3 leading spaces before a heading's '#', and the repo uses them:
# four " ### <b> <font ...> Setup Configuration" heads (CH08/09/10/11) render as h3 in Jupyter
# but are missed by a '^#'-anchored pattern, silently surviving the remap as legacy crimson.
HEADER = re.compile(r'^ {0,3}(#{1,6})\s+(.*)$')

# titles that belong to back-matter (rendered without a § number)
BACK_MATTER = {
    "summary", "references", "references and further reading",
    "further reading", "further exploration", "interactive exploration",
    "conclusion",
    # Takeaway closes. Membership keeps --autonumber from ever numbering one.
    "key takeaways", "summary and key takeaways", "summary and best practices",
}

def clean_title(raw: str, keep_emoji: bool):
    """Strip <b>/<font>/markdown-bold/keycaps/emoji; return (clean_title, keycap_digit|None)."""
    t = raw
    t = re.sub(r'</?b>', '', t, flags=re.I)
    t = re.sub(r'<font[^>]*>', '', t, flags=re.I)
    t = re.sub(r'</font>', '', t, flags=re.I)
    # Markdown bold wrapping a header title (CH01 uses ``## **<font ...>1️⃣ Title</font>**``).
    # The title becomes raw HTML, where ``**`` renders as literal asterisks rather than bold.
    t = t.replace('**', '')
    m = KEYCAP.search(t)
    keycap = m.group(1) if m else None
    if not keep_emoji:
        t = KEYCAP.sub('', t)
        t = EMOJI.sub('', t)
    t = re.sub(r'\s{2,}', ' ', t).strip(" :·-–—")
    return t.strip(), keycap

class Opts:
    """Per-run options plus per-notebook section-numbering state."""

    def __init__(self, eyebrow: str = "", autonumber: bool = False):
        self.eyebrow = eyebrow
        self.autonumber = autonumber
        self.counter = 0

    def reset(self) -> None:
        self.counter = 0


def transform_header(level: int, body: str, keep_emoji: bool, log: list, opts: Opts):
    clean, keycap = clean_title(body, keep_emoji)
    low = clean.lower()

    if level == 1:
        # "Chapter 3: Title" and "Appendix C: Title" both split into eyebrow + title.
        m = re.match(r'(chapter|appendix)\s+(\w+)\s*[:.\-]?\s*(.*)', clean, flags=re.I)
        if m:
            kind, ident = m.group(1).capitalize(), m.group(2)
            eyebrow = f"{kind} {ident if ident.isdigit() else ident.upper()}"
            title = m.group(3).strip() or clean
        else:
            # Source carries no "Chapter N"/"Appendix X" (the CH12 subfolder notebooks).
            # Fall back to --eyebrow; with neither, emit the title alone rather than a
            # dangling "CHAPTER" eyebrow with no number after it.
            eyebrow, title = opts.eyebrow, clean
        log.append(f"  h1  -> {eyebrow or '(no eyebrow)'} | {title}")
        head = sub(T_EYEBROW, eyebrow=eyebrow) if eyebrow else ""
        return head + sub(T_CHAPTER, title=title)

    if level == 2:
        if "environment" in low or low.startswith("setup"):
            log.append(f"  h2  -> §0 {clean}")
            return sub(T_SEC_NUM, sec="0", title=clean)
        if low in BACK_MATTER:
            log.append(f"  h2  -> (back-matter) {clean}")
            return sub(T_SEC_PLAIN, title=clean)
        if keycap:
            opts.counter = max(opts.counter, int(keycap))
            log.append(f"  h2  -> §{keycap} {clean}")
            return sub(T_SEC_NUM, sec=keycap, title=clean)
        # Text numbering: "Part 7: Title" -> §7. The number is read from the source, so a
        # gap in the source sequence is preserved rather than closed. (ch12-antibody jumps
        # Part 6 -> Part 12; those missing parts are an open author decision, Ch12·Q2.)
        mp = re.match(r'part\s+(\d+)\s*[:.\-]?\s*(.*)', clean, flags=re.I)
        if mp:
            sec, title = mp.group(1), mp.group(2).strip() or clean
            opts.counter = max(opts.counter, int(sec))
            log.append(f"  h2  -> §{sec} {title}   (from 'Part {sec}')")
            return sub(T_SEC_NUM, sec=sec, title=title)
        if opts.autonumber:
            opts.counter += 1
            log.append(f"  h2  -> §{opts.counter} {clean}   (autonumbered)")
            return sub(T_SEC_NUM, sec=str(opts.counter), title=clean)
        log.append(f"  h2  -> (unnumbered) {clean}")
        return sub(T_SEC_PLAIN, title=clean)

    if level == 3:
        log.append(f"  h3  -> {clean}")
        return sub(T_H3, title=clean)

    # level 4+
    log.append(f"  h4  -> {clean}")
    return sub(T_H4, title=clean)

def already_converted(line: str) -> bool:
    return ("font-family:Georgia" in line) or ("§</span>" in line) or \
           ('letter-spacing:.24em' in line)

# ----------------------------------------------------------------------
# RESTYLE / UPGRADE
# A header emitted by a *previous* version of this template is not a "#" line, so
# HEADER never matches it. These detectors recognise the earlier Monograph output
# (mono-uppercase h3, muted "&middot;"-prefixed h4, "§"-span back-matter) and let a
# re-run upgrade it to the current templates. Each is anchored on the OLD style
# signature (a font or colour the new template no longer uses), so upgrading is a
# fixed point: the new output no longer matches, and a second run is a no-op.
# ----------------------------------------------------------------------
_OLD_H3 = re.compile(r'<h3 style="font-family:ui-monospace[^"]*">(.*?)</h3>')
_OLD_H4 = re.compile(r'<h4 style="font-family:ui-monospace[^"]*color:#6B6461[^"]*">'
                     r'\s*(?:&middot;|&mdash;|—|·)?\s*(.*?)</h4>')
_OLD_BACKMATTER = re.compile(
    r'<h2 style="font-family:Georgia[^"]*"><span style="[^"]*">§</span>(.*?)</h2>')

def upgrade_header(line: str, log: list) -> str | None:
    """Upgrade one emitted-Monograph header line to the current template, else None."""
    m = _OLD_BACKMATTER.search(line)
    if m:
        log.append(f"  restyle back-matter -> {m.group(1)[:48]}")
        return sub(T_SEC_PLAIN, title=m.group(1))
    m = _OLD_H3.search(line)
    if m:
        log.append(f"  restyle h3 -> {m.group(1)[:48]}")
        return sub(T_H3, title=m.group(1))
    m = _OLD_H4.search(line)
    if m:
        log.append(f"  restyle h4 -> {m.group(1)[:48]}")
        return sub(T_H4, title=m.group(1).strip())
    return None

def process_cell(cell: dict, keep_emoji: bool, log: list, opts: Opts) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    src = cell["source"]
    lines = "".join(src).split("\n")
    changed = False
    out = []
    for ln in lines:
        m = HEADER.match(ln)
        if m and not already_converted(ln):
            level = len(m.group(1))
            html = transform_header(level, m.group(2), keep_emoji, log, opts)
            out.append(html)
            changed = True
        else:
            upgraded = upgrade_header(ln, log)
            if upgraded is not None:
                out.append(upgraded)
                changed = True
            else:
                out.append(ln)
    if changed:
        # re-join, keep trailing-newline convention of nbformat (list of lines)
        joined = "\n".join(out)
        cell["source"] = [l + "\n" for l in joined.split("\n")]
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip("\n")
    return changed

# ----------------------------------------------------------------------
# DRIVER
# ----------------------------------------------------------------------
def process_notebook(path: Path, in_place: bool, dry_run: bool, keep_emoji: bool, opts: Opts):
    nb = json.loads(path.read_text(encoding="utf-8"))
    log: list[str] = []
    opts.reset()  # section numbering restarts at each notebook
    n = sum(process_cell(c, keep_emoji, log, opts) for c in nb["cells"])
    print(f"\n=== {path.name} ===")
    if not n:
        print("  no legacy headers found (already Monograph, or none present)")
        return
    print("\n".join(log))
    print(f"  {n} cell(s) updated")
    if dry_run:
        print("  [dry-run] nothing written")
        return
    if in_place:
        shutil.copy(path, path.with_suffix(".ipynb.bak"))
        out = path
    else:
        out = path.with_name(path.stem + "_monograph.ipynb")
    # Trailing newline: the repo's notebooks all end in one, and json.dumps does not emit it.
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  written -> {out.name}" + ("  (.bak saved)" if in_place else ""))

def main():
    ap = argparse.ArgumentParser(description="Apply the ML4DD Monograph header template.")
    ap.add_argument("notebooks", nargs="+", help=".ipynb files (globs ok)")
    ap.add_argument("--in-place", action="store_true", help="edit files, save .bak")
    ap.add_argument("--dry-run", action="store_true", help="log only, write nothing")
    ap.add_argument("--keep-emoji", action="store_true", help="keep 📚/⚙️/keycaps")
    ap.add_argument("--eyebrow", default="", metavar="TEXT",
                    help='eyebrow for notebooks whose h1 has no "Chapter N"/"Appendix X" '
                         '(e.g. --eyebrow "Chapter 12" for the CH12 subfolder trio)')
    ap.add_argument("--autonumber", action="store_true",
                    help="number otherwise-unnumbered content sections in source order "
                         "(§0 Environment and back-matter are skipped); for sources with "
                         "no keycap or 'Part N' numbering, e.g. APPENDIX_C")
    a = ap.parse_args()
    opts = Opts(eyebrow=a.eyebrow, autonumber=a.autonumber)
    paths = [Path(p) for pat in a.notebooks for p in sorted(Path().glob(pat))] \
            if any(("*" in p or "?" in p) for p in a.notebooks) \
            else [Path(p) for p in a.notebooks]
    if not paths:
        sys.exit("no notebooks matched")
    for p in paths:
        if p.exists():
            process_notebook(p, a.in_place, a.dry_run, a.keep_emoji, opts)
        else:
            print(f"skip (not found): {p}")

if __name__ == "__main__":
    main()
