# ML4DD Notebook Template — "Monograph"

This is the styling contract for the companion notebooks in
`nrflynn2/ml-drug-discovery`. Follow it exactly. It defines a quiet,
journal-style system: a book masthead, a crimson monospace eyebrow, a serif
title under a single hairline rule, section-numbered serif headings, a serif
subhead, a crimson monospace sub-subhead, and understated margin-note callouts.
Boldness lives in one place (the crimson accent and type treatment), not in
decoration.

`TEMPLATE_CHAPTER.ipynb` is the canonical reference: it contains one filled
example of every cell type below. When in doubt, match that file.

---

## 0. The one rule that shapes everything: render targets

These notebooks are read in three kinds of environment, and they render HTML
differently:

| Target | Inline `style=` / CSS | Result |
|---|---|---|
| **GitHub** | **stripped** by the sanitizer | headings, lists, tables, `<hr>`, `<details>`, `<strong>`, `<code>`, `<img width>`, and links still render; styled boxes flatten to clean text |
| **Jupyter / nbviewer** | **kept** | full Monograph look |
| **Colab** | **typographic kept, layout dropped** | color, `font-family`, `font-size`, `text-transform` survive; `margin`, `padding`, `display`, `gap`, `min-width` do not |

Colab is a third target, not a Jupyter alias. Because it keeps the type but drops
the box, a Colab page looks almost right and fails only where spacing was left to
CSS, which is easy to miss when reviewing in Lab.

Consequences you must honor:

1. **Always use semantic HTML that degrades well.** Real `<h1>/<h2>/<h3>/<h4>`
   tags (never a styled `<div>` in their place) so the table of contents and
   GitHub headings survive. Every callout/label is a `<strong>` so it stays
   **bold** on GitHub even when the color and border vanish.
2. **Never rely on color alone to carry meaning.** The label word
   ("Note", "Runtime", "Caution", "Key result") does the work; the color is a
   bonus where CSS is honored.
3. **The masthead is a `<table>`, and the cover is sized with the HTML `width`
   attribute** (not only CSS), so both survive on GitHub.
4. **Never let a space between two pieces of text be produced by CSS.** If
   `margin`, `padding`, `gap`, or `min-width` is the only thing separating a glyph
   from a title or a label from its sentence, Colab renders them fused
   (`§0Environment Setup`, `ColabRun the install cell`). Put a literal `&#8195;`
   (em space) or `&nbsp;` in the markup, or separate the two with a real `<br>`.
   Keep the CSS as well where it earns its keep for alignment, as the Local/Colab
   rows do; just never let it be load-bearing.

Do **not** try to fix GitHub with a `<style>` block — JupyterLab strips or
scopes it and it is unreliable. Inline styles only. **No em dashes, no emoji,
anywhere in copy** (author style); use a period, a semicolon, or `&middot;`.

---

## 1. Palette (do not introduce other colors)

```
crimson       #A20025   accent, rules, primary labels, h4, key-result
crimson-900   #6E0019   (reserved, rarely needed)
amber         #B4690E   Caution callout only
muted         #6B6461   Runtime callout, intro note, references, captions
tint          #FDF3F5   key-result background only
ink           #1A1416   headings (h1/h2/h3)
body          #3d3436   callout body text
hairline      #E7E2DF   exercise rules, cover border
mono   font:  ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace
serif  font:  Georgia,'Iowan Old Style','Palatino Linotype',serif
```

---

## 2. Chapter skeleton (cell order)

Every chapter notebook follows this order. Cells marked **(auto)** are produced
or restyled by `apply_template.py`; cells marked **(agent)** are judgment work.

1. **Book masthead** — identical across all notebooks **(agent/script, §3.0)**
2. Chapter header — eyebrow + title + rule **(auto)**
3. One-line note + crimson Colab badge **(agent, §3.9)**
4. Objectives **(agent, §3.10)**
5. `§0 Environment Setup` header **(auto)** + Local/Colab rows **(agent, §3.7)**
6. Setup/install code cell — includes the `fetch()` helper **(agent, §3.8)**
7. `Import Packages` subhead **(auto)** + imports code cell **(unchanged)**
8. Numbered sections `§1…§N`, each: header **(auto)** → prose/code → callouts **(agent)**
9. Exercises where the chapter has them, in `<details>` under Interactive Exploration **(agent, §3.11)**
10. `Summary` (back-matter) **(auto header)** + book-derived prose **(agent, §5)**
11. `References` / `Interactive Exploration` (back-matter) **(auto header)** + content **(agent)**

**Remove the book's "This chapter covers" bullet list from the notebook
entirely; the Objectives block replaces it.** Code cells, outputs, images, and
body prose are otherwise never restyled. Body paragraphs stay ordinary Markdown.

---

## 3. Cell blocks (copy-paste, fill the `{...}`)

### 3.0 Book masthead — the first markdown cell of every notebook
Identical across all notebooks; book-level, sits above the chapter header. The
`<table>` and the `<img width>` survive on GitHub. The cover and Colab URLs point
at **`main`** (the publish branch), never a feature branch — a feature branch is
deleted after merge and any hardcoded branch name would then 404.

```html
<table style="border:none;border-collapse:collapse;margin:0 0 8px;width:100%;">
<tr style="border:none;">
  <td style="border:none;padding:0;vertical-align:middle;">
    <div style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;color:#A20025;margin:0 0 7px;">Companion notebook</div>
    <div style="font-family:Georgia,'Iowan Old Style',serif;font-size:18px;color:#1A1416;line-height:1.25;margin:0 0 7px;">
      <a href="https://www.manning.com/books/build-ai-drug-discovery-pipelines" style="color:#1A1416;text-decoration:none;border-bottom:1px solid #A20025;padding-bottom:1px;">Build AI Drug Discovery Pipelines</a>
    </div>
    <div style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:12px;color:#6B6461;line-height:1.7;">
      Noah Flynn&nbsp;&middot;&nbsp;Manning Publications<br>
      Code&nbsp;&middot;&nbsp;<a href="https://github.com/nrflynn2/ml-drug-discovery" style="color:#A20025;text-decoration:none;">github.com/nrflynn2/ml-drug-discovery</a>
    </div>
  </td>
  <td style="border:none;padding:0 0 0 22px;vertical-align:middle;width:104px;text-align:right;">
    <a href="https://www.manning.com/books/build-ai-drug-discovery-pipelines">
      <img src="https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/assets/cover-sm.png" width="88" alt="Build AI Drug Discovery Pipelines cover" style="border:1px solid #E7E2DF;border-radius:3px;display:inline-block;">
    </a>
  </td>
</tr>
</table>
<hr style="border:0;border-top:1px solid #A20025;opacity:.55;margin:0 0 22px;">
```

### 3.1 Chapter header — produced by the script
```html
<div style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.24em;text-transform:uppercase;color:#A20025;margin:0 0 12px;">Chapter {N}</div>

<h1 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:33px;line-height:1.12;margin:0 0 16px;color:#1A1416;letter-spacing:-.01em;">{Chapter Title}</h1>

<hr style="border:0;border-top:1px solid #A20025;opacity:.55;margin:0 0 4px;">
```
The eyebrow reads `Appendix C` for the appendix; the script derives it from the
`# Chapter N` / `# Appendix X` line, or takes `--eyebrow` for notebooks (like the
CH12 trio) whose title carries no chapter word.

### 3.2 Opener note (one line, right after the chapter header)
The masthead already frames the book, so keep this to one line, followed by the
badge in §3.9.
```html
<em style="color:#6B6461;font-size:14.5px;">Runnable code for Chapter {N}; read alongside the book.</em>
```

### 3.3 Numbered section header (h2) — produced by the script
```html
<h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§{N}&#8195;</span>{Section Title}</h2>
```
`§0` is Environment Setup; `§1…§N` are the body sections, in source order.
The gutter is the `&#8195;` em space **inside** the span, sized by the span's own
13px mono, not a `margin-right` on it (rule 4 in §0). Use the literal `§`, not
`&sect;`.

### 3.4 Back-matter header (h2) — produced by the script
Summary, Interactive Exploration, References / Further Reading, Conclusion: a thin
crimson hairline rule above, then the serif title with **no `§`**.
```html
<hr style="border:0;border-top:1px solid #A20025;opacity:.55;margin:34px 0 12px;">
<h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:0 0 10px;color:#1A1416;">{Summary | Interactive Exploration | References}</h2>
```

### 3.5 Subsection (h3) and sub-subsection (h4) — produced by the script
Four heading levels, each visually distinct and legible. h3 is **serif, present,
normal case** (the fix for too-quiet subheads); h4 is **mono, uppercase, crimson,
heavier, with no leading glyph** (distinguished from h3 by color + weight +
mono/caps).
```html
<h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">{Subsection}</h3>
```
```html
<h4 style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-weight:600;font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:#A20025;margin:18px 0 6px;">{Sub-subsection}</h4>
```

### 3.6 Callouts — four types; color signals severity
```html
<!-- Note: neutral aside / info (crimson rule) -->
<div style="margin:0 0 16px;padding:2px 0 2px 18px;border-left:2px solid #A20025;">
<strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:#A20025;">Note</strong><br>
<span style="color:#3d3436;font-size:14.5px;">{text}</span>
</div>

<!-- Runtime: "this cell takes N minutes; you may skip it and load the precomputed result" (grey rule) -->
<div style="margin:0 0 16px;padding:2px 0 2px 18px;border-left:2px solid #6B6461;">
<strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:#6B6461;">Runtime</strong><br>
<span style="color:#3d3436;font-size:14.5px;">{text}</span>
</div>

<!-- Caution: genuine correctness / data pitfalls ONLY (amber rule) -->
<div style="margin:0 0 16px;padding:2px 0 2px 18px;border-left:2px solid #B4690E;">
<strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:#B4690E;">Caution</strong><br>
<span style="color:#3d3436;font-size:14.5px;">{text}</span>
</div>

<!-- Key result: concrete takeaway, inline where earned (tinted) -->
<div style="margin:0 0 16px;padding:12px 18px;border-left:2px solid #A20025;background:#FDF3F5;">
<strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:#A20025;">Key result</strong><br>
<span style="color:#3d3436;font-size:14.5px;">{text}</span>
</div>
```
Routing:
- **Note** — a neutral aside.
- **Runtime** — pure timing ("this cell takes N minutes; you may skip it and load
  the precomputed result"). Not amber; a long cell is not a hazard.
- **Caution** — reserve amber for **real traps only**: leakage, an unguarded
  `None` that will crash, a silent broadcast, a wrong-axis reduction.
- **Key result** — a concrete takeaway, placed **inline at the point it is
  earned**, roughly one per section where a result lands. If it cites specific
  numbers, they **must match the notebook's current seeded output**; flag a
  mismatch rather than editing prose to fit.

`<code>…</code>` is fine inside any callout.

### 3.7 Environment callout (`§0`, after the header)
uv / Python 3.12 is the default path. Use the chapter's real install **tier**:
`uv sync` (Ch1–4 core), `uv sync --extra advanced` (Ch5–8, 10, 11, Appendix C),
`uv sync --extra full` (+ Ch9 helpers); Ch9 additionally needs the conda island;
Ch12 uses `pip install -e ".[notebooks]"`. Match the repo's current README /
INSTALL / pyproject if they differ.
```html
<div style="font-size:14px;margin:0 0 6px;">
  <div style="display:flex;gap:14px;padding:9px 0;border-bottom:1px solid #EFEBE8;"><strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#A20025;min-width:96px;">Local&#8195;&#8195;</strong><span>Install <a href="https://docs.astral.sh/uv/" style="color:#A20025;">uv</a>, then run <code>uv sync{ --extra TIER}</code> from the repo root (Python 3.12) and launch with <code>uv run jupyter lab</code>.</span></div>
  <div style="display:flex;gap:14px;padding:9px 0;"><strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#A20025;min-width:96px;">Colab&#8195;&#8195;</strong><span>Run the install cell below (same pinned versions), then restart the runtime.</span></div>
</div>
```
The two `&#8195;` after each label are load-bearing in Colab, which drops
`display:flex` / `gap` / `min-width` and would otherwise render `ColabRun the
install cell` (rule 4 in §0). Where flex is honored they sit inside the 96px
column and are invisible, so the aligned look is unchanged.

Chapter 9 only keeps the conda path (Local: `conda env create -f ml4dd2025.yml`).
Remove stale conda / `ml4dd2025.yml` references from every other chapter.

### 3.8 Data loading — one portable `fetch()` for every notebook
Replace every shell download (`!wget`, `!curl`) with a pure-Python helper so the
notebook runs identically on WSL2, native Windows, macOS, and Colab. Define it
inline in the setup cell (no import or clone dependency); idempotent; pathlib
based; makes parent dirs.
```python
from pathlib import Path
import urllib.request

def fetch(url: str, dest, force: bool = False) -> Path:
    """Download `url` to `dest` if not already present. Cross-platform, no shell."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if force or not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest
```
Map old flags: `wget -O file url` → `fetch(url, file)`; `wget -nc` → the default
skip-if-present. For sources plain urllib cannot handle (Google Drive, a 403,
auth), switch case by case (`gdown` for Drive, `requests` with a user-agent). Do
not restyle a `!pip install` cell — only downloads move to `fetch()`.

### 3.9 Colab badge (crimson, per-chapter — its URL differs per notebook)
Kept in the chapter opener, right after the one-line note.
```html
<a href="https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/CH0{N}_FLYNN_ML4DD.ipynb" target="_blank"><img src="https://img.shields.io/badge/Open%20in-Colab-A20025?logo=googlecolab&logoColor=white" alt="Open in Colab" style="height:20px;vertical-align:middle;"></a>
```

### 3.10 Objectives (new cell, right after the note+badge)
```html
<strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#A20025;">Objectives</strong>
<ol style="margin:8px 0 0;padding-left:22px;font-size:15.5px;color:#241c1d;">
<li style="margin:6px 0;">{objective 1}</li>
<li style="margin:6px 0;">{objective 2}</li>
<li style="margin:6px 0;">{objective 3}</li>
</ol>
```
Write 3–5 objectives from the chapter's actual learning goals. Start each with a
verb (Curate, Generate, Train, Prevent). No trailing period.

### 3.11 Exercise (collapsible; GitHub renders `<details>` natively)
Open-ended, under the Interactive Exploration back-matter head. Keep the chapter's
prompts verbatim; do **not** invent a "Solution." where the book supplies none.
```html
<details style="margin:0 0 14px;border-top:1px solid #E7E2DF;border-bottom:1px solid #E7E2DF;">
<summary style="cursor:pointer;padding:11px 0;font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:#A20025;">Try it yourself &middot; {short title}</summary>
<div style="padding:4px 0 14px;font-size:14.5px;color:#3d3436;">
<p>{prompt}</p>
</div>
</details>
```

### 3.12 References list (after the back-matter `References` header)
```html
<ol style="font-size:13.5px;color:#6B6461;padding-left:20px;">
<li style="margin:7px 0;">{Author} ({year}). <em>{Title}.</em> {Venue}.</li>
</ol>
```

### 3.13 Caption (not a heading)
"Figure N", "Table N", "Listing N" are captions, not headings. Render as muted
caption text under the relevant image or output, never in the heading tree.
```html
<div style="font-size:12.5px;color:#6B6461;margin:6px 0 14px;">{Figure N. Caption text.}</div>
```

---

## 4. Two-step workflow

**Step 1 — headers (deterministic, run once):**
```bash
python apply_template.py CHNN_FLYNN_ML4DD.ipynb --in-place        # .bak saved
python apply_template.py '*.ipynb' --in-place                    # whole repo
python apply_template.py APPENDIX_C_FLYNN_ML4DD.ipynb --in-place --autonumber
python apply_template.py CH12_FLYNN_ML4DD/notebooks/*.ipynb --in-place --eyebrow "Chapter 12"
```
This converts every legacy `<font color>` header to §3.1/3.3/3.4/3.5, strips the
emoji/keycaps, and **upgrades headers emitted by an older version of this template**
to the current blocks. It is idempotent and touches nothing but header lines.

**Step 2 — semantic cells (agent judgment):** open the converted notebook and, using
the blocks above, add the masthead (§3.0), the one-line note + badge (§3.2, 3.9),
Objectives (§3.10), the `§0` Local/Colab rows (§3.7), the `fetch()` swap (§3.8),
callouts (§3.6) where the text flags a tip, a timing, a trap, or a takeaway,
exercises (§3.11), the book-derived Summary (§5), and the References list (§3.12).

Then run the acceptance check in §6.

---

## 5. Summary vs. Key result — keep them separate

- **Summary** (back-matter) is the **book-derived prose recap**: a cohesive prose
  adaptation of the book's chapter summary, high fidelity to the book, **not
  bulleted and not an AI-rewritten version**. Do **not** aggregate takeaways there.
- **Key result** callouts are the takeaways, dispersed **inline** at the point each
  result is earned (§3.6).

Do not fold the takeaways into Summary, and do not turn Summary into a bullet list.

---

## 6. Acceptance checklist (per notebook)

- [ ] First cell is the book masthead (table layout, `<img width>`, absolute raw
      cover URL on `main`); chapter header follows.
- [ ] Cover and Colab URLs target `main`, not a feature branch.
- [ ] Opener: one Objectives list, no "This chapter covers", a one-line note, the
      crimson Colab badge.
- [ ] `§0` Environment, `§1…§N` body; back-matter headings have a crimson hairline
      rule and no `§`.
- [ ] Four distinct, legible heading levels; h3 serif and present; h4 mono crimson
      with no leading `·`; no em dashes anywhere.
- [ ] Callouts: amber only on real pitfalls; timing is Runtime; Note and Key result
      correct; any numbers in a Key result match current output.
- [ ] No "Figure/Table/Listing N" left in the heading tree.
- [ ] Summary is the book-derived prose recap; takeaways are inline Key results.
- [ ] Environment callout is uv/3.12 at the chapter's tier (conda only in ch9).
- [ ] No `!wget`/`!curl`; all downloads go through `fetch()`.
- [ ] No emoji, no `style` on any code cell; code cells and outputs unchanged.
- [ ] Renders on GitHub as clean headings + bold labels + surviving masthead
      (spot-check an nbviewer/GitHub preview).

---

## 7. Known edge cases

- **CH12 subfolder notebooks** use `## Part 1: …` text numbering, not keycaps. The
  script reads the number from `Part N` (so a gap like antibody's Part 6 → Part 12
  is preserved, never closed) and maps it to `§N`; pass `--eyebrow "Chapter 12"`.
- **Appendix C** has no keycaps; run with `--autonumber` for `§1…§N` in source order
  (Environment stays `§0`; Conclusion is back-matter).
- **Chapter 9** keeps the conda environment callout; every other chapter is uv.
- Re-running the script is a safe no-op, and upgrades any header left in an older
  Monograph style to the current one.

---

## 8. Self-contained bonus notebooks (extra content, no book chapter)

The 15 chapters are **companion** notebooks: code only, with prose, figures, and
equations in the book — hence the "read alongside the book" opener. A **bonus**
notebook is extra content for the same book with **no corresponding chapter**, so
the write-up lives in the notebook itself. Same Monograph look, five differences:

1. **Masthead eyebrow** is `Bonus notebook`, not `Companion notebook`.
2. **Title eyebrow** is a topic or series label (e.g. `Bonus`, `Deep Dive`), **not
   `Chapter N`** — there is no book chapter number.
3. **Opener note** is the self-contained one (no "read alongside the book"):
   *"A self-contained bonus notebook: the explanation and the code both live here…"*
4. **Prose is first-class.** Body markdown carries the narrative that a chapter would
   have pushed to the book. **Equations render** in Jupyter/nbviewer and on GitHub —
   `$…$` inline, `$$…$$` display. Figures are code-generated with a `caption()` under
   them (§3.13), never a heading.
5. **Summary is written from the notebook**, not sourced from the book (there is no
   book chapter to adapt); still a cohesive prose recap, not a bullet list.

Everything else (objectives, `§0` environment, the four callouts, `fetch()`,
References, no em dashes, no emoji) is identical.

**Tooling.** Don't hand-build these. `scaffold_notebook.py` mints a ready-to-fill
skeleton with the above baked in; `nbedit.py` is the block-builder library it and the
chapter pass both use (masthead, headers, callouts, objectives, env rows, references —
all emitting the exact blocks in §3, with notebook JSON written to this repo's
convention). `apply_template.py` is **not** needed for greenfield content (there are no
legacy headers to remap); author headers with `nbedit` or run the script over
plain-markdown headers if you prefer to write `##`/`###` first.

```bash
python qa_reports/visual_format_sprint/scaffold_notebook.py \
    --title "Explainable AI for Binding-Affinity Models" --eyebrow "Bonus" \
    --sections 4 --tier advanced --out bonus/xai_binding.ipynb
```
