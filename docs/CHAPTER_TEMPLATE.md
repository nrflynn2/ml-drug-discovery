# Chapter Template — Canonical Cell Order

Every chapter notebook (except the CH12 capstone package) follows this order.
CH01 is the golden reference implementation; measure new chapters against it.

> Legend: **(md)** = markdown cell, **(code)** = code cell,
> **★ mandatory** = the QA definition-of-done requires it.

1. **(md) Title** ★
   `# 📚 Chapter N: <Title>` — portable markdown only. **No `<font color>` HTML**
   (GitHub won't render it).

2. **(md) Companion note + Colab badge**
   One standard companion-reading note (single house style) and the "Open in
   Colab" badge for this notebook.

3. **(md) This chapter covers** ★
   3–6 bullet learning objectives. Present in *every* chapter (CH04 and CH11
   were missing/empty — fix on contact).

4. **(md) Runtime / GPU expectations**
   One line on expected runtime and whether a GPU is recommended (not just for
   DL chapters).

5. **(code) Environment setup**
   ```python
   import bookutils
   data_dir = bookutils.setup_environment("chNN", tier="advanced")
   ```
   On Colab this also handles the tiered install cell (pip tier, or condacolab +
   `LD_LIBRARY_PATH` patch for Chapter 9). No copy-pasted setup blocks.

6. **(code) Grouped imports**
   Standard-library, third-party, then local — grouped and de-duplicated.

7. **(code) Reproducibility + style**
   ```python
   bookutils.set_seed()      # 42, book-wide
   bookutils.setup_style()   # house palette / figsize / DPI
   # cheminformatics chapters:
   bookutils.setup_rdkit_drawing()
   # DL chapters:
   device = bookutils.get_device()
   ```

8. **(md/code) Numbered sections**
   `## 1. …`, `## 2. …` with "Now let's…" lead-ins. Reusable logic lives in
   **docstring'd functions**, not free-floating cells.

9. **(md) Post-figure interpretation cells** ★
   After *every* figure: a short "what this shows / what to look for" cell. This
   is the biggest gap vs. d2l/HF and is required.
   Save figures with `bookutils.save_figure(fig, "name", "chNN")`.

10. **(md) Chapter Summary** ★
    Bullet recap of what was built and learned. (CH11 and Appendix C were
    missing this — fix on contact.)

11. **(md) Exercises**
    A few exercises extending the chapter.

12. **(md) References** ★
    Papers, datasets, and tools cited in the chapter.

## Notes

- **Keep committed outputs** — execute before committing; do not strip outputs.
- **CH12 is the intentional exception**: it is a real `src/` + `scripts/`
  package with its own tests, not a template chapter.
