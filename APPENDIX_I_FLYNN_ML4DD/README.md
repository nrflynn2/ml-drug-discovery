# Appendix I &middot; Diffusion and Flow Matching

Companion material for **Appendix I** of *Build AI Drug Discovery Pipelines* (Manning). Five bonus
notebooks, the cached results they read, and every figure the appendix prints.

The notebooks are self-contained and run top to bottom. Each one loads cached results by default, so you
can read the whole appendix through without a GPU, then flip a single flag in any cell to recompute that
result yourself.

```
Appendix_I_FLYNN_ML4DD/
  notebooks/    the five bonus notebooks, in reading order
  data/         cached results the notebooks load (committed, about 4 MB)
  figures/      every figure as png + svg + pdf (committed)
  artifacts/    empty; large downloads land here at runtime (git-ignored)
  manuscript/   the tracked-changes draft and the three review documents
```

---

## Quick start

```bash
git clone https://github.com/nrflynn2/ml-drug-discovery.git
cd ml-drug-discovery/Appendix_I_FLYNN_ML4DD

conda create -n appendix-i python=3.11 -y
conda activate appendix-i
pip install -r requirements.txt

jupyter lab notebooks/
```

Open `APPENDIX_I_NB1_FLYNN_ML4DD.ipynb` and run all cells. That is the whole setup for Notebooks 1, 2 and
5. Notebooks 3 and 4 need two extra installs, described below, because they load third-party model code.

Every notebook resolves its own paths, so `jupyter lab` from `notebooks/` works and so does running from
the folder root. The anchor at the top of each notebook is:

```python
HERE = Path.cwd()
APPENDIX = HERE.parent if HERE.name == "notebooks" else HERE
DATA = APPENDIX / "data"
FIGURES = APPENDIX / "figures"
ARTIFACTS = APPENDIX / "artifacts"
```

On Google Colab the first cell clones this repository and changes into the right directory for you, so
you can open any notebook from a Colab badge and run it without editing paths.

---

## The five notebooks

Read them in order. Each builds on the one before, and the last two carry the results the appendix's
closing sections rely on.

| Notebook | What it builds | Runs on | Wall clock |
|---|---|---|---|
| **NB1** Diffusion from the ground up | a DDPM on a 2-D two-moons target, plus the DDIM sampler and the eta knob | CPU | about 3 min (trains inline) |
| **NB2** Flow matching | a velocity field on the same target, then the three-sampler budget sweep and the eta sweep | CPU | about 2 min cached, about 25 min fresh |
| **NB3** FlowMol3 | 3-D molecule generation, then PoseBusters, UFF strain, REOS alerts and novelty against GEOM-Drugs | CPU | about 2 min cached, about 20 min fresh |
| **NB4** DiffSBDD | pocket-conditioned design in the A2A receptor, three tasks from one checkpoint, pocket-aware validity, novelty and diversity | CPU | about 3 min cached, about 30 min fresh |
| **NB5** Frontier models | RFdiffusion self-consistency, then the controlled Boltz-2 comparison and the DiffDock re-docking cross-check | CPU for the cached path; A100 to re-run Boltz-2 | about 2 min cached |

`USE_CACHED = True` at the top of each heavy cell is the switch. Leave it alone to read the numbers the
appendix prints; set it `False` to regenerate that specific result.

---

## Setup, step by step

### 1. The base environment (NB1, NB2, NB5)

```bash
conda create -n appendix-i python=3.11 -y
conda activate appendix-i
pip install -r requirements.txt
```

That gives you PyTorch, RDKit, PoseBusters, PoseCheck, pandas, matplotlib and TorchCFM. Nothing here
needs a GPU.

One note on PoseCheck: it pins an old pandas, so `requirements.txt` installs it with `--no-deps` and
pins pandas separately. If you install PoseCheck yourself, do the same or pandas will be downgraded out
from under the other packages.

PoseCheck protonates proteins with the `hydride` command-line tool, which pip installs cleanly. Its
docstring mentions a `reduce` binary; you do not need it, and you do not need a compiler.

### 2. FlowMol3 (NB3 only)

FlowMol3 depends on DGL, which needs to match your PyTorch build:

```bash
pip install torch==2.1.2
pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/torch-2.1/repo.html
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
pip install git+https://github.com/Dunni3/FlowMol.git
```

`flowmol.load_pretrained('flowmol3')` downloads the checkpoint on first use, about 68 MB, into
`artifacts/`. NB3's cached path does not need it, so skip this section entirely if you only want to read.

### 3. DiffSBDD (NB4 only)

NB4 clones the DiffSBDD repository and imports its modules directly:

```bash
git clone https://github.com/arneschneuing/DiffSBDD.git
pip install torch==2.0.1 torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
```

The pocket-conditioned checkpoint (`crossdocked_fullatom_cond.ckpt`, about 18 MB) downloads from Zenodo
record 8183747 into `artifacts/` when you run the fresh path. The A2A receptor and its co-crystallized
caffeine come from DiffSBDD's own `example/` directory after the clone.

### 4. Boltz-2 and DiffDock (NB5, optional)

NB5 reads cached results by default and needs nothing extra. To re-run the affinity comparison yourself:

```bash
pip install boltz==2.2.1
```

Boltz-2's affinity head wants 24 to 48 GB of VRAM, so this is an A100-class job. Each of the seven
ligands took about 150 seconds on an A100; the whole batch was about 18 minutes of GPU time.

The multiple-sequence alignment matters more than you might expect. Fetch AlphaFold-DB's precomputed
alignment rather than letting the MSA server build one:

```python
import urllib.request
urllib.request.urlretrieve(
    "https://alphafold.ebi.ac.uk/files/msa/AF-P29274-F1-msa_v6.a3m",
    "artifacts/a2a.a3m")
```

That is 17,819 sequences for the A2A receptor. Point the YAML's `msa:` field at it.

The re-docking cross-check used DiffDock through NVIDIA's hosted endpoint. Those served weights were
retrained by NVIDIA on PLINDER and SAIR, so re-running from `gcorso/DiffDock` will not reproduce the
numbers exactly. The pose disagreement is the finding, and it is robust; the specific RMSD values are not.

---

## Reproducing the appendix's numbers

Every number the appendix prints is in `data/`, and `data/appendix_I_review_assets.json` is the index.
Each entry carries the value, the file it came from, that file's SHA-256 prefix, the definition of the
metric, and the caveat if there is one. If you want to check a number in the book, start there.

**One parsing hazard, and it will bite you.** Several CSVs in `data/` carry provenance comments as leading
`#` lines, and several SMILES columns contain `#` for triple bonds. Passing `comment='#'` to
`pandas.read_csv` truncates those rows mid-line and silently gives you NaN. Strip only the leading
comment lines:

```python
lines = open(path).read().splitlines()
i = 0
while i < len(lines) and lines[i].lstrip().startswith("#"):
    i += 1
df = pd.read_csv(io.StringIO("\n".join(lines[i:])))
```

The notebooks already do this. It is called out here because it is the kind of bug that produces a
plausible wrong answer rather than an error.

**Two results are not bit-reproducible, by nature.** PoseCheck's strain energy embeds a conformer
ensemble without a fixed random seed, so a fresh run lands within a couple of percent rather than
exactly. And the toy sweeps train small networks on multi-threaded CPU, which is not deterministic; the
crossing points are stable, the third decimal place is not.

---

## Licensing

The models in this appendix are licensed separately for code and for weights, and the weights are what
you redistribute. `data/model_inventory.json` records both for all nine, with the URL each licence was
read from and the date. `data/table_I_1_print_ready.md` is the same thing as a table.

Three worth knowing before you build on them:

- **FlowMol3**: the repository LICENSE is MIT and covers the source. The weights ship separately from a
  plain HTTP directory with no licence statement of any kind. Absence of a grant is more restrictive in
  effect than MIT.
- **DiffSBDD**: MIT code, CC-BY-4.0 weights. That adds attribution and indicate-changes obligations the
  MIT notice clause does not.
- **RFdiffusion**: the cleanest case. Its BSD licence states explicitly that it covers both the source
  code and the model weights.

The two hosted NIM endpoints are a different surface again, governed by NVIDIA's service terms rather
than the repositories' MIT terms, and their trial tier is for internal evaluation rather than production.

---

## What is in `data/`

| file | what it is | read by |
|---|---|---|
| `toy_sampler_sweep.csv` | three samplers over ten budgets, five seeds | NB2 |
| `toy_eta_sweep.csv` | the stochasticity decomposition at a fixed budget | NB2 |
| `toy_ddpm_trained.pt`, `toy_fm_trained.pt` | the two trained toy networks, cached so the figures cannot drift | NB1, NB2 |
| `flowmol3_generated.sdf`, `flowmol3_metrics.csv` | the twelve generated molecules and their properties | NB3 |
| `posebusters_flowmol3.csv` | intramolecular validity, twelve molecules | NB3 |
| `posecheck_strain_flowmol3.csv` | UFF strain for the FlowMol3 batch | NB3 |
| `reos_flowmol3.csv` | REOS structural alerts, Glaxo rule set | NB3 |
| `novelty_flowmol3.csv` | similarity to GEOM-Drugs under two fingerprints | NB3 |
| `diffsbdd_generated.sdf`, `diffsbdd_metrics.csv` | the twenty pocket-conditioned ligands | NB4 |
| `diffsbdd_optimize_metrics.json`, `diffsbdd_optimized.sdf` | the QED optimization trajectory | NB4 |
| `diffsbdd_inpainted.sdf`, `diffsbdd_inpaint_metrics.json` | scaffold-constrained elaborations | NB4 |
| `posebusters_diffsbdd.csv` | pocket-aware validity against the full receptor | NB4, NB5 |
| `posecheck_diffsbdd.csv` | strain, clashes and interactions, with a caffeine reference row | NB4 |
| `novelty_diffsbdd.csv` | similarity to CrossDocked under two fingerprints | NB4 |
| `diversity_metadata.json` | the fingerprint definitions behind the diversity numbers | NB3, NB4 |
| `corpus_sample_crossdocked.smi`, `corpus_sample_geom_drugs.smi` | the sampled training corpora | NB3, NB4 |
| `boltz2_comparison.csv` | seven ligands, two independent Boltz-2 implementations | NB5 |
| `diffdock_redock_posebusters.csv` | the re-docking cross-check | NB5 |
| `diffdock_redocked_top_poses.sdf` | the re-docked poses, for your own inspection | NB5 |
| `selfconsistency_full.json`, `designability_scaling.json` | RFdiffusion self-consistency and length scaling | NB5 |
| `boltz_a2a_msa_summary.json`, `boltz_a2a_complex.cif` | the single-molecule Boltz-2 triage run | NB5 |
| `3rfm_protein.pdb`, `3rfm_caffeine_ref.sdf` | the A2A receptor and its co-crystallized caffeine | NB4, NB5 |
| `model_inventory.json`, `table_I_1_*.csv`, `table_I_1_print_ready.md` | licences, versions and observed runtimes | Table I.1 |
| `appendix_I_review_assets.json` | the keyed index of every number in the appendix | all |

The corpus samples deserve one line of provenance. They are seeded samples of documented public
preprocessings of CrossDocked2020 (CC0) and GEOM-Drugs (CC-BY-4.0), not the checkpoints' exact training
splits, which are not published as SMILES lists. Of the CrossDocked sample, 1,749 ligands are confirmed
present in DiffSBDD's own published training split, and that subset is the figure to quote if you want
one without a hedge.

---

## Figures

Every figure is in `figures/` as PNG at 300 dpi, SVG and PDF. Filenames match the stems the notebooks
embed, so a figure in the book, a figure in a notebook and a file on disk are always the same object.

Captions live in the notebooks and in the manuscript, never inside the image, so the same file works in
print, on the web and in a talk.

---

## `manuscript/`

The appendix draft with tracked changes, plus the three documents that explain them:

- `Appendix_I_Flynn_tracked.docx` &middot; accept or reject each revision in Word.
- `APPENDIX_I_REVIEW_FINDINGS.md` &middot; the review findings and what the measurements showed.
- `APPENDIX_I_INSERTION_GUIDE.md` &middot; each inserted sentence keyed to the asset-pack entry behind it.
- `DOCX_FIGURE_SWAP_AND_RENUMBER_GUIDE.md` &middot; the figure swaps, the new figures and the numbering options.

This directory is for the author's workflow. Drop it before publishing the repository if you would rather
not ship the draft.
