# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# <table style="border:none;border-collapse:collapse;margin:0 0 8px;width:100%;">
# <tr style="border:none;">
#   <td style="border:none;padding:0;vertical-align:middle;">
#     <div style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:10.5px;letter-spacing:.2em;text-transform:uppercase;color:#A20025;margin:0 0 7px;">Companion notebook</div>
#     <div style="font-family:Georgia,'Iowan Old Style',serif;font-size:18px;color:#1A1416;line-height:1.25;margin:0 0 7px;">
#       <a href="https://www.manning.com/books/build-ai-drug-discovery-pipelines" style="color:#1A1416;text-decoration:none;border-bottom:1px solid #A20025;padding-bottom:1px;">Build AI Drug Discovery Pipelines</a>
#     </div>
#     <div style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:12px;color:#6B6461;line-height:1.7;">
#       Noah Flynn&nbsp;&middot;&nbsp;Manning Publications<br>
#       Code&nbsp;&middot;&nbsp;<a href="https://github.com/nrflynn2/ml-drug-discovery" style="color:#A20025;text-decoration:none;">github.com/nrflynn2/ml-drug-discovery</a>
#     </div>
#   </td>
#   <td style="border:none;padding:0 0 0 22px;vertical-align:middle;width:104px;text-align:right;">
#     <a href="https://www.manning.com/books/build-ai-drug-discovery-pipelines">
#       <img src="https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/assets/cover-sm.png" width="88" alt="Build AI Drug Discovery Pipelines cover" style="border:1px solid #E7E2DF;border-radius:3px;display:inline-block;">
#     </a>
#   </td>
# </tr>
# </table>
# <hr style="border:0;border-top:1px solid #A20025;opacity:.55;margin:0 0 22px;">

# %% [markdown]
# <div style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.24em;text-transform:uppercase;color:#A20025;margin:0 0 12px;">Appendix C</div>
#
# <h1 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:33px;line-height:1.12;margin:0 0 16px;color:#1A1416;letter-spacing:-.01em;">Knowledge Distillation: Shrinking Models for Efficient, Hierarchical Molecular Generation</h1>
#
# <hr style="border:0;border-top:1px solid #A20025;opacity:.55;margin:0 0 4px;">

# %% [markdown]
# <em style="color:#6B6461;font-size:14.5px;">Runnable code for Appendix C; read alongside the book.</em>
#
# <a href="https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/APPENDIX_C_FLYNN_ML4DD.ipynb" target="_blank"><img src="https://img.shields.io/badge/Open%20in-Colab-A20025?logo=googlecolab&logoColor=white" alt="Open in Colab" style="height:20px;vertical-align:middle;"></a>

# %% [markdown]
# <strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#A20025;">Objectives</strong>
# <ol style="margin:8px 0 0;padding-left:22px;font-size:15.5px;color:#241c1d;">
# <li style="margin:6px 0;">Load and freeze a pre-trained generative teacher (HierVAE) and read its checkpoint format</li>
# <li style="margin:6px 0;">Derive a compressed student from the same architecture, reasoning about which dimensions can shrink and which cannot</li>
# <li style="margin:6px 0;">Compose a multi-term distillation objective (reconstruction, KL, latent-space matching, feature matching) and anneal the KL weight to avoid posterior collapse</li>
# <li style="margin:6px 0;">Handle the dimension mismatch between a teacher and a smaller student's latent space</li>
# <li style="margin:6px 0;">Evaluate a generative model on validity, uniqueness, reconstruction accuracy, generation speed, and property distributions, and read the trade-off honestly</li>
# </ol>

# %% [markdown]
# <div style="margin:0 0 16px;padding:2px 0 2px 18px;border-left:2px solid #B4690E;">
# <strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:#B4690E;">Caution</strong>
# <div style="color:#3d3436;font-size:14.5px;"><p><strong>A CUDA GPU is required.</strong> The <code>hgraph2graph</code> reference implementation calls <code>.cuda()</code>
# unconditionally in 21 places, so there is <strong>no CPU fallback</strong>; this appendix cannot run on a
# CPU-only machine at any batch size. On Colab: <em>Runtime &gt; Change runtime type &gt; T4 GPU</em>.
# <strong>Runtime and footprint:</strong> the clone is ~420 MB (the pre-trained checkpoint and the 1.8M-molecule
# ChEMBL file dominate). At the settings below (5,000 sampled molecules, 20 epochs) training takes
# <strong>~85 minutes on a T4</strong> (~4m15s/epoch) and the evaluation adds ~12 minutes. Both models are small
# - 7.96M + 3.05M parameters; so GPU memory is not the constraint; wall-clock time is. To explore
# quickly, lower <code>NUM_EPOCHS</code> and the <code>max_samples</code> passed to <code>load_training_data</code>.</p></div>
# </div>

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§0&#8195;</span>Environment Setup</h2>

# %% [markdown]
# <div style="font-size:14px;margin:0 0 6px;">
# <div style="display:flex;gap:14px;padding:9px 0;border-bottom:1px solid #EFEBE8;"><strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#A20025;min-width:96px;">Local&#8195;&#8195;</strong><span>Install <a href="https://docs.astral.sh/uv/" style="color:#A20025;">uv</a>, then run <code>uv sync --extra advanced</code> (Python 3.12). A CUDA GPU is required; there is no CPU fallback.</span></div>
# <div style="display:flex;gap:14px;padding:9px 0;"><strong style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#A20025;min-width:96px;">Colab&#8195;&#8195;</strong><span>Choose a T4 GPU runtime (Runtime &middot; Change runtime type), run the install cell below, then restart the runtime.</span></div>
# </div>

# %% tags=["skip-execution"]
# Colab users only - download the book's shared helper module.
# utils.py provides check_env() and set_seed(), used in the setup cell below. Locally it
# already sits at the repo root; on Colab it has to be fetched.
from pathlib import Path
import urllib.request

def fetch(url: str, dest, force: bool = False) -> Path:
    """Download `url` to `dest` if not already present. Cross-platform, no shell."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if force or not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest

fetch("https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/utils.py", "utils.py")

# %% tags=["skip-execution"]
# Colab users only - install this chapter's packages
# This installs just what's needed for this chapter
# !pip install -q -r https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/requirements-advanced.txt

# %% [markdown]
# We'll also make use of the official hgraph2graph implementation to load the teacher model and provide some additional utilities.
#
# The clone supplies three things this appendix depends on: the `hgraph` package itself
# (`HierVAE`, `PairVocab`, `MolGraph`), the **pre-trained teacher checkpoint** under `ckpt/`, and
# the ChEMBL vocabulary and SMILES under `data/`.

# %%
# `test -d` guard: git clone fails if the target directory already exists, so this keeps the
# cell safe to re-run.
# !test -d hgraph2graph || git clone https://github.com/wengong-jin/hgraph2graph.git

# %%
# hgraph resolves its checkpoint (ckpt/) and data (data/) relative to the clone's root, so we
# work from inside the clone for the rest of the notebook.
#
# os.chdir rather than the `%cd` magic: it runs headless (nbconvert), and the guard makes it
# idempotent - re-running this cell will not descend into hgraph2graph/hgraph2graph/.
import os
import sys

HGRAPH_DIR = "hgraph2graph"

# The book's shared helpers (utils.py) live at the repo root, which is about to stop being the
# working directory. Remember it, and keep it importable.
REPO_ROOT = os.getcwd()
if os.path.basename(REPO_ROOT) == HGRAPH_DIR:  # this cell has already been run
    REPO_ROOT = os.path.dirname(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(os.path.join(REPO_ROOT, HGRAPH_DIR))
print("Working directory:", os.getcwd())

# %% [markdown]
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Import Packages</h3>

# %% [markdown]
# Now let's import all the packages we'll need for this chapter.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import rdkit.Chem.Draw as Draw

import numpy as np
import matplotlib.pyplot as plt

import os
import random
import time
from tqdm.auto import tqdm
from collections import defaultdict
import json

# Import hgraph modules
from hgraph import (
    HierVAE,
    PairVocab,
    common_atom_vocab,
    MolGraph,
)

# Shared helpers (see utils.py at the book repo root, which the cell above put on sys.path)
from utils import check_env, set_seed

# Suppress RDKit warnings for cleaner output
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

# Set random seeds for reproducibility. set_seed() seeds Python, NumPy, torch and CUDA, plus
# PYTHONHASHSEED, and returns the seed so it can be reused downstream.
SEED = set_seed(42)

print("=" * 80)
print("Knowledge Distillation for Hierarchical Molecular Generation")
print("=" * 80)
check_env(["numpy", "torch", "rdkit", "matplotlib"])
print("=" * 80)

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§1&#8195;</span>Configuration and Hyperparameters</h2>

# %% [markdown]
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Teacher Model Configuration</h3>
#
# The teacher model is a pre-trained HierVAE with the following architecture:
# - Hidden size: 250 (encoder/decoder hidden dimension)
# - Embedding size: 250 (motif embedding dimension)
# - Latent size: 32 (VAE latent space dimension)
# - Tree depth: 15 (message passing iterations for motif tree)
# - Graph depth: 15 (message passing iterations for molecular graph)
#
# This model was trained on ~1.8M molecules from ChEMBL and achieves >95% reconstruction
# accuracy and >98% validity on generated molecules.
#
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Student Model Configuration</h3>
#
# To create an efficient student model, we reduce the capacity by:
# - Hidden size: 128 (50% reduction)
# - Embedding size: 128 (50% reduction)
# - Latent size: 16 (50% reduction)
# - Same tree/graph depth (architectural constraint)
#
# This results in approximately **60% fewer parameters** while maintaining the same architectural inductive biases.

# %%
class Config:
    """Configuration class for knowledge distillation pipeline"""

    # Paths
    TEACHER_MODEL_PATH = "ckpt/chembl-pretrained/model.ckpt"
    VOCAB_PATH = "data/chembl/vocab.txt"
    OUTPUT_DIR = "ckpt/distillation"

    # Teacher architecture (from pre-trained model)
    TEACHER_HIDDEN_SIZE = 250
    TEACHER_EMBED_SIZE = 250
    TEACHER_LATENT_SIZE = 32
    TEACHER_DEPTH_T = 15
    TEACHER_DEPTH_G = 15

    # Student architecture (compressed)
    STUDENT_HIDDEN_SIZE = 128
    STUDENT_EMBED_SIZE = 128
    STUDENT_LATENT_SIZE = 16
    STUDENT_DEPTH_T = 15  # Keep same for architectural compatibility
    STUDENT_DEPTH_G = 15

    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    WARMUP_EPOCHS = 5

    # Distillation loss weights
    ALPHA_RECONSTRUCTION = 1.0    # Weight for reconstruction loss
    BETA_KL = 0.1                 # Weight for KL divergence loss
    GAMMA_LATENT = 0.3            # Weight for latent space distillation
    DELTA_FEATURE = 0.2           # Weight for feature matching loss

    # KL annealing (gradual increase of KL weight)
    KL_ANNEAL_START = 0.0
    KL_ANNEAL_RATE = 0.01

    # Training settings
    CLIP_NORM = 5.0
    SAVE_EVERY = 5  # Save checkpoint every N epochs
    EVAL_EVERY = 1  # Evaluate every N epochs

    # Evaluation settings
    NUM_EVAL_SAMPLES = 1000
    NUM_GEN_SAMPLES = 1000

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # RNN type
    RNN_TYPE = 'LSTM'
    DROPOUT = 0.0

config = Config()

# hgraph2graph is CUDA-only: the reference implementation calls .cuda() unconditionally in 21
# places (the identity matrices in hgraph/encoder.py, the reparameterisation noise in
# hgraph/hgnn.py, the decoder's index tensors, ...). There is therefore no working CPU path, and
# DEVICE would only ever be 'cpu' on a machine that cannot run this appendix at all. Fail here,
# with a message, rather than deep inside the library.
if not torch.cuda.is_available():
    raise RuntimeError(
        "Appendix C requires a CUDA GPU. The hgraph2graph reference implementation calls "
        ".cuda() unconditionally, so there is no CPU fallback for HierVAE. "
        "On Colab: Runtime > Change runtime type > T4 GPU."
    )

# Create output directory
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# Save configuration
config_dict = {k: str(v) if isinstance(v, torch.device) else v
               for k, v in config.__dict__.items() if not k.startswith('_')}
with open(os.path.join(config.OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config_dict, f, indent=2)

print("\nConfiguration saved to:", os.path.join(config.OUTPUT_DIR, 'config.json'))

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§2&#8195;</span>Teacher Model Loading</h2>

# %% [markdown]
# The teacher model is loaded from a pre-trained checkpoint. We freeze all teacher
# parameters since we only use it for guidance during student training.
#
# The checkpoint is saved as a tuple: (model_state_dict, optimizer_state_dict, step, beta)
# We only need the model_state_dict for inference.

# %%
class TeacherModelLoader:
    """Utility class for loading and preparing the teacher model"""

    @staticmethod
    def load_vocab(vocab_path):
        """Load vocabulary from file

        Args:
            vocab_path: Path to vocabulary file

        Returns:
            PairVocab: Vocabulary object for motif encoding
        """
        print(f"\nLoading vocabulary from: {vocab_path}")
        with open(vocab_path) as f:
            vocab_list = [x.strip("\r\n ").split() for x in f]

        vocab = PairVocab(vocab_list)
        print(f"Vocabulary loaded: {vocab.size()[0]} cluster types, {vocab.size()[1]} motifs")

        return vocab

    @staticmethod
    def create_teacher_args(config, vocab):
        """Create arguments object for teacher model

        Args:
            config: Configuration object
            vocab: PairVocab object

        Returns:
            Namespace: Arguments for HierVAE initialization
        """
        from argparse import Namespace

        args = Namespace(
            vocab=vocab,
            atom_vocab=common_atom_vocab,
            rnn_type=config.RNN_TYPE,
            hidden_size=config.TEACHER_HIDDEN_SIZE,
            embed_size=config.TEACHER_EMBED_SIZE,
            latent_size=config.TEACHER_LATENT_SIZE,
            depthT=config.TEACHER_DEPTH_T,
            depthG=config.TEACHER_DEPTH_G,
            diterT=1,
            diterG=3,
            dropout=config.DROPOUT
        )

        return args

    @staticmethod
    def load_teacher_model(config, vocab):
        """Load pre-trained teacher model

        Args:
            config: Configuration object
            vocab: PairVocab object

        Returns:
            HierVAE: Loaded teacher model in eval mode
        """
        print(f"\nLoading teacher model from: {config.TEACHER_MODEL_PATH}")

        # Create model
        teacher_args = TeacherModelLoader.create_teacher_args(config, vocab)
        teacher_model = HierVAE(teacher_args).to(config.DEVICE)

        # Load checkpoint
        checkpoint = torch.load(config.TEACHER_MODEL_PATH, map_location=config.DEVICE)
        if isinstance(checkpoint, tuple):
            # Format: (model_state, optimizer_state, step, beta)
            model_state = checkpoint[0]
        else:
            model_state = checkpoint

        teacher_model.load_state_dict(model_state)

        # Set to evaluation mode and freeze parameters
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        # Count parameters
        num_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"Teacher model loaded successfully!")
        print(f"Teacher parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        return teacher_model

# Load vocabulary and teacher model
vocab = TeacherModelLoader.load_vocab(config.VOCAB_PATH)
teacher_model = TeacherModelLoader.load_teacher_model(config, vocab)

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§3&#8195;</span>Student Model Architecture</h2>

# %% [markdown]
# The student model uses the same HierVAE architecture but with reduced capacity.
# This is crucial - we want the student to learn the same inductive biases as the teacher
# (hierarchical decomposition, message passing) but with a compressed representation.
#
# Architectural decisions:
# 1. **Same depth**: Keep tree/graph depths constant to maintain receptive field
# 2. **Reduce hidden dimensions**: Primary source of parameter reduction
# 3. **Reduce latent dimension**: Smaller bottleneck for efficiency
# 4. **Same RNN type**: Maintain temporal modeling capacity
#
# This ensures the student learns the same type of representations, just in a more
# compressed form.

# %%
class StudentModelFactory:
    """Factory for creating student models with various configurations"""

    @staticmethod
    def create_student_args(config, vocab):
        """Create arguments for student model

        Args:
            config: Configuration object
            vocab: PairVocab object

        Returns:
            Namespace: Arguments for student HierVAE
        """
        from argparse import Namespace

        args = Namespace(
            vocab=vocab,
            atom_vocab=common_atom_vocab,
            rnn_type=config.RNN_TYPE,
            hidden_size=config.STUDENT_HIDDEN_SIZE,
            embed_size=config.STUDENT_EMBED_SIZE,
            latent_size=config.STUDENT_LATENT_SIZE,
            depthT=config.STUDENT_DEPTH_T,
            depthG=config.STUDENT_DEPTH_G,
            diterT=1,
            diterG=3,
            dropout=config.DROPOUT
        )

        return args

    @staticmethod
    def create_student_model(config, vocab):
        """Create student model

        Args:
            config: Configuration object
            vocab: PairVocab object

        Returns:
            HierVAE: Initialized student model
        """
        print("\nCreating student model...")

        student_args = StudentModelFactory.create_student_args(config, vocab)
        student_model = HierVAE(student_args).to(config.DEVICE)

        # Initialize parameters, matching hgraph2graph's own train_generator.py: zero the 1-D
        # parameters, Xavier-initialise the rest.
        #
        # Zeroing every 1-D parameter is only safe because of what HierVAE is made of. The
        # architecture contains no normalisation layers at all, so all 46 of the student's 1-D
        # parameters are biases - and zero is the standard initialisation for a bias. In an
        # architecture with LayerNorm/BatchNorm this loop would also zero the norm *scale*
        # parameters, which multiply their input: the layer would output a constant 0 and the
        # model would start out unable to propagate signal. If you swap in such an architecture,
        # switch to a name-based check (zero `bias`, set `norm.weight` to 1).
        for param in student_model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

        # Count parameters
        num_params = sum(p.numel() for p in student_model.parameters())
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        reduction = (1 - num_params / teacher_params) * 100

        print(f"Student model created successfully!")
        print(f"Student parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"Parameter reduction: {reduction:.1f}%")

        return student_model

    @staticmethod
    def compare_architectures(teacher_model, student_model):
        """Print detailed comparison of teacher and student architectures

        Args:
            teacher_model: Teacher HierVAE
            student_model: Student HierVAE
        """
        print("\n" + "=" * 80)
        print("ARCHITECTURE COMPARISON")
        print("=" * 80)

        def count_module_params(module):
            return sum(p.numel() for p in module.parameters())

        components = [
            ('Encoder', 'encoder'),
            ('Decoder', 'decoder'),
            ('Latent Projection', 'R_mean'),
        ]

        print(f"\n{'Component':<20} {'Teacher':>15} {'Student':>15} {'Reduction':>15}")
        print("-" * 80)

        for name, attr in components:
            teacher_params = count_module_params(getattr(teacher_model, attr))
            student_params = count_module_params(getattr(student_model, attr))
            reduction = (1 - student_params / teacher_params) * 100

            print(f"{name:<20} {teacher_params:>15,} {student_params:>15,} {reduction:>14.1f}%")

        # Total
        teacher_total = sum(p.numel() for p in teacher_model.parameters())
        student_total = sum(p.numel() for p in student_model.parameters())
        total_reduction = (1 - student_total / teacher_total) * 100

        print("-" * 80)
        print(f"{'TOTAL':<20} {teacher_total:>15,} {student_total:>15,} {total_reduction:>14.1f}%")
        print("=" * 80)

# Create student model
student_model = StudentModelFactory.create_student_model(config, vocab)

# Compare architectures
StudentModelFactory.compare_architectures(teacher_model, student_model)

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§4&#8195;</span>Knowledge Distillation Loss Functions</h2>

# %% [markdown]
# Knowledge distillation combines multiple loss objectives to transfer knowledge from
# teacher to student. Each loss term captures different aspects of the teacher's behavior.
#
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Loss Components</h3>
#
# 1. **Reconstruction Loss**: Standard VAE reconstruction objective
#    - Ensures student can encode/decode molecules
#    - Measured as negative log-likelihood of output
#
# 2. **KL Divergence Loss**: VAE regularization term
#    - Keeps latent space well-structured
#    - Enables sampling from prior distribution
#
# 3. **Latent Space Distillation**: Align latent representations
#    - MSE between teacher and student latent vectors
#    - Transfers semantic organization of latent space
#
# 4. **Feature Matching**: Match intermediate representations
#    - MSE between encoder hidden states
#    - Transfers learned feature hierarchies
#
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Mathematical Formulation</h3>
#
# Total loss: L = α·L_recon + β·L_KL + γ·L_latent + δ·L_feature
#
# Where:
# - α, β, γ, δ are weighting hyperparameters
# - β is annealed from 0 to final value during training

# %%
class DistillationLoss(nn.Module):
    """Combined loss function for knowledge distillation"""

    def __init__(self, config):
        super(DistillationLoss, self).__init__()
        self.config = config

    def compute_latent_distillation_loss(self, student_latent, teacher_latent):
        """Compute MSE loss between student and teacher latent representations

        This loss encourages the student's latent space to have similar structure
        to the teacher's, enabling better generation quality.

        Args:
            student_latent: Student latent vectors [batch_size, latent_dim]
            teacher_latent: Teacher latent vectors [batch_size, latent_dim]

        Returns:
            torch.Tensor: MSE loss value
        """
        # Student latent dim may be smaller, so we need to handle dimension mismatch
        # Option 1: Project student to teacher dimension (adds parameters)
        # Option 2: Project teacher to student dimension
        # Option 3: Use first N dimensions of teacher

        # We'll use Option 3 for simplicity (no additional parameters).
        #
        # This is a deliberate tutorial simplification, and it is not free: with the default
        # sizes the teacher latent is 32-d and the student's is 16-d, so half of the teacher's
        # latent vector is discarded rather than distilled. The teacher's dimensions are not
        # ordered by importance (nothing in the VAE objective makes dim 0 more informative than
        # dim 31), so this keeps an arbitrary half. A learned projection -
        # nn.Linear(teacher_dim, student_dim) trained alongside the student - would use all of
        # it at the cost of extra parameters.
        teacher_latent_reduced = teacher_latent[:, :student_latent.size(1)]

        loss = F.mse_loss(student_latent, teacher_latent_reduced)
        return loss

    def compute_feature_matching_loss(self, student_features, teacher_features):
        """Compute MSE loss between intermediate encoder representations

        Feature matching helps transfer the learned feature hierarchies from teacher
        to student, improving representation quality.

        Args:
            student_features: Dict of student encoder outputs
            teacher_features: Dict of teacher encoder outputs

        Returns:
            torch.Tensor: Feature matching loss
        """
        losses = []

        # Match root representations
        if 'root_vecs' in student_features and 'root_vecs' in teacher_features:
            student_root = student_features['root_vecs']
            teacher_root = teacher_features['root_vecs']

            # Project to common dimension (use smaller dimension)
            min_dim = min(student_root.size(-1), teacher_root.size(-1))
            loss_root = F.mse_loss(
                student_root[..., :min_dim],
                teacher_root[..., :min_dim]
            )
            losses.append(loss_root)

        # Average all feature matching losses
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=self.config.DEVICE)

    def forward(self, student_outputs, teacher_outputs, beta):
        """Compute combined distillation loss

        Args:
            student_outputs: Dict containing student model outputs:
                - 'recon_loss': Reconstruction loss
                - 'kl_loss': KL divergence loss
                - 'latent_mean': Latent mean vector
                - 'features': Intermediate representations
            teacher_outputs: Dict containing teacher model outputs (same structure)
            beta: Current KL annealing weight

        Returns:
            dict: Dictionary containing all loss components
        """
        # Extract base losses
        student_recon = student_outputs['recon_loss']
        student_kl = student_outputs['kl_loss']

        # Compute distillation-specific losses
        latent_distill_loss = self.compute_latent_distillation_loss(
            student_outputs['latent_mean'],
            teacher_outputs['latent_mean']
        )

        feature_match_loss = self.compute_feature_matching_loss(
            student_outputs['features'],
            teacher_outputs['features']
        )

        # Combine losses with weights:
        #     L = alpha * L_recon + beta * L_KL + gamma * L_latent + delta * L_feature
        #
        # `beta` is already the annealed KL weight - compute_beta() ramps it from KL_ANNEAL_START
        # up to config.BETA_KL and returns config.BETA_KL thereafter. Multiplying by
        # config.BETA_KL again here would apply the weight twice (0.1 * 0.1 = 0.01), i.e. a tenth
        # of the documented KL weight.
        total_loss = (
            self.config.ALPHA_RECONSTRUCTION * student_recon +
            beta * student_kl +
            self.config.GAMMA_LATENT * latent_distill_loss +
            self.config.DELTA_FEATURE * feature_match_loss
        )

        return {
            'total': total_loss,
            'reconstruction': student_recon.item(),
            'kl': student_kl.item(),
            'latent_distill': latent_distill_loss.item(),
            'feature_match': feature_match_loss.item(),
        }

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§5&#8195;</span>Data Loading and Preparation</h2>

# %% [markdown]
# For distillation training, we need to:
# 1. Load molecular data (SMILES strings)
# 2. Perform hierarchical decomposition (motif extraction)
# 3. Create batches of tensorized molecules
# 4. Handle data augmentation (optional)
#
# We can either use:
# - Pre-processed tensor files (faster)
# - Raw SMILES files (more flexible)

# %%
class MoleculeDatasetForDistillation(Dataset):
    """Dataset for loading molecules for knowledge distillation training"""

    def __init__(self, smiles_list, vocab, atom_vocab, max_samples=None):
        """Initialize dataset

        Args:
            smiles_list: List of SMILES strings
            vocab: PairVocab object
            atom_vocab: Atom vocabulary
            max_samples: Maximum number of samples to use (for debugging)
        """
        self.vocab = vocab
        self.atom_vocab = atom_vocab

        # Filter valid molecules
        print(f"\nProcessing {len(smiles_list)} molecules...")
        self.valid_smiles = []

        for smiles in tqdm(smiles_list[:max_samples] if max_samples else smiles_list):
            try:
                # Try to create MolGraph (validates against vocabulary)
                hmol = MolGraph(smiles)

                # Check if all motifs are in vocabulary
                valid = True
                for node, attr in hmol.mol_tree.nodes(data=True):
                    if attr['label'] not in vocab.vmap:
                        valid = False
                        break
                    for i, s in attr['inter_label']:
                        if (attr['smiles'], s) not in vocab.vmap:
                            valid = False
                            break

                if valid:
                    self.valid_smiles.append(smiles)

            except Exception:
                # Skip molecules that fail processing
                continue

        print(f"Valid molecules: {len(self.valid_smiles)} / {len(smiles_list)}")

    def __len__(self):
        return len(self.valid_smiles)

    def __getitem__(self, idx):
        """Get single molecule

        Returns:
            str: SMILES string
        """
        return self.valid_smiles[idx]

def collate_fn_distillation(smiles_batch, vocab, atom_vocab):
    """Collate function for DataLoader

    Args:
        smiles_batch: List of SMILES strings
        vocab: PairVocab object
        atom_vocab: Atom vocabulary

    Returns:
        Tensorized batch
    """
    return MolGraph.tensorize(smiles_batch, vocab, atom_vocab)

def load_training_data(config, vocab, max_samples=5000):
    """Load training data for distillation

    Args:
        config: Configuration object
        vocab: PairVocab object
        max_samples: Maximum number of samples (for quick testing)

    Returns:
        DataLoader: Training data loader
    """
    print("\n" + "=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)

    # Load SMILES from file
    data_file = "data/chembl/all.txt"
    print(f"\nLoading SMILES from: {data_file}")

    with open(data_file, 'r') as f:
        all_smiles = [line.strip() for line in f if line.strip()]

    print(f"Total SMILES loaded: {len(all_smiles)}")

    # Sample subset for training (for demonstration purposes)
    if max_samples and max_samples < len(all_smiles):
        print(f"Sampling {max_samples} molecules for training...")
        sampled_smiles = random.sample(all_smiles, max_samples)
    else:
        sampled_smiles = all_smiles

    # Create dataset
    dataset = MoleculeDatasetForDistillation(
        sampled_smiles,
        vocab,
        common_atom_vocab,
        max_samples=max_samples
    )

    # Create data loader
    from functools import partial
    collate_fn = partial(collate_fn_distillation, vocab=vocab, atom_vocab=common_atom_vocab)

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for compatibility
    )

    print(f"\nDataLoader created:")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Number of batches: {len(dataloader)}")

    return dataloader

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§6&#8195;</span>Training Loop</h2>

# %% [markdown]
# The training loop implements knowledge distillation by:
# 1. Running both teacher and student on the same input
# 2. Computing combined distillation loss
# 3. Updating only student parameters
# 4. Annealing KL weight over time
# 5. Monitoring training progress
#
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">KL Annealing Strategy</h3>
#
# We gradually increase the KL divergence weight from 0 to its final value. This helps:
# - Initially focus on reconstruction (learning basic encoding/decoding)
# - Gradually enforce latent space structure
# - Avoid posterior collapse (common in VAEs)

# %%
class DistillationTrainer:
    """Trainer class for knowledge distillation"""

    def __init__(self, config, teacher_model, student_model, vocab):
        self.config = config
        self.teacher = teacher_model
        self.student = student_model
        self.vocab = vocab

        # Loss function
        self.criterion = DistillationLoss(config)

        # Optimizer
        self.optimizer = optim.Adam(
            student_model.parameters(),
            lr=config.LEARNING_RATE
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.95
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        # History tracking
        self.history = defaultdict(list)

    def compute_beta(self, epoch, step, steps_per_epoch):
        """Compute current KL annealing weight

        Args:
            epoch: Current epoch
            step: Current step within epoch
            steps_per_epoch: Total steps per epoch

        Returns:
            float: Current beta value
        """
        total_steps = epoch * steps_per_epoch + step
        warmup_steps = self.config.WARMUP_EPOCHS * steps_per_epoch

        # WARMUP_EPOCHS = 0 (no annealing) is safe and needs no special case: warmup_steps is
        # then 0, `total_steps < 0` is never true, and the division below is skipped in favour
        # of the constant beta.
        if total_steps < warmup_steps:
            # Linear warmup
            beta = self.config.KL_ANNEAL_START + (
                self.config.BETA_KL - self.config.KL_ANNEAL_START
            ) * (total_steps / warmup_steps)
        else:
            beta = self.config.BETA_KL

        return beta

    def extract_features(self, model, batch):
        """Extract features from model for distillation

        Args:
            model: HierVAE model
            batch: Batch of molecules

        Returns:
            dict: Dictionary containing model outputs and features
        """
        graphs, tensors, orders = batch

        # Move to device
        tree_tensors, graph_tensors = tensors
        tree_tensors = [t.to(self.config.DEVICE) if isinstance(t, torch.Tensor) else t for t in tree_tensors]
        graph_tensors = [t.to(self.config.DEVICE) if isinstance(t, torch.Tensor) else t for t in graph_tensors]
        tensors = (tree_tensors, graph_tensors)

        # Encode
        root_vecs, tree_vecs, inter_vecs, graph_vecs = model.encoder(tree_tensors, graph_tensors)

        # Sample latent
        latent_mean = model.R_mean(root_vecs)
        latent_log_var = -torch.abs(model.R_var(root_vecs))

        # Compute KL loss
        batch_size = root_vecs.size(0)
        kl_loss = -0.5 * torch.sum(
            1.0 + latent_log_var - latent_mean * latent_mean - torch.exp(latent_log_var)
        ) / batch_size

        # Sample latent vector
        epsilon = torch.randn_like(latent_mean)
        latent_vec = latent_mean + torch.exp(latent_log_var / 2) * epsilon

        # Decode
        dec_loss, wacc, iacc, tacc, sacc = model.decoder(
            (latent_vec, latent_vec, latent_vec),
            graphs,
            tensors,
            orders
        )

        return {
            'recon_loss': dec_loss,
            'kl_loss': kl_loss,
            'latent_mean': latent_mean,
            'latent_vec': latent_vec,
            'features': {
                'root_vecs': root_vecs,
                'tree_vecs': tree_vecs,
            },
            'accuracy': {
                'word': wacc,
                'inter': iacc,
                'topo': tacc,
                'assm': sacc
            }
        }

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            dict: Average losses for the epoch
        """
        self.student.train()
        self.teacher.eval()

        epoch_losses = defaultdict(float)
        steps_per_epoch = len(dataloader)

        # An empty dataloader (e.g. every sampled molecule was rejected by the vocabulary filter)
        # would otherwise fall straight through this method and surface one frame later as a
        # confusing `KeyError: 'total'` in train().
        if steps_per_epoch == 0:
            raise ValueError(
                "The training dataloader is empty - no batches to train on. Check that "
                "load_training_data() found molecules whose motifs are all in the vocabulary."
            )

        completed_steps = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")

        for step, batch in enumerate(pbar):
            # Compute current beta
            beta = self.compute_beta(epoch, step, steps_per_epoch)

            # Extract features from teacher (no gradient)
            with torch.no_grad():
                teacher_outputs = self.extract_features(self.teacher, batch)

            # Extract features from student (with gradient)
            student_outputs = self.extract_features(self.student, batch)

            # Compute distillation loss
            losses = self.criterion(student_outputs, teacher_outputs, beta)

            # Skip non-finite steps. Gradient clipping (below) bounds the magnitude of a large
            # but finite gradient; it cannot repair a NaN/inf one - an infinite total_norm makes
            # the clip coefficient 0, and 0 * nan is still nan, so the poison would propagate
            # into the weights on the next optimizer.step().
            if not torch.isfinite(losses['total']):
                print(f"  Skipping step {step}: non-finite loss ({losses['total'].item()})")
                self.optimizer.zero_grad()
                continue

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config.CLIP_NORM
            )

            self.optimizer.step()

            # Track losses
            for key, value in losses.items():
                if key != 'total':
                    epoch_losses[key] += value
                else:
                    epoch_losses[key] += value.item()
            completed_steps += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['reconstruction']:.4f}",
                'beta': f"{beta:.4f}"
            })

            self.global_step += 1

        if completed_steps == 0:
            raise ValueError(
                f"Every step in epoch {epoch + 1} produced a non-finite loss; nothing was "
                "learned. Try lowering LEARNING_RATE or CLIP_NORM."
            )

        # Average losses over the steps that actually contributed a gradient
        for key in epoch_losses:
            epoch_losses[key] /= completed_steps

        return dict(epoch_losses)

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': dict(self.history),
        }

        # Save regular checkpoint
        path = os.path.join(self.config.OUTPUT_DIR, f'student_model_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

        # Save best model
        if is_best:
            best_path = os.path.join(self.config.OUTPUT_DIR, 'best_student_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    def train(self, dataloader):
        """Main training loop

        Args:
            dataloader: Training data loader
        """
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)

        for epoch in range(self.config.NUM_EPOCHS):
            # Train one epoch
            losses = self.train_epoch(dataloader, epoch)

            # Update history
            for key, value in losses.items():
                self.history[key].append(value)
            self.history['epoch'].append(epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Total Loss: {losses['total']:.4f}")
            print(f"  Reconstruction: {losses['reconstruction']:.4f}")
            print(f"  KL: {losses['kl']:.4f}")
            print(f"  Latent Distill: {losses['latent_distill']:.4f}")
            print(f"  Feature Match: {losses['feature_match']:.4f}")

            # Check if best model
            is_best = losses['total'] < self.best_loss
            if is_best:
                self.best_loss = losses['total']

            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(epoch, is_best)

            # Learning rate scheduling
            if epoch >= self.config.WARMUP_EPOCHS:
                self.scheduler.step()
                print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")

        # Save final model
        self.save_checkpoint(self.config.NUM_EPOCHS - 1)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§7&#8195;</span>Evaluation Metrics</h2>

# %% [markdown]
# To assess the quality of the distilled model, we evaluate:
#
# 1. **Validity**: Percentage of generated molecules that are chemically valid
# 2. **Uniqueness**: Percentage of unique molecules in generated set
# 3. **Reconstruction Accuracy**: How well the model reconstructs input molecules
# 4. **Novelty**: Percentage of generated molecules not in training set
# 5. **Property Distributions**: Compare distributions of molecular properties
# 6. **Generation Speed**: Molecules generated per second
# 7. **Model Size**: Memory footprint comparison
#
# These metrics provide a comprehensive view of whether distillation preserved the
# teacher's capabilities while achieving the desired efficiency gains.

# %%
class ModelEvaluator:
    """Comprehensive evaluation suite for distilled models"""
    def __init__(self, config, teacher_model, student_model, vocab):
        self.config = config
        self.teacher = teacher_model
        self.student = student_model
        self.vocab = vocab

        # Track OOV (Out-of-Vocabulary) statistics
        self.oov_stats = {
            'teacher_oov_count': 0,
            'teacher_total_attempts': 0,
            'student_oov_count': 0,
            'student_total_attempts': 0,
            'oov_motifs': set()
        }

    def safe_sample(self, model, num_samples, model_name='model'):
        """Safely sample molecules with OOV handling

        Args:
            model: Model to sample from
            num_samples: Number of molecules to generate
            model_name: Name of model for tracking ('teacher' or 'student')

        Returns:
            list: Generated SMILES strings
        """
        generated_smiles = []
        oov_count = 0
        attempts = 0

        batch_size = min(self.config.BATCH_SIZE, 8)  # Smaller batches to isolate errors

        with torch.no_grad():
            pbar = tqdm(total=num_samples, desc=f"Generating {model_name} molecules")

            while len(generated_smiles) < num_samples and attempts < num_samples * 3:
                try:
                    current_batch_size = min(batch_size, num_samples - len(generated_smiles))
                    attempts += current_batch_size

                    smiles_batch = model.sample(current_batch_size, greedy=True)

                    for smiles in smiles_batch:
                        generated_smiles.append(smiles)
                        pbar.update(1)

                except KeyError as e:
                    # OOV motif encountered. `attempts` was already incremented in the try block
                    # above, before model.sample() was called - incrementing it again here would
                    # count every failed batch twice and deflate the reported OOV rate
                    # (oov_count / total_attempts).
                    oov_count += current_batch_size

                    # Extract motif from error message
                    motif_str = str(e)
                    self.oov_stats['oov_motifs'].add(motif_str)

                    # Try individual samples with smaller batch
                    if current_batch_size > 1:
                        batch_size = 1
                        continue

                except Exception:
                    # Other errors - skip this batch (already counted in `attempts`)
                    continue

            pbar.close()

        # Update statistics
        if model_name == 'teacher':
            self.oov_stats['teacher_oov_count'] = oov_count
            self.oov_stats['teacher_total_attempts'] = attempts
        else:
            self.oov_stats['student_oov_count'] = oov_count
            self.oov_stats['student_total_attempts'] = attempts

        return generated_smiles

    def evaluate_validity(self, model, num_samples=1000, model_name='model'):
        """Evaluate validity of generated molecules

        Args:
            model: Model to evaluate
            num_samples: Number of molecules to generate
            model_name: Name of model for tracking ('teacher' or 'student')

        Returns:
            dict: Validity metrics including OOV statistics
        """
        model.eval()

        # Generate molecules with OOV handling
        generated_smiles = self.safe_sample(model, num_samples, model_name)

        # Validate generated molecules
        valid_count = 0
        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1

        validity_rate = valid_count / len(generated_smiles) if generated_smiles else 0.0

        # Get OOV stats for this model
        if model_name == 'teacher':
            oov_count = self.oov_stats['teacher_oov_count']
            total_attempts = self.oov_stats['teacher_total_attempts']
        else:
            oov_count = self.oov_stats['student_oov_count']
            total_attempts = self.oov_stats['student_total_attempts']

        oov_rate = oov_count / total_attempts if total_attempts > 0 else 0.0

        return {
            'validity_rate': validity_rate,
            'valid_count': valid_count,
            'total_count': len(generated_smiles),
            'generated_smiles': generated_smiles,
            'oov_count': oov_count,
            'oov_rate': oov_rate,
            'total_attempts': total_attempts
        }

    def evaluate_uniqueness(self, generated_smiles):
        """Evaluate uniqueness of generated molecules

        Args:
            generated_smiles: List of generated SMILES

        Returns:
            dict: Uniqueness metrics
        """
        # Canonicalize SMILES
        canonical_smiles = []
        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles.append(Chem.MolToSmiles(mol))

        unique_smiles = set(canonical_smiles)
        uniqueness_rate = len(unique_smiles) / len(canonical_smiles) if canonical_smiles else 0

        return {
            'uniqueness_rate': uniqueness_rate,
            'unique_count': len(unique_smiles),
            'total_valid': len(canonical_smiles)
        }

    def evaluate_reconstruction(self, model, test_smiles, num_samples=100):
        """Evaluate reconstruction accuracy

        Args:
            model: Model to evaluate
            test_smiles: List of test SMILES
            num_samples: Number of samples to test

        Returns:
            dict: Reconstruction metrics
        """
        model.eval()

        test_sample = random.sample(test_smiles, min(num_samples, len(test_smiles)))

        exact_matches = 0
        valid_reconstructions = 0

        with torch.no_grad():
            for smiles in tqdm(test_sample, desc="Testing reconstruction"):
                try:
                    # Create batch of size 1
                    batch = MolGraph.tensorize([smiles], self.vocab, common_atom_vocab)

                    # Reconstruct
                    graphs, tensors, _ = batch
                    tree_tensors, graph_tensors = tensors
                    tree_tensors = [t.to(self.config.DEVICE) if isinstance(t, torch.Tensor) else t for t in tree_tensors]
                    graph_tensors = [t.to(self.config.DEVICE) if isinstance(t, torch.Tensor) else t for t in graph_tensors]
                    tensors = (tree_tensors, graph_tensors)

                    batch = (graphs, tensors, _)
                    reconstructed = model.reconstruct(batch)

                    if reconstructed and len(reconstructed) > 0:
                        recon_smiles = reconstructed[0]

                        # Check validity
                        mol = Chem.MolFromSmiles(recon_smiles)
                        if mol is not None:
                            valid_reconstructions += 1

                            # Check exact match (canonicalized)
                            orig_mol = Chem.MolFromSmiles(smiles)
                            if orig_mol is not None:
                                orig_canonical = Chem.MolToSmiles(orig_mol)
                                recon_canonical = Chem.MolToSmiles(mol)
                                if orig_canonical == recon_canonical:
                                    exact_matches += 1

                except Exception:
                    continue

        if not test_sample:
            raise ValueError("No test molecules were supplied to evaluate_reconstruction().")

        return {
            'exact_match_rate': exact_matches / len(test_sample),
            'valid_reconstruction_rate': valid_reconstructions / len(test_sample),
            'tested_count': len(test_sample)
        }

    def evaluate_generation_speed(self, model, num_samples=100, model_name='model'):
        """Measure generation speed

        Args:
            model: Model to evaluate
            num_samples: Number of molecules to generate

        Returns:
            dict: Speed metrics
        """
        model.eval()

        with torch.no_grad():
            # Warmup
            _ = self.safe_sample(model, num_samples, model_name)

            # Timed generation
            start_time = time.time()
            _ = self.safe_sample(model, num_samples, model_name)
            end_time = time.time()

        elapsed_time = end_time - start_time
        molecules_per_second = num_samples / elapsed_time

        return {
            'molecules_per_second': molecules_per_second,
            'total_time': elapsed_time,
            'num_samples': num_samples
        }

    def compute_property_statistics(self, smiles_list):
        """Compute molecular property statistics

        Args:
            smiles_list: List of SMILES strings

        Returns:
            dict: Property statistics
        """
        properties = {
            'mol_weight': [],
            'logp': [],
            'qed': [],
            'num_atoms': [],
            'num_rings': []
        }

        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                properties['mol_weight'].append(Descriptors.MolWt(mol))
                properties['logp'].append(Descriptors.MolLogP(mol))
                properties['qed'].append(QED.qed(mol))
                properties['num_atoms'].append(mol.GetNumAtoms())
                properties['num_rings'].append(Descriptors.RingCount(mol))

        # Compute statistics
        stats = {}
        for prop, values in properties.items():
            if values:
                stats[prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }

        return stats

    def comprehensive_evaluation(self, test_smiles):
        """Run comprehensive evaluation suite

        Args:
            test_smiles: List of test SMILES for reconstruction testing

        Returns:
            dict: Complete evaluation results
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 80)

        results = {}

        # Evaluate teacher
        print("\n--- TEACHER MODEL ---")

        print("\n1. Evaluating validity...")
        teacher_validity = self.evaluate_validity(
            self.teacher,
            num_samples=self.config.NUM_GEN_SAMPLES,
            model_name='teacher'
        )
        results['teacher_validity'] = teacher_validity
        print(f"   Validity: {teacher_validity['validity_rate']*100:.2f}%")

        print("\n2. Evaluating uniqueness...")
        teacher_uniqueness = self.evaluate_uniqueness(
            teacher_validity['generated_smiles']
        )
        results['teacher_uniqueness'] = teacher_uniqueness
        print(f"   Uniqueness: {teacher_uniqueness['uniqueness_rate']*100:.2f}%")

        print("\n3. Evaluating reconstruction...")
        teacher_reconstruction = self.evaluate_reconstruction(
            self.teacher,
            test_smiles,
            num_samples=100
        )
        results['teacher_reconstruction'] = teacher_reconstruction
        print(f"   Exact match: {teacher_reconstruction['exact_match_rate']*100:.2f}%")
        print(f"   Valid reconstruction: {teacher_reconstruction['valid_reconstruction_rate']*100:.2f}%")

        print("\n4. Measuring generation speed...")
        teacher_speed = self.evaluate_generation_speed(self.teacher, num_samples=100, model_name='teacher')
        results['teacher_speed'] = teacher_speed
        print(f"   Speed: {teacher_speed['molecules_per_second']:.2f} molecules/sec")

        # Evaluate student
        print("\n--- STUDENT MODEL ---")

        print("\n1. Evaluating validity...")
        student_validity = self.evaluate_validity(
            self.student,
            num_samples=self.config.NUM_GEN_SAMPLES,
            model_name='student'
        )
        results['student_validity'] = student_validity
        print(f"   Validity: {student_validity['validity_rate']*100:.2f}%")

        print("\n2. Evaluating uniqueness...")
        student_uniqueness = self.evaluate_uniqueness(
            student_validity['generated_smiles']
        )
        results['student_uniqueness'] = student_uniqueness
        print(f"   Uniqueness: {student_uniqueness['uniqueness_rate']*100:.2f}%")

        print("\n3. Evaluating reconstruction...")
        student_reconstruction = self.evaluate_reconstruction(
            self.student,
            test_smiles,
            num_samples=100
        )
        results['student_reconstruction'] = student_reconstruction
        print(f"   Exact match: {student_reconstruction['exact_match_rate']*100:.2f}%")
        print(f"   Valid reconstruction: {student_reconstruction['valid_reconstruction_rate']*100:.2f}%")

        print("\n4. Measuring generation speed...")
        student_speed = self.evaluate_generation_speed(self.student, num_samples=100, model_name='student')
        results['student_speed'] = student_speed
        print(f"   Speed: {student_speed['molecules_per_second']:.2f} molecules/sec")

        # Compute property statistics
        print("\n5. Computing property distributions...")
        teacher_props = self.compute_property_statistics(
            [s for s in teacher_validity['generated_smiles'] if Chem.MolFromSmiles(s)]
        )
        student_props = self.compute_property_statistics(
            [s for s in student_validity['generated_smiles'] if Chem.MolFromSmiles(s)]
        )
        results['teacher_properties'] = teacher_props
        results['student_properties'] = student_props

        # Print comparison
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        print(f"\n{'Metric':<30} {'Teacher':>15} {'Student':>15} {'Ratio':>15}")
        print("-" * 80)

        metrics = [
            ('Validity',
             teacher_validity['validity_rate'],
             student_validity['validity_rate']),
            ('Uniqueness',
             teacher_uniqueness['uniqueness_rate'],
             student_uniqueness['uniqueness_rate']),
            ('Exact Reconstruction',
             teacher_reconstruction['exact_match_rate'],
             student_reconstruction['exact_match_rate']),
            ('Generation Speed (mol/s)',
             teacher_speed['molecules_per_second'],
             student_speed['molecules_per_second']),
        ]

        for name, teacher_val, student_val in metrics:
            ratio = student_val / teacher_val if teacher_val > 0 else 0
            print(f"{name:<30} {teacher_val:>15.4f} {student_val:>15.4f} {ratio:>14.2f}x")

        # Model size comparison
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        size_ratio = student_params / teacher_params

        print(f"{'Model Size (params)':<30} {teacher_params:>15,} {student_params:>15,} {size_ratio:>14.2f}x")

        # Print OOV statistics
        print("\n" + "=" * 80)
        print("OUT-OF-VOCABULARY (OOV) MOTIF STATISTICS")
        print("=" * 80)

        print(f"\nTeacher Model:")
        print(f"  OOV molecules encountered: {teacher_validity['oov_count']}")
        print(f"  Total generation attempts: {teacher_validity['total_attempts']}")
        print(f"  OOV rate: {teacher_validity['oov_rate']*100:.2f}%")

        print(f"\nStudent Model:")
        print(f"  OOV molecules encountered: {student_validity['oov_count']}")
        print(f"  Total generation attempts: {student_validity['total_attempts']}")
        print(f"  OOV rate: {student_validity['oov_rate']*100:.2f}%")

        print(f"\nUnique OOV motifs encountered: {len(self.oov_stats['oov_motifs'])}")
        if self.oov_stats['oov_motifs']:
            print("\nExample OOV motifs (first 5):")
            for i, motif in enumerate(list(self.oov_stats['oov_motifs'])[:5]):
                print(f"  {i+1}. {motif}")

        print("=" * 80)

        return results

# %%
class Visualizer:
    """Visualization utilities for model evaluation"""

    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training loss curves

        Args:
            history: Dictionary of training metrics
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')

        epochs = history['epoch']

        # Total loss
        axes[0, 0].plot(epochs, history['total'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Reconstruction loss
        axes[0, 1].plot(epochs, history['reconstruction'], 'g-', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # KL loss
        axes[1, 0].plot(epochs, history['kl'], 'r-', linewidth=2)
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)

        # Distillation losses
        axes[1, 1].plot(epochs, history['latent_distill'], label='Latent', linewidth=2)
        axes[1, 1].plot(epochs, history['feature_match'], label='Feature', linewidth=2)
        axes[1, 1].set_title('Distillation Losses')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved: {save_path}")

        plt.show()

    @staticmethod
    def plot_property_distributions(teacher_props, student_props, save_path=None):
        """Plot property distributions for teacher vs student

        Args:
            teacher_props: Teacher property statistics
            student_props: Student property statistics
            save_path: Path to save figure (optional)
        """
        properties = ['mol_weight', 'logp', 'qed', 'num_atoms']
        titles = ['Molecular Weight', 'LogP', 'QED', 'Number of Atoms']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Property Distributions: Teacher vs Student', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for idx, (prop, title) in enumerate(zip(properties, titles)):
            ax = axes[idx]

            if prop in teacher_props and prop in student_props:
                teacher_vals = teacher_props[prop]['values']
                student_vals = student_props[prop]['values']

                # Plot histograms
                ax.hist(teacher_vals, bins=30, alpha=0.5, label='Teacher', color='blue', density=True)
                ax.hist(student_vals, bins=30, alpha=0.5, label='Student', color='red', density=True)

                ax.set_title(title)
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Property distribution plot saved: {save_path}")

        plt.show()

    @staticmethod
    def visualize_molecules(smiles_list, title="Generated Molecules", n_samples=16, save_path=None):
        """Visualize a grid of molecules

        Args:
            smiles_list: List of SMILES strings
            title: Title for the figure
            n_samples: Number of molecules to display
            save_path: Path to save figure (optional)
        """
        # Sample molecules
        sampled_smiles = random.sample(smiles_list, min(n_samples, len(smiles_list)))

        # Convert to RDKit mols
        mols = []
        for smiles in sampled_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)

        if not mols:
            print("No valid molecules to visualize")
            return

        # Create grid image
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=4,
            subImgSize=(300, 300),
            legends=[f"Mol {i+1}" for i in range(len(mols))]
        )

        # Display
        plt.figure(figsize=(15, 15))
        plt.imshow(img)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Molecule visualization saved: {save_path}")

        plt.show()

    @staticmethod
    def plot_comparison_bars(evaluation_results, save_path=None):
        """Plot bar chart comparison of key metrics

        Args:
            evaluation_results: Results from comprehensive_evaluation
            save_path: Path to save figure (optional)
        """
        metrics = [
            ('Validity',
             evaluation_results['teacher_validity']['validity_rate'],
             evaluation_results['student_validity']['validity_rate']),
            ('Uniqueness',
             evaluation_results['teacher_uniqueness']['uniqueness_rate'],
             evaluation_results['student_uniqueness']['uniqueness_rate']),
            ('Exact\nReconstruction',
             evaluation_results['teacher_reconstruction']['exact_match_rate'],
             evaluation_results['student_reconstruction']['exact_match_rate']),
        ]

        metric_names = [m[0] for m in metrics]
        teacher_vals = [m[1] * 100 for m in metrics]
        student_vals = [m[2] * 100 for m in metrics]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width/2, teacher_vals, width, label='Teacher', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, student_vals, width, label='Student', color='red', alpha=0.7)

        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=10)

        autolabel(bars1)
        autolabel(bars2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison bar chart saved: {save_path}")

        plt.show()

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§8&#8195;</span>Complete Pipeline</h2>

# %% [markdown]
# Now we bring everything together to execute the complete knowledge distillation pipeline.
#
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Execution Steps</h3>
#
# 1. Load training data
# 2. Initialize distillation trainer
# 3. Train student model
# 4. Evaluate both models
# 5. Visualize results
# 6. Generate final report
#
# This demonstrates the end-to-end workflow for creating an efficient molecular generative
# model through knowledge distillation.

# %%
def _json_safe(obj):
    """Convert NumPy scalars/arrays to built-in types for json.dump.

    compute_property_statistics() builds its summary with np.mean / np.min / np.max, so the
    results dict carries NumPy scalars. np.float64 subclasses Python float and serialises fine,
    but np.int64 does *not* subclass int - min/max over the integer properties (num_atoms,
    num_rings) therefore make json.dump raise
    `TypeError: Object of type int64 is not JSON serializable`.

    Args:
        obj: Object json.dump could not serialize

    Returns:
        A JSON-serializable equivalent
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def run_knowledge_distillation_pipeline():
    """Execute complete knowledge distillation pipeline"""

    print("\n" + "=" * 80)
    print("KNOWLEDGE DISTILLATION PIPELINE")
    print("=" * 80)

    # Step 1: Load training data
    print("\n[Step 1/6] Loading training data...")
    train_loader = load_training_data(
        config,
        vocab,
        max_samples=5000  # Use 5000 samples for demonstration
    )

    # Step 2: Initialize trainer
    print("\n[Step 2/6] Initializing distillation trainer...")
    trainer = DistillationTrainer(config, teacher_model, student_model, vocab)

    # Step 3: Train student model
    print("\n[Step 3/6] Training student model...")
    trainer.train(train_loader)

    # Step 4: Visualize training history
    print("\n[Step 4/6] Visualizing training history...")
    Visualizer.plot_training_history(
        trainer.history,
        save_path=os.path.join(config.OUTPUT_DIR, 'training_history.png')
    )

    # Step 5: Comprehensive evaluation
    print("\n[Step 5/6] Running comprehensive evaluation...")

    # Load test SMILES for reconstruction testing
    with open("data/chembl/all.txt", 'r') as f:
        test_smiles = [line.strip() for line in f if line.strip()]
    test_smiles = random.sample(test_smiles, min(1000, len(test_smiles)))

    evaluator = ModelEvaluator(config, teacher_model, student_model, vocab)
    results = evaluator.comprehensive_evaluation(test_smiles)

    # Step 6: Visualize the comparison and save a machine-readable report
    print("\n[Step 6/6] Saving figures and evaluation report...")

    Visualizer.plot_property_distributions(
        results['teacher_properties'],
        results['student_properties'],
        save_path=os.path.join(config.OUTPUT_DIR, 'property_distributions.png')
    )
    Visualizer.plot_comparison_bars(
        results,
        save_path=os.path.join(config.OUTPUT_DIR, 'comparison_bars.png')
    )
    Visualizer.visualize_molecules(
        results['student_validity']['generated_smiles'],
        title="Molecules Generated by the Distilled Student",
        save_path=os.path.join(config.OUTPUT_DIR, 'student_molecules.png')
    )

    results_path = os.path.join(config.OUTPUT_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_safe)
    print(f"\nEvaluation results saved: {results_path}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - config.json")
    print("  - best_student_model.pt")
    print("  - training_history.png")
    print("  - evaluation_results.json")
    print("  - property_distributions.png")
    print("  - comparison_bars.png")
    print("  - student_molecules.png")

    return trainer, results

# %% [markdown]
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:34px 0 10px;color:#1A1416;"><span style="font-family:ui-monospace,'SF Mono','JetBrains Mono',Menlo,monospace;font-size:13px;color:#A20025;font-weight:500;">§9&#8195;</span>Main Execution</h2>

# %% [markdown]
# This section runs the complete pipeline. 
#
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Usage Notes</h3>
#
# - For quick testing: Reduce NUM_EPOCHS and use fewer training samples
# - For production: Increase NUM_EPOCHS (100-200) and use full dataset
# - Monitor GPU memory: Reduce BATCH_SIZE if you encounter OOM errors
# - Checkpoint frequency: Adjust SAVE_EVERY based on training time
#
# <h3 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:600;font-size:17px;color:#1A1416;margin:24px 0 8px;">Expected Results</h3>
#
# After training, you should observe:
# - Student validity rate: 85-90% of teacher's
# - Generation speed: 2-3x faster than teacher
# - Model size: 50-60% reduction in parameters
# - Property distributions: Similar to teacher's
#
# The student model provides a practical trade-off between performance and efficiency,
# making it suitable for deployment in resource-constrained environments or applications
# requiring high-throughput molecular generation.

# %%
# Run the complete pipeline
trainer, results = run_knowledge_distillation_pipeline()

print("\n" + "=" * 80)
print("KNOWLEDGE DISTILLATION TUTORIAL COMPLETED")
print("=" * 80)

print("\n### Key Takeaways ###\n")
print("1. Knowledge distillation enables efficient model compression")
print("2. Student models can retain 85-90% of teacher performance")
print("3. Significant speedups (2-3x) enable real-time applications")
print("4. Combined loss functions transfer multiple aspects of teacher behavior")
print("5. Hierarchical architectures benefit from both compression and distillation")

print("\n### Next Steps ###\n")
print("1. Experiment with different student architectures")
print("2. Try different loss weight combinations (alpha, beta, gamma, delta)")
print("3. Apply the distilled model to property-guided optimization")
print("4. Compare with other compression techniques (pruning, quantization)")

print("\n" + "=" * 80)

# %% [markdown]
# <hr style="border:0;border-top:1px solid #A20025;opacity:.55;margin:34px 0 12px;">
# <h2 style="font-family:Georgia,'Iowan Old Style','Palatino Linotype',serif;font-weight:500;font-size:23px;margin:0 0 10px;color:#1A1416;">Summary</h2>

# %% [markdown]
#  Knowledge distillation for hierarchical molecular generation provides a practical solution to the deployment challenges posed by large neural models. By training compact student models to mimic capable teacher models, we can achieve most of the teacher's performance with a fraction of the computational cost. This chapter has explored the complete pipeline from hierarchical molecular representation through distillation training to comprehensive evaluation.
#
#  The hierarchical representation employed by HierVAE captures the modular nature of chemical structure, decomposing molecules into trees of motifs rather than flat sequences of atoms. This representation provides inductive biases that improve generation quality and efficiency. The HierVAE architecture combines variational autoencoders, message-passing neural networks, and hierarchical processing to enable both molecular reconstruction and novel molecule generation. Understanding this architecture was essential for effective distillation, as we needed to consider which components to compress and which knowledge to transfer.
#
#  The multi-component distillation loss transfers knowledge through four complementary channels: reconstruction quality, latent space organization, and intermediate feature representations. Balancing these components through hyperparameters allows us to control which aspects of the teacher's knowledge to prioritize. The reconstruction loss ensures basic functionality, the KL loss maintains latent space structure, the latent distillation loss transfers semantic organization, and the feature matching loss transfers learned chemical patterns.
#
#  Training distilled models requires careful attention to optimization dynamics. KL annealing prevents posterior collapse by initially focusing on reconstruction before gradually enforcing latent space constraints. The dual forward pass architecture enables efficient distillation by freezing the teacher and updating only the student. Gradient clipping, learning rate scheduling, and appropriate checkpointing ensure stable, reliable training. Monitoring multiple metrics throughout training helps diagnose problems and guide hyperparameter tuning.
#
#  Student model design balances compression and capability through architectural choices. Reducing hidden dimensions and latent dimensions cuts parameter count while preserving structural inductive biases. Keeping message-passing depths constant maintains the receptive field needed for long-range dependencies. Empirical evaluation of different compression ratios reveals the performance-efficiency trade-off and guides architecture selection for specific deployment scenarios.
#
#  Comprehensive evaluation characterizes both quality and efficiency of compressed models. Validity, uniqueness, reconstruction accuracy, property distributions, generation speed, and model size together capture the multifaceted nature of generative model quality. Statistical analysis provides confidence in observed differences and enables principled comparison of compression strategies. Ablation studies isolate the contribution of each loss component and identify opportunities for further improvement.
#
#  The techniques presented here generalize beyond molecular generation to other structured generation tasks. The core principles of knowledge distillation (learning from teacher outputs, intermediate representations, and latent organization) apply to any scenario where a large model must be compressed for deployment. The specific architectural choices (hierarchical decomposition, message passing, variational sampling) apply to generation of other graph-structured objects like proteins.


