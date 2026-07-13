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
#     display_name: ml4dd2025
#     language: python
#     name: python3
# ---

# %% [markdown]
# # <b> <font color='#A20025'> 📚 Chapter 6: Case Study: Small Molecule Binding to an RNA Target

# %% [markdown]
# *This notebook contains the code examples in chapter 6. For readability, the chapter notebooks only contain runnable code blocks and section titles. They omit the rest of the material in the book, i.e., text paragraphs, figures (unless generated as part of one of the code blocks), equations, and pseudocode. I recommend reading the notebooks side-by-side with the book!*
#
# You can work through this notebook locally as well as via Google Colab:
# <a target="_blank" href="https://colab.research.google.com/github/nrflynn2/ml-drug-discovery/blob/main/CH06_FLYNN_ML4DD.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>
#
# This chapter covers
# - An exemplary quantitative structure-activity relationship (QSAR) pipeline for understanding small molecule binding to RNA targets
# - Advanced molecular representation and descriptor calculation methods, especially in low data availability contexts
# - Representative data splitting with the Kennard-stone algorithm and dimensionality reduction algorithms like principal component analysis (PCA)
# - Sequential ensemble learning with gradient boosting
# - Advanced methods for model-specific and model-agnostic interpretability
#
# Note that this chapter involves two case studies, where the second case study starts at section 4 and involves a large data set and more computationally intensve methods.

# %% [markdown]
# ## <b> <font color='#A20025'> ⚙️ Environment Setup

# %% [markdown]
# **❗️LOCAL ENVIRONMENT:** If you are running the Python notebook locally on your computer, follow the README setup instructions. You can use:
# - **Advanced setup** (for this chapter): `pip install -r requirements-advanced.txt`
# - **Full setup** (all chapters): `conda env create -f ml4dd2025.yml`
#
# **❗️COLAB ENVIRONMENT:** If you are running on Google Colab, choose ONE of the installation options below based on your needs.

# %% tags=["skip-execution"]
# Colab users only
import os
CHAPTER = "ch06"
os.makedirs(f"artifacts/{CHAPTER}", exist_ok=True)
os.makedirs(f"data/{CHAPTER}", exist_ok=True)
os.makedirs(f"figures/{CHAPTER}", exist_ok=True)

# Download necessary data files.
# NOTE: the .pkl/.pkl.gz artifacts must land in artifacts/ch06/, NOT data/ch06/ --
# load_molecular_dataframe(..., chapter="ch06") and the binder_data.pkl reload both
# read from artifacts/<chapter>/, so downloading them into data/ch06/ would make
# those cells fail with FileNotFoundError.
# !wget "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/artifacts/ch06/binder_data.pkl" -O "artifacts/ch06/binder_data.pkl"
# !wget "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/artifacts/ch06/binder_model.pkl" -O "artifacts/ch06/binder_model.pkl"
# !wget "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/artifacts/ch06/hiv_tar1_lnkd.pkl.gz" -O "artifacts/ch06/hiv_tar1_lnkd.pkl.gz"
# !wget "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/artifacts/ch06/hiv_tar1_lnkoff.pkl.gz" -O "artifacts/ch06/hiv_tar1_lnkoff.pkl.gz"
# !wget "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/artifacts/ch06/hiv_tar1_lnkon.pkl.gz" -O "artifacts/ch06/hiv_tar1_lnkon.pkl.gz"
# !wget "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/data/ch06/CH06_hargrove_data.csv" -O "data/ch06/CH06_hargrove_data.csv"
# !wget "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/data/ch06/CH06_Mordred_Test_Compounds_3D.csv" -O "data/ch06/CH06_Mordred_Test_Compounds_3D.csv"
# !wget "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/ml4dd2025.yml" -O "env.yml"
# !wget "https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/utils.py" -O "utils.py"

# %% tags=["skip-execution"]
# Colab users only - OPTION 1: Quick Install (Advanced Packages)
# This installs the advanced packages needed for this chapter (gradient boosting, xgboost, shap, etc.)
# !pip install -q -r https://raw.githubusercontent.com/nrflynn2/ml-drug-discovery/main/requirements-advanced.txt

# %% tags=["skip-execution"]
# Colab users only - OPTION 2: Full Conda Environment (Part 1)
# Install condacolab for full conda environment support
# !pip install -q condacolab
import condacolab
condacolab.install()  # The kernel will restart after this cell runs

# %% tags=["skip-execution"]
# Colab users only - OPTION 2: Full Conda Environment (Part 2)
# After kernel restart, run this to install the full environment
# Some Colab runtimes do not define LD_LIBRARY_PATH before condacolab.check().
import os
ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
if "/usr/local/lib" not in ld_library_path.split(":"):
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib" + (":" + ld_library_path if ld_library_path else "")

import condacolab
condacolab.check()
# !mamba env update -n base -f env.yml

# %% tags=["skip-execution"]
# Colab users only - OPTION 2: Full Conda Environment (Part 3)
# Restart runtime to make all packages available
import os
os.kill(os.getpid(), 9)

# %% [markdown]
# ### <b> <font color='#A20025'> Import Packages 

# %% [markdown]
# Now let's import all the packages we'll need for this chapter.

# %%
# Core libraries
import itertools
import numpy as np
import pandas as pd
import pickle as pkl
import os
import subprocess

# Chemistry libraries
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Descriptors3D
from rdkit.Chem.MolStandardize import rdMolStandardize
from dimorphite_dl import protonate_smiles  # For protonation states (dimorphite-dl >= 2.0)

# Machine Learning libraries
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import r2_score, average_precision_score
from sklearn.metrics.pairwise import euclidean_distances
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor # Optional: uncomment if used
# from catboost import CatBoostClassifier, CatBoostRegressor # Optional: uncomment if used

# Interpretation libraries
import shap

# Visualization libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol # For 3D visualization

# Utility libraries
from scipy.spatial import distance
from scipy import stats
from tqdm.auto import tqdm # Works in notebooks and headless/CLI execution

# Display helper (resolves outside an interactive kernel too)
from IPython.display import display

# Utility functions
from utils import (
    save_molecular_dataframe,
    load_molecular_dataframe,
    list_saved_dataframes,
    set_seed,
    check_env,
    preview_df,
)

# %% [markdown]
# ### <b> <font color='#A20025'> Configure Settings

# %%
# Standardized setup: report the environment, then seed every RNG the chapter
# touches (Python, NumPy, and PyTorch if present) through the shared helper.
check_env([
    "numpy", "pandas", "scipy", "sklearn", "rdkit",
    "xgboost", "shap", "dimorphite_dl", "matplotlib", "seaborn",
])
# Runtime note: CPU-only chapter. Three cells are compute-heavy and flagged
# below with "Long Running Code Block" (3D conformer/descriptor generation, the
# exhaustive MLR search, and the 10-fold XGBoost cross-validation).
RANDOM_SEED = set_seed(42)

# %%
# Matplotlib and Seaborn setup for consistent visualizations
def setup_visualization_style():
    """Configure consistent visualization style for the notebook"""
    colors = ["#A20025", "#6C8EBF"]  # Define a color palette
    sns.set_palette(sns.color_palette(colors))
    plt.rcParams['figure.dpi'] = 100 # Adjust for better resolution inline
    plt.rcParams['savefig.dpi'] = 600 # High resolution for saved figures
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14   
    plt.rcParams['xtick.labelsize'] = 12   
    plt.rcParams['ytick.labelsize'] = 12    

setup_visualization_style()
# %matplotlib inline

# %% [markdown]
# ## <b> <font color='#A20025'> 1️⃣ Small Molecule Binding to an RNA Target

# %% [markdown]
# In this section, we'll explore how small molecules bind to RNA targets, focusing on the HIV TAR RNA as a case study. We'll examine both the structural representation of molecules and their binding kinetics.

# %% [markdown]
# ### <b> <font color='#A20025'> Structure: Computing Descriptors

# %% [markdown]
# First, let's load our dataset containing small molecules that bind to the HIV TAR RNA target.

# %%
def load_hiv_tar_data(filepath="data/ch06/CH06_hargrove_data.csv"):
    """
    Load the HIV TAR RNA binding dataset.
    
    Parameters:
        filepath (str): Path to the dataset file
        
    Returns:
        pandas.DataFrame: DataFrame containing molecule data and binding parameters
    """
    try:
        # Load data with relevant columns
        tar_df = pd.read_csv(
            filepath, 
            usecols=["Name", "SMILES", "kon", "koff", "KD", "lnkon", "lnkoff", "lnKD"]
        )
        
        print(f"Loaded dataset with {len(tar_df)} compounds")
        return tar_df
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

# Load the HIV TAR data
hiv_tar1 = load_hiv_tar_data()

# Display basic information
if hiv_tar1 is not None:
    display(preview_df(hiv_tar1, "hiv_tar1 (raw HIV TAR binding kinetics)"))
else:
    print("Failed to load the dataset. Please check the file path and format.")

# %% [markdown]
# #### <b> <font color='#A20025'> 3D Molecular Conformers

# %% [markdown]
# Predicting binding affinities often requires understanding the 3D structure of molecules. Since molecules are flexible, they can adopt multiple shapes (conformers). We'll generate multiple conformers and consider protonation states (protomers) and tautomers to better represent the molecule's behavior in a biological environment. We then calculate 3D descriptors and aggregate them using Boltzmann weighting based on estimated conformer energies.
#
# Let's generate and visualize 3D conformers for a drug molecule (esomeprazole) to understand the conformational flexibility:

# %%
def generate_and_align_conformers(smiles, n_conformers=10):
    """
    Generate and align multiple conformers for a molecule.
    
    Parameters:
        smiles (str): SMILES string of the molecule
        n_conformers (int): Number of conformers to generate
        
    Returns:
        rdkit.Chem.rdchem.Mol: Molecule with multiple conformers
    """
    # Create molecule from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES string")
        return None
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate multiple conformers
    print(f"Generating {n_conformers} conformers...")
    AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=AllChem.ETKDG())
    
    # Align conformers using a common substructure. For esomeprazole we align on
    # the benzimidazole core. The aromatic-nitrogen SMARTS below uses a bare `n`
    # (no explicit hydrogen count), so it matches the ring whether the nitrogen is
    # the neutral [nH] tautomer or the anionic [n-] form present in this SMILES --
    # a plain `[nH]` pattern would return an empty match and silently align on
    # nothing. Note: the pattern is specific to this molecule.
    try:
        substructure = Chem.MolFromSmarts('c1nc2ccccc2n1')  # Benzimidazole core
        match = mol.GetSubstructMatch(substructure)
        if match:
            AllChem.AlignMolConformers(mol, list(match))
            print(f"Successfully aligned {mol.GetNumConformers()} conformers on the benzimidazole core")
        else:
            # No common substructure found -> fall back to whole-molecule alignment
            AllChem.AlignMolConformers(mol)
            print(f"Substructure not found; aligned {mol.GetNumConformers()} conformers on all atoms")
    except Exception as e:
        print(f"Warning: Could not align conformers: {str(e)}")
    
    return mol

def visualize_molecule_3d(mol, conf_id=-1):
    """
    Visualize a molecule in 3D using py3Dmol.
    
    Parameters:
        mol (rdkit.Chem.rdchem.Mol): Molecule to visualize
        conf_id (int): Conformer ID to visualize (-1 for all)
        
    Returns:
        py3Dmol.view: 3D visualization
    """
    # Create a py3Dmol view
    view = py3Dmol.view(width=500, height=500)
    
    if conf_id == -1:
        # Visualize all conformers
        for i in range(mol.GetNumConformers()):
            mol_block = Chem.MolToMolBlock(mol, confId=i)
            view.addModel(mol_block, 'mol')
    else:
        # Visualize a specific conformer
        mol_block = Chem.MolToMolBlock(mol, confId=conf_id)
        view.addModel(mol_block, 'mol')
    
    # Set visualization style
    view.setStyle({'stick': {}})
    view.setBackgroundColor('white')
    view.zoomTo()
    
    return view

# Generate conformers for esomeprazole
esomeprazole_smiles = 'COc1ccc2[n-]c([S@@+]([O-])Cc3ncc(C)c(OC)c3C)nc2c1'
esomeprazole = generate_and_align_conformers(esomeprazole_smiles)

# %% [markdown]
# We generated 10 conformers of esomeprazole and overlaid them on top of each other, demonstrating the broad spatial range that a molecule’s structure can inhabit.

# %%
# Check how many conformers were generated
if esomeprazole is not None:
    print(f"Number of conformers: {esomeprazole.GetNumConformers()}")
    
    # Visualize just the first conformer
    view_single = visualize_molecule_3d(esomeprazole, conf_id=0)
    display(view_single.show())
    
    # Visualize all conformers overlaid
    view_all = visualize_molecule_3d(esomeprazole, conf_id=-1)
    display(view_all.show())
else:
    print("Failed to generate molecule conformers")

# %% [markdown]
# #### <b> <font color='#A20025'> Protonation States and Tautomers

# %% [markdown]
# Molecules can exist in different protonation states and tautomeric forms depending on the pH and environment. We'll enumerate these states for our set of molecules:

# %%
def generate_protonation_states_and_tautomers(smiles_list, ph_min=6.4, ph_max=8.4, max_variants=128):
    """
    Generate protonation states and tautomers for a list of molecules.

    Parameters:
        smiles_list (list): List of SMILES strings
        ph_min (float): Lower bound of the pH range to enumerate protomers over
        ph_max (float): Upper bound of the pH range to enumerate protomers over
        max_variants (int): Maximum protonation variants per input compound

    Returns:
        dict: Dictionary mapping molecule indices to their protonation states and tautomers
    """
    # Protonation states are enumerated over a physiological pH range (6.4-8.4)
    # with dimorphite-dl. Note: v2.0 replaced the old `DimorphiteDL` class with
    # the `protonate_smiles()` function used below.

    # Initialize tautomer enumerator
    tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
    
    # Dictionary to store protonation states and tautomers
    molecule_variants = {}
    
    print("Generating protonation states and tautomers...")
    for idx, smi in enumerate(tqdm(smiles_list)):
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Warning: Could not parse SMILES {smi}")
            continue
        
        # Generate protonation states, then enumerate tautomers from EACH
        # protonation state (not only the original molecule). Deduplicate the
        # combined set by canonical SMILES and guard against states that fail
        # to parse or enumerate.
        protonated_smiles = protonate_smiles(
            smi,
            ph_min=ph_min,
            ph_max=ph_max,
            max_variants=max_variants,
            label_states=False,
        )

        variants = []
        seen_smiles = set()
        for protonated_smi in protonated_smiles:
            protonated_mol = Chem.MolFromSmiles(protonated_smi)
            if protonated_mol is None:
                continue
            for variant in tautomer_enumerator.Enumerate(protonated_mol):
                if variant is None:
                    continue
                canonical_smi = Chem.MolToSmiles(variant)
                if canonical_smi not in seen_smiles:
                    seen_smiles.add(canonical_smi)
                    variants.append(variant)

        molecule_variants[idx] = variants
        
    print(f"Generated variants for {len(molecule_variants)} molecules")
    return molecule_variants

# Generate protonation states and tautomers for our dataset
ligand2protomers = generate_protonation_states_and_tautomers(hiv_tar1["SMILES"].tolist())

# %% [markdown]
# #### <b> <font color='#A20025'> Conformer Generation and Energy Minimization

# %% [markdown]
# For accurate 3D descriptors, we need to generate reasonable 3D conformers for our molecules. Let's implement a function to generate and energy-minimize conformers:

# %%
def generate_conformers(protomers, num_confs=10):
    """
    Generate and energy-minimize conformers for a list of molecules.
    
    Parameters:
        protomers (list): List of RDKit molecule objects
        num_confs (int): Number of conformers to generate per molecule
        
    Returns:
        tuple: (list of molecules with conformers, dict of conformer energies)
    """
    # Set up parameters for conformer generation
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d  # For reproducibility
    params.numThreads = 4       # Parallel computation
    
    # Lists and dictionaries to store results
    protomers_with_confs = []
    protomer_energies = {}
    
    print(f"Generating conformers for {len(protomers)} molecules...")
    n_unconverged = 0
    for protomer in tqdm(protomers):
        # Skip None molecules
        if protomer is None:
            continue

        # Add hydrogens (important for accurate conformer generation)
        protomer = Chem.AddHs(protomer)

        # Generate multiple conformers
        conformer_ids = list(AllChem.EmbedMultipleConfs(
            protomer,
            numConfs=num_confs,
            params=params
        ))

        # If no conformers could be embedded, skip this molecule
        if len(conformer_ids) == 0:
            continue

        # Energy minimize conformers using the UFF forcefield. UFF can fail
        # outright on unsupported atom types or bad geometries, so guard the call.
        try:
            UFF_output = AllChem.UFFOptimizeMoleculeConfs(protomer)
        except Exception as e:
            print(f"Warning: UFF optimization failed ({e}); skipping this molecule")
            continue

        # UFFOptimizeMoleculeConfs returns one (converged, energy) pair per
        # conformer, where converged == 0 means the minimization converged.
        n_unconverged += sum(1 for converged, _ in UFF_output if converged != 0)

        # Map each conformer ID to its minimized energy, so IDs and energies can
        # never desynchronize regardless of the IDs RDKit assigned.
        id_to_energy = {
            conf_id: energy
            for conf_id, (_, energy) in zip(conformer_ids, UFF_output)
        }

        # Keep only low-energy conformers (within 3 kcal/mol of the minimum).
        # Build the removal list FIRST and delete afterwards: removing conformers
        # while iterating over them desynchronizes conformer IDs from energies.
        min_energy_UFF = min(id_to_energy.values())
        energy_threshold = min_energy_UFF + 3
        ids_to_remove = [
            conf_id for conf_id, energy in id_to_energy.items()
            if energy >= energy_threshold
        ]
        for conf_id in ids_to_remove:
            protomer.RemoveConformer(conf_id)

        # Get the remaining conformer IDs, in the order GetConformers() yields them
        remaining_ids = [conformer.GetId() for conformer in protomer.GetConformers()]
        if not remaining_ids:
            continue

        # Store the molecule and its conformer energies. Key the energies by the
        # molecule's POSITION in protomers_with_confs (not the original index), so
        # they stay aligned with that list after None/failed molecules are skipped.
        protomers_with_confs.append(protomer)
        protomer_energies[len(protomers_with_confs) - 1] = [
            id_to_energy[conf_id] for conf_id in remaining_ids
        ]

    if n_unconverged:
        print(f"Note: {n_unconverged} conformer minimizations did not fully converge")
    print(f"Generated conformers for {len(protomers_with_confs)} molecules")
    return protomers_with_confs, protomer_energies

# %% [markdown]
# #### <b> <font color='#A20025'> Boltzmann Weighting of Conformers

# %% [markdown]
# Different conformers contribute to the properties of a molecule based on their relative energies, according to the Boltzmann distribution. Let's compute Boltzmann weights for our conformers:

# %%
def compute_boltzmann_weights(protomer_energies, temperature=295):
    """
    Compute Boltzmann weights for conformers based on their energies.
    
    Parameters:
        protomer_energies (dict): Dictionary mapping protomer indices to conformer energies
        temperature (float): Temperature in Kelvin (default: 295K, room temperature)
        
    Returns:
        dict: Dictionary mapping protomer indices to conformer weights
    """
    # Boltzmann constant in kcal/(mol·K)
    k_B = 1.987E-3
    kT = k_B * temperature

    # Dictionary to store weights
    protomer_weights = {}

    # Guard the empty case (no conformer survived the energy filter)
    all_energies = [e for energies in protomer_energies.values() for e in energies]
    if not all_energies:
        return protomer_weights

    # Exponentiate energies RELATIVE to the global minimum. A Boltzmann weight is
    # a ratio, so subtracting a constant from every energy leaves it mathematically
    # unchanged while keeping exp() away from overflow/underflow -- raw UFF energies
    # can be large and negative, and exp(-E/kT) would otherwise overflow to inf.
    min_energy = min(all_energies)

    # Calculate partition function (denominator)
    partition_function = sum(
        np.exp(-(energy - min_energy) / kT) for energy in all_energies
    )

    # Calculate weights for each conformer
    for protomer_idx, conformer_energies in protomer_energies.items():
        protomer_weights[protomer_idx] = [
            np.exp(-(energy - min_energy) / kT) / partition_function
            for energy in conformer_energies
        ]

    return protomer_weights

# %% [markdown]
# #### <b> <font color='#A20025'> Computing 3D Descriptors with Boltzmann Weighting

# %% [markdown]
# Now let's calculate 3D molecular descriptors for each conformer and combine them using Boltzmann weighting:
#
# **😱 Long Running Code Block**
#
# The following code block may take a long time to run (up to 30 minutes). This is a great chance to read the corresponding section in the book! But if you want to save time, you can skip this cell and load the resulting, processed data at the beginning of the section labeled **Representative Data Splitting with Kennard-Stone Algorithm**

# %% tags=["skip-execution"]
def calculate_boltzmann_weighted_descriptors(molecules_dict, num_confs=10):
    """
    Calculate Boltzmann-weighted 3D descriptors for a set of molecules.
    
    Parameters:
        molecules_dict (dict): Dictionary mapping molecule indices to variants
        num_confs (int): Number of conformers to generate
        
    Returns:
        numpy.ndarray: Array of Boltzmann-weighted descriptors
    """
    all_descriptors = []
    
    print("Calculating Boltzmann-weighted descriptors...")
    for ligand_idx, protomers in tqdm(molecules_dict.items()):
        # Generate conformers and calculate their energies
        protomers_with_confs, protomer_energies = generate_conformers(protomers, num_confs)
        
        # Calculate Boltzmann weights
        protomer_weights = compute_boltzmann_weights(protomer_energies)
        
        # Calculate descriptors for each conformer and apply weights
        weighted_descriptors = []
        for protomer_idx, protomer in enumerate(protomers_with_confs):
            # Skip molecules with no conformers
            if protomer.GetNumConformers() == 0:
                continue
                
            for conf_idx, conf in enumerate(protomer.GetConformers()):
                # Calculate 3D descriptors for this conformer
                descriptors = Chem.Descriptors3D.CalcMolDescriptors3D(protomer, confId=conf.GetId())
                
                # Weight by Boltzmann probability
                weight = protomer_weights[protomer_idx][conf_idx]
                weighted_desc = [weight * value for desc_name, value in descriptors.items()]
                weighted_descriptors.append(weighted_desc)
        
        # Sum the weighted descriptors if we have any
        all_descriptors.append(np.sum(weighted_descriptors, axis=0))
    
    return np.array(all_descriptors)

# Calculate Boltzmann-weighted descriptors
ligand_descs = calculate_boltzmann_weighted_descriptors(ligand2protomers, num_confs=5)

print(f"Generated descriptor matrix with shape: {ligand_descs.shape}")


# %% [markdown]
# ### <b> <font color='#A20025'> Activity: Experimentally Measuring Binding Profiles

# %% [markdown]
# Now let's examine the binding kinetics of our molecules to the HIV TAR RNA target. The dataset includes kon (association rate), koff (dissociation rate), and KD (equilibrium dissociation constant) values.

# %%
classes = ["AG"]*9 + ["DMA"]*11 + ["Nucleic acid dye"]*2 + ["DPF"]*2 + ["Nucleic acid dye"]*3 + ["DMA"]*6 + ["Nucleic acid dye"] + ["DPF"]*5 + ["Nucleic acid dye"] + ["DPF"]*5 + ["DMZ"]*3
hiv_tar1["class"] = classes

fig, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
g = sns.scatterplot(data=hiv_tar1, x="kon", y="koff", hue="class", style="class", s=80, ax=ax)
g.get_legend().set_title(None)
ax.set_ylabel("$k_{off}$ $[s^{-1}]$")
ax.set_xlabel("$k_{on}$ $[s^{-1}M^{-1}]$")

plt.tight_layout()
plt.savefig('figures/ch06/ligand_target_dist.png', bbox_inches='tight', dpi=600)
plt.savefig('figures/ch06/ligand_target_dist.pdf', bbox_inches='tight', dpi=600);

# %% [markdown]
# ## <b> <font color='#A20025'> 2️⃣ Representative Data Splitting & Dimensionality Reduction
#

# %% [markdown]
# ### <b> <font color='#A20025'> Data Refinement

# %% [markdown]
# Before building models, we need to refine our dataset. First, let's examine the distribution of our target variables and consider log-transformation:

# %%
def visualize_target_distributions(df):
    """
    Visualize the distribution of raw and log-transformed target variables.
    
    Parameters:
        df (pandas.DataFrame): DataFrame with target variables
        
    Returns:
        matplotlib.figure.Figure: Figure with the visualizations
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    # KD distributions
    sns.histplot(data=df, x="KD", ax=axes[0][0], kde=True)
    sns.histplot(data=df, x="lnKD", ax=axes[0][1], kde=True, binwidth=0.5)
    
    # kon distributions
    sns.histplot(data=df, x="kon", ax=axes[1][0], kde=True)
    sns.histplot(data=df, x="lnkon", ax=axes[1][1], kde=True, binwidth=0.5)
    
    # koff distributions
    sns.histplot(data=df, x="koff", ax=axes[2][0], kde=True)
    sns.histplot(data=df, x="lnkoff", ax=axes[2][1], kde=True, binwidth=0.2)
    
    # Set axis labels
    axes[0][0].set(xlabel="$K_{D}$", ylabel="Count")
    axes[0][1].set(xlabel="$lnK_{D}$", ylabel="Count")
    axes[1][0].set(xlabel="$k_{on}$", ylabel="Count")
    axes[1][1].set(xlabel="$lnk_{on}$", ylabel="Count")
    axes[2][0].set(xlabel="$k_{off}$", ylabel="Count")
    axes[2][1].set(xlabel="$lnk_{off}$", ylabel="Count")
    
    # Add a title to the figure
    fig.suptitle("Distribution of Raw vs. Log-transformed Target Variables", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.5)
    plt.savefig('figures/ch06/log_norm_targets.png', bbox_inches='tight', dpi=600)
    plt.savefig('figures/ch06/log_norm_targets.pdf', bbox_inches='tight', dpi=600)
    
    return fig

# Visualize target distributions
distributions_fig = visualize_target_distributions(hiv_tar1)

# %% [markdown]
# Now let's prepare separate dataframes for modeling each of our target variables (lnkon, lnkoff, lnKD):

# %%
def prepare_target_dataframes(df):
    """
    Prepare separate dataframes for each target variable.
    
    Parameters:
        df (pandas.DataFrame): DataFrame with multiple target variables
        
    Returns:
        tuple: (lnkon_df, lnkoff_df, lnkd_df) Dataframes for each target
    """
    # Create a dataframe for lnkon
    lnkon_df = df.drop(columns=["Name", "SMILES", "kon", "koff", "KD", "lnkoff", "lnKD"])
    
    # Create a dataframe for lnkoff
    lnkoff_df = df.drop(columns=["Name", "SMILES", "kon", "koff", "KD", "lnkon", "lnKD"])
    
    # Create a dataframe for lnKD
    lnkd_df = df.drop(columns=["Name", "SMILES", "kon", "koff", "KD", "lnkon", "lnkoff"])
    
    print(f"Prepared dataframes with {len(lnkon_df)} compounds:")
    print(f"- lnkon dataframe with {len(lnkon_df.columns)} columns")
    print(f"- lnkoff dataframe with {len(lnkoff_df.columns)} columns")
    print(f"- lnKD dataframe with {len(lnkd_df.columns)} columns")
    
    return lnkon_df, lnkoff_df, lnkd_df

# Reload the dataset with all descriptor columns
hiv_tar1_full = pd.read_csv("data/ch06/CH06_hargrove_data.csv")

# Prepare dataframes for each target
hiv_tar1_lnkon, hiv_tar1_lnkoff, hiv_tar1_lnkd = prepare_target_dataframes(hiv_tar1_full)

# %% [markdown]
# #### <b> <font color='#A20025'> Feature Selection and Removal of Redundant Descriptors

# %% [markdown]
# Let's refine our data by removing redundant or uninformative descriptors:

# %%
def refine_descriptors(df, target_col, redundancy_threshold=0.8, correlation_threshold=0.95):
    """
    Remove redundant descriptors based on variance and correlation.
    
    Parameters:
        df (pandas.DataFrame): DataFrame with descriptors and target
        target_col (str): Name of the target column
        redundancy_threshold (float): Threshold for removing low-variance descriptors
        correlation_threshold (float): Threshold for removing highly correlated descriptors
        
    Returns:
        pandas.DataFrame: Refined DataFrame with selected descriptors
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Step 1: Remove descriptors with low variance
    # (those where one value appears more than the threshold percentage of the time)
    low_var_cols = [col for col in df_copy if 
                   col != target_col and 
                   df_copy[col].value_counts(normalize=True).max() > redundancy_threshold]
    
    if low_var_cols:
        print(f"Removing {len(low_var_cols)} descriptors with low variance")
        df_copy = df_copy.drop(columns=low_var_cols)
    
    # Step 2: Remove highly correlated descriptors
    # Calculate correlation matrix excluding the target column
    corr_matrix = df_copy.drop(columns=[target_col]).corr().abs()
    
    # Set diagonal to zero to avoid self-correlations
    np.fill_diagonal(corr_matrix.values, 0)
    
    cols_to_remove = []
    
    # Iteratively remove highly correlated features
    while corr_matrix.gt(correlation_threshold).any().any():
        # Find the column with the most highly correlated pairs
        correlated_cols = corr_matrix.apply(lambda x: x[x > correlation_threshold].index.tolist()).tolist()
        num_correlated = [len(cols) for cols in correlated_cols]
        
        if max(num_correlated) > 0:
            max_index = np.argmax(num_correlated)
            
            if num_correlated[max_index] > 2:
                # Case 1: One descriptor is correlated with multiple others
                max_col = corr_matrix.columns[max_index]
                cols_to_remove.append(max_col)
                corr_matrix = corr_matrix.drop(index=max_col, columns=max_col)
            else:
                # Case 2: Pair of correlated descriptors
                for i, corr_list in enumerate(correlated_cols):
                    if len(corr_list) > 0:
                        # Find the pair that's correlated
                        col1 = corr_matrix.columns[i]
                        col2 = corr_list[0]
                        
                        # Keep the one more correlated with the target
                        target_corr = df_copy[[col1, col2, target_col]].corr()[target_col].abs()
                        drop_col = col1 if target_corr[col1] < target_corr[col2] else col2
                        
                        cols_to_remove.append(drop_col)
                        corr_matrix = corr_matrix.drop(index=drop_col, columns=drop_col)
                        break
        else:
            # No more highly correlated features
            break
    
    if cols_to_remove:
        print(f"Removing {len(cols_to_remove)} highly correlated descriptors")
        df_copy = df_copy.drop(columns=cols_to_remove)
    
    print(f"Refined dataset from {len(df.columns)} to {len(df_copy.columns)} columns")
    return df_copy

# %%
# Refine each dataframe by removing redundant descriptors
hiv_tar1_lnkon = refine_descriptors(hiv_tar1_lnkon, target_col="lnkon")
hiv_tar1_lnkoff = refine_descriptors(hiv_tar1_lnkoff, target_col="lnkoff")
hiv_tar1_lnkd = refine_descriptors(hiv_tar1_lnkd, target_col="lnKD")

# Print the number of descriptors retained for each target
print("\nNumber of descriptors after refinement:")
print(f"- lnkon: {len(hiv_tar1_lnkon.columns) - 1} descriptors")
print(f"- lnkoff: {len(hiv_tar1_lnkoff.columns) - 1} descriptors")
print(f"- lnKD: {len(hiv_tar1_lnkd.columns) - 1} descriptors")

# Sanity checks: refinement must keep every compound and each target column
for _df, _target in [(hiv_tar1_lnkon, "lnkon"), (hiv_tar1_lnkoff, "lnkoff"), (hiv_tar1_lnkd, "lnKD")]:
    assert _target in _df.columns, f"refinement dropped the target column {_target}"
    assert len(_df) == len(hiv_tar1_full), "refinement should drop columns, never rows"

preview_df(hiv_tar1_lnkd, "hiv_tar1_lnkd (refined descriptors + lnKD target)")


# %%
# Uncomment the below line if you want to save the hiv_tar1_lnk{on,off,d} data up to this point
# save_molecular_dataframe(hiv_tar1_lnkon, "hiv_tar1_lnkon", chapter="ch06")
# save_molecular_dataframe(hiv_tar1_lnkoff, "hiv_tar1_lnkoff", chapter="ch06")
# save_molecular_dataframe(hiv_tar1_lnkd, "hiv_tar1_lnkd", chapter="ch06")

# %% [markdown]
# ### <b> <font color='#A20025'> Representative Data Splitting with Kennard-Stone Algorithm

# %% [markdown]
# Before starting this section, if you skipped the conformer and descriptor processing code, make sure to load the dataframes below from artifacts/ch06. If you did not skip those sections, you can skip the next cell:

# %%
hiv_tar1_lnkon = load_molecular_dataframe("hiv_tar1_lnkon", chapter="ch06")
hiv_tar1_lnkoff = load_molecular_dataframe("hiv_tar1_lnkoff", chapter="ch06")
hiv_tar1_lnkd = load_molecular_dataframe("hiv_tar1_lnkd", chapter="ch06")

# %% [markdown]
# The Kennard-Stone algorithm ensures that we select a representative test set across the chemical space of our dataset. We'll implement it and visualize the distribution of samples:
#

# %%
def kennard_stone_algorithm(X, n_samples, distance_metric='euclidean'):
    """
    Kennard-Stone algorithm for representative sample selection.
    
    Parameters:
        X (numpy.ndarray): Feature matrix
        n_samples (int): Number of samples to select
        distance_metric (str): Distance metric to use ('euclidean' or 'mahalanobis')
        
    Returns:
        list: Indices of selected samples
    """
    n_total_samples = X.shape[0]
    
    if n_samples > n_total_samples:
        raise ValueError(f"Cannot select {n_samples} samples from a dataset of size {n_total_samples}")
    
    # Start with the point farthest from the mean
    if distance_metric == 'mahalanobis':
        # Calculate covariance matrix for Mahalanobis distance. With few samples
        # relative to the number of descriptors (this dataset has only ~48
        # compounds), the covariance matrix is singular or badly conditioned and
        # np.linalg.inv raises LinAlgError. The Moore-Penrose pseudo-inverse is
        # well defined in that regime and matches inv() when X is full rank.
        cov_matrix = np.cov(X.T)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

        # Find point farthest from the mean
        mean_point = X.mean(axis=0).reshape(1, -1)
        distances = distance.cdist(X, mean_point, 'mahalanobis', VI=inv_cov_matrix).ravel()
    else:
        # Find point farthest from the mean using Euclidean distance
        distances = np.sum(euclidean_distances(X, [np.mean(X, axis=0)]), axis=1)
    
    # Select the first point (farthest from mean)
    selected_indices = [np.argmax(distances)]
    remaining_indices = list(range(n_total_samples))
    remaining_indices.remove(selected_indices[0])
    
    # Iteratively select remaining points
    for _ in range(1, n_samples):
        if distance_metric == 'mahalanobis':
            # Calculate Mahalanobis distances to selected points
            min_distances = []
            for idx in remaining_indices:
                dists = distance.cdist(X[selected_indices], [X[idx]], 'mahalanobis', VI=inv_cov_matrix).min()
                min_distances.append(dists)
        else:
            # Calculate Euclidean distances to selected points
            min_distances = np.min(euclidean_distances(X[remaining_indices], X[selected_indices]), axis=1)
        
        # Select the point with maximum distance to any selected point
        new_idx = remaining_indices[np.argmax(min_distances)]
        selected_indices.append(new_idx)
        remaining_indices.remove(new_idx)
    
    return selected_indices

# %% [markdown]
# Let's visualize how the Kennard-Stone algorithm selects representative samples from a synthetic dataset:

# %%
# Generate a synthetic dataset with clusters for illustration
from sklearn.datasets import make_blobs

def visualize_kennard_stone_example():
    """
    Visualize the Kennard-Stone algorithm on a synthetic dataset.
    
    Returns:
        matplotlib.figure.Figure: Figure with the visualization
    """
    # Generate synthetic data with clusters
    X, _ = make_blobs(n_samples=1000, centers=10, random_state=42)
    
    # Apply Kennard-Stone algorithm
    n_selected_samples = 100
    selected_indices = kennard_stone_algorithm(X, n_selected_samples)
    selected_samples = X[selected_indices]
    
    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], label='All Samples', color='#6C8EBF', alpha=0.6)
    plt.scatter(selected_samples[:, 0], selected_samples[:, 1], color='#A20025', 
                marker='x', label='Selected Samples', s=100)
    plt.title('Kennard-Stone Algorithm for Representative Sample Selection', fontsize=14)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('figures/ch06/KSA_example.png', bbox_inches='tight', dpi=600)
    plt.savefig('figures/ch06/KSA_example.pdf', bbox_inches='tight', dpi=600)
    
    return plt.gcf()

# Visualize the Kennard-Stone algorithm
ks_example_fig = visualize_kennard_stone_example()

# %% [markdown]
# ### <b> <font color='#A20025'> Dimensionality Reduction: Principal Component Analysis

# %% [markdown]
# Before applying the Kennard-Stone algorithm to our molecular descriptors, we'll use Principal Component Analysis (PCA) to reduce dimensionality and visualize the data in a lower-dimensional space. An illustrative example of what PCA does is provided below.

# %% [markdown]
# **Code for Figure 6.8** Dimension $x_2$ carries almost no information since the $x_2$-values for each point barely deviate relative to the values along dimension $x_1$. Conversely, the $x_1$ dimension carries a lot of information and removing it would result in a compressed form of the data that is very dissimilar from the original data. 

# %%
# Generate 2D data with noise
np.random.seed(0)
X = np.random.randn(100, 2)
X[:, 1] *= 0.1  # Add small amounts of noise to the second dimension

# Plot the original data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=60, c="#6C8EBF")
plt.ylim(-2, 2)
plt.title('Original 2D Data')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.yticks(range(-2, 3, 1))
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)

# Perform PCA and reduce to single dimension
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# Plot the data reduced to single dimension
plt.subplot(1, 2, 2)
plt.scatter(X_pca, np.zeros_like(X_pca), s=60, c="#A20025")
plt.ylim(-2, 2)
plt.title('Data Reduced to Single Dimension')
plt.xlabel('$x_1$')
plt.yticks(range(-2, 3, 1))
plt.axhline(0, color='black', linewidth=2)

plt.tight_layout()
plt.savefig('figures/ch06/2D_to_1D.png', bbox_inches='tight', dpi=600)
plt.savefig('figures/ch06/2D_to_1D.pdf', bbox_inches='tight', dpi=600);

# %%
def apply_pca_to_descriptors(df_list, target_cols):
    """
    Apply PCA to the descriptor matrices for each target variable.
    
    Parameters:
        df_list (list): List of DataFrames for each target
        target_cols (list): List of target column names
        
    Returns:
        tuple: (X_dr_list, explained_variance_list, pca_list)
            - X_dr_list: List of dimensionality-reduced feature matrices
            - explained_variance_list: List of explained variance ratios for each PCA
            - pca_list: List of fitted PCA objects
    """
    X_dr_list = []
    explained_variance_list = []
    pca_list = []
    
    # Apply standardization and PCA to each target's descriptors
    for df, target_col in zip(df_list, target_cols):
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(df.drop(columns=[target_col]))
        
        # Apply PCA, retaining 99% of variance
        pca = PCA(n_components=0.99)
        X_dr = pca.fit_transform(X)
        
        # Store results
        X_dr_list.append(X_dr)
        explained_variance_list.append(pca.explained_variance_ratio_)
        pca_list.append(pca)
        
        print(f"PCA for {target_col}: Reduced from {X.shape[1]} to {X_dr.shape[1]} dimensions")
    
    return X_dr_list, explained_variance_list, pca_list

# Apply PCA to our descriptor matrices
df_list = [hiv_tar1_lnkon, hiv_tar1_lnkoff, hiv_tar1_lnkd]
target_cols = ["lnkon", "lnkoff", "lnKD"]
X_dr_list, explained_variance_list, pca_list = apply_pca_to_descriptors(df_list, target_cols)

# Unpack the results for easier access
X_lnkon_dr, X_lnkoff_dr, X_lnkd_dr = X_dr_list
lnkon_explained_variance, lnkoff_explained_variance, lnkd_explained_variance = explained_variance_list
pca_lnkon, pca_lnkoff, pca_lnkd = pca_list


# %% [markdown]
# Let's examine different methods for determining the optimal number of principal components to retain:

# %%
def visualize_optimal_component_selection(pca, eigenvalues, cumsum):
    """
    Visualize criteria for selecting the optimal number of principal components.
    
    Parameters:
        pca (PCA): Fitted PCA object
        eigenvalues (numpy.ndarray): Eigenvalues (explained variance)
        cumsum (numpy.ndarray): Cumulative sum of explained variance ratios
        
    Returns:
        matplotlib.figure.Figure: Figure with the visualizations
    """
    # Create a figure with three subplots
    plt.figure(figsize=(24, 6))
    
    # 1. Eigenvalue criterion (Kaiser's rule)
    plt.subplot(1, 3, 1)
    n_components = np.argmax(eigenvalues < 1)
    plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--', linewidth=2)
    plt.xlabel('Principal Component', fontsize=18)
    plt.ylabel('Eigenvalue', fontsize=18)
    plt.title(f'Eigenvalue Criterion (k = {n_components})', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.axhline(1, color="black", linewidth=2)
    plt.plot(n_components, eigenvalues[n_components - 1], "D", color="#6C8EBF", markersize=10)
    
    # 2. Proportion of Variance Explained (PVE) criterion
    plt.subplot(1, 3, 2)
    n_components_99 = np.argmax(cumsum >= 0.99) + 1
    plt.plot(np.arange(1, len(cumsum) + 1), cumsum, marker='o', linestyle='--')
    plt.plot(n_components_99, cumsum[n_components_99 - 1], "D", color="#6C8EBF", markersize=10)
    plt.xlabel('Principal Component', fontsize=18)
    plt.ylabel('Explained Variance', fontsize=18)
    plt.title(f'PVE Criterion (k = {n_components_99})', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.axhline(0.99, color="black", linewidth=2)
    
    # 3. Scree plot criterion (elbow method)
    plt.subplot(1, 3, 3)
    # Visual inspection suggests the "elbow" is around component 4
    scree_k = 4  
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, marker='o', linestyle='--')
    plt.plot(scree_k, pca.explained_variance_ratio_[scree_k - 1], "D", color="#6C8EBF", markersize=10)
    plt.xlabel('Principal Component', fontsize=18)
    plt.ylabel('PVE', fontsize=18)
    plt.title(f'Scree Plot Criterion (k = {scree_k})', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.tight_layout()
    plt.savefig('figures/ch06/choosing_k_pca.png', bbox_inches='tight', dpi=600)
    plt.savefig('figures/ch06/choosing_k_pca.pdf', bbox_inches='tight', dpi=600)
    
    return plt.gcf()

# Analyze the eigenvalues and explained variance for lnKD
eigenvalues = pca_lnkd.explained_variance_
cumsum = np.cumsum(pca_lnkd.explained_variance_ratio_)

# Visualize criteria for selecting the optimal number of components
component_selection_fig = visualize_optimal_component_selection(pca_lnkd, eigenvalues, cumsum)

# %% [markdown]
# We will now apply PCA using the chosen number of components (e.g., based on 99% PVE) and then use the Kennard-Stone algorithm on the reduced PCA space to select a representative test set. Mahalanobis distance can be beneficial in PCA space as PCs are uncorrelated but may have different variances.

# %%
def split_data_with_kennard_stone(X_dr_list, df_list, target_cols, test_size=12):
    """
    Split data into training and test sets using the Kennard-Stone algorithm.
    
    Parameters:
        X_dr_list (list): List of dimensionality-reduced feature matrices
        df_list (list): List of DataFrames for each target
        target_cols (list): List of target column names
        test_size (int): Number of samples to include in the test set
        
    Returns:
        tuple: (train_indices_list, test_indices_list, X_train_list, X_test_list, y_train_list, y_test_list)
    """
    train_indices_list = []
    test_indices_list = []
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    
    # For reproducibility, use fixed test indices from the paper
    test_indices_paper = [1, 5, 9, 12, 18, 23, 26, 31, 33, 35, 42, 47]
    
    # Process each target's data
    for i, (X_dr, df, target_col) in enumerate(zip(X_dr_list, df_list, target_cols)):
        # Use Kennard-Stone to select test samples
        if False:  # Set to True to use Kennard-Stone instead of paper indices
            test_indices = kennard_stone_algorithm(X_dr, test_size)
        else:
            test_indices = test_indices_paper
        
        # Create train/test masks
        mask = df.index.isin(test_indices)
        
        # Extract features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Standard scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[~mask])
        X_test = scaler.transform(X[mask])
        
        # Store results
        train_indices_list.append(df.index[~mask].tolist())
        test_indices_list.append(test_indices)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y[~mask])
        y_test_list.append(y[mask])
        
        print(f"Split data for {target_col}: {len(y_train_list[i])} train, {len(y_test_list[i])} test samples")
    
    return (train_indices_list, test_indices_list, X_train_list, 
            X_test_list, y_train_list, y_test_list)

# Split data into training and test sets
(train_indices_list, test_indices_list, X_train_list, 
 X_test_list, y_train_list, y_test_list) = split_data_with_kennard_stone(X_dr_list, df_list, target_cols)

# Unpack for easier access
lnkon_train_indices, lnkoff_train_indices, lnkd_train_indices = train_indices_list
lnkon_test_indices, lnkoff_test_indices, lnkd_test_indices = test_indices_list
X_lnkon_train, X_lnkoff_train, X_lnkd_train = X_train_list
X_lnkon_test, X_lnkoff_test, X_lnkd_test = X_test_list
y_lnkon_train, y_lnkoff_train, y_lnkd_train = y_train_list
y_lnkon_test, y_lnkoff_test, y_lnkd_test = y_test_list

# Sanity checks: the split must partition the compounds with no train/test overlap
for _Xtr, _Xte, _ytr, _yte, _df in zip(X_train_list, X_test_list, y_train_list, y_test_list, df_list):
    assert len(_ytr) + len(_yte) == len(_df), "train + test must cover every compound"
    assert _Xtr.shape[0] == len(_ytr) and _Xte.shape[0] == len(_yte), "X/y row mismatch"
assert not set(lnkd_train_indices) & set(lnkd_test_indices), "train/test indices overlap"

# %% [markdown]
# Let's visualize the distribution of train and test samples in our dimensionality-reduced space:

# %%
def visualize_data_split(X_dr, test_indices, explained_variance, target_name):
    """
    Visualize the distribution of train and test samples in 2D PCA space.
    
    Parameters:
        X_dr (numpy.ndarray): Dimensionality-reduced feature matrix
        test_indices (list): Indices of test samples
        explained_variance (list): Explained variance ratios
        target_name (str): Name of the target variable
        
    Returns:
        matplotlib.figure.Figure: Figure with the visualization
    """
    # Create a mask for test samples
    mask = np.zeros(X_dr.shape[0], dtype=bool)
    mask[test_indices] = True
    
    # Use the first two principal components for visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X_dr[~mask, 0], X_dr[~mask, 1], label='Training Set', color='#A20025', s=80)
    plt.scatter(X_dr[mask, 0], X_dr[mask, 1], color='#6C8EBF', marker="D", 
                label='Test Set', s=100)
    plt.title(f'{target_name} Train/Test Split', fontsize=22)
    plt.legend(fontsize=18)
    plt.box(False)
    plt.xlabel(f"PC1 ({explained_variance[0]:.2%})", fontsize=18)
    plt.ylabel(f"PC2 ({explained_variance[1]:.2%})", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.axvline(linewidth=3, color='black')
    plt.axhline(linewidth=3, color='black')
    
    plt.tight_layout()
    plt.savefig(f'figures/ch06/{target_name}_KSA_split.png', bbox_inches='tight', dpi=600)
    plt.savefig(f'figures/ch06/{target_name}_KSA_split.pdf', bbox_inches='tight', dpi=600)
    
    return plt.gcf()

# Visualize the data split for lnKD
lnkd_split_fig = visualize_data_split(X_lnkd_dr, lnkd_test_indices, lnkd_explained_variance, "lnKD")

# %%
def visualize_target_distribution_by_split():
    """
    Visualize the distribution of target variables in the train and test sets.
    
    Returns:
        matplotlib.figure.Figure: Figure with the visualization
    """
    from matplotlib.legend_handler import HandlerTuple
    
    # Combine all target values with their corresponding set (train/test)
    boxplot_y = (y_lnkd_train.tolist() + y_lnkd_test.tolist() + 
                y_lnkon_train.tolist() + y_lnkon_test.tolist() + 
                y_lnkoff_train.tolist() + y_lnkoff_test.tolist())
    
    boxplot_hue = (["Train"] * len(y_lnkd_train) + ["Test"] * len(y_lnkd_test) + 
                  ["Train"] * len(y_lnkon_train) + ["Test"] * len(y_lnkon_test) + 
                  ["Train"] * len(y_lnkoff_train) + ["Test"] * len(y_lnkoff_test))
    
    boxplot_x = (["lnKD"] * (len(y_lnkd_train) + len(y_lnkd_test)) + 
               ["lnkon"] * (len(y_lnkon_train) + len(y_lnkon_test)) + 
               ["lnkoff"] * (len(y_lnkoff_train) + len(y_lnkoff_test)))
    
    # Create the boxplot with overlaid strip plot
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x=boxplot_x, y=boxplot_y, hue=boxplot_hue, boxprops={'alpha': 0.4}, showfliers=False)
    sns.stripplot(x=boxplot_x, y=boxplot_y, hue=boxplot_hue, dodge=True, ax=ax, s=6, linewidth=1, alpha=.8)
    
    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[(handles[0], handles[2]), (handles[1], handles[3])],
              labels=['Train', 'Test'],
              loc='upper left', handlelength=4,
              handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=14)
    
    # Customize the plot
    ax.xaxis.set_tick_params(labelsize='x-large')
    ax.yaxis.set_tick_params(labelsize='x-large')
    plt.title('Distribution of Target Variables in Train and Test Sets', fontsize=18)
    
    plt.tight_layout()
    plt.savefig('figures/ch06/dist_response_vars.png', bbox_inches='tight', dpi=600)
    plt.savefig('figures/ch06/dist_response_vars.pdf', bbox_inches='tight', dpi=600)
    
    return plt.gcf()

# Visualize target distributions
target_dist_fig = visualize_target_distribution_by_split()

# %% [markdown]
# ## <b> <font color='#A20025'> 3️⃣ QSAR Modeling: Mapping Descriptors to Measurements
#

# %% [markdown]
# Quantitative Structure-Activity Relationship (QSAR) modeling aims to build mathematical models that relate the chemical structure (represented by descriptors) to biological activity or properties. Here, we'll replicate a workflow using Multiple Linear Regression (MLR) with feature selection (LASSO) and exhaustive search, and then explore Gradient Boosting Machines (GBMs)

# %% [markdown]
# ### <b> <font color='#A20025'> Exemplary QSAR Modeling Workflow

# %% [markdown]
# This workflow mimics a common QSAR approach involving:
# 1.  Splitting data (using indices from KSA).
# 2.  Scaling features.
# 3.  LASSO regression for initial feature selection (penalizes coefficients, shrinking some to zero).
# 4.  Exhaustive search using MLR on the LASSO-selected features to find the best small subset based on cross-validation ($Q^2$) and statistical significance (p-values).

# %% [markdown]
# First we define all of our helper functions:

# %%
def lasso_feature_selection(X_train, X_test, y_train, y_test):
    """
    Perform feature selection using Lasso regression with cross-validation.
    
    Parameters:
        X_train (numpy.ndarray): Training feature matrix
        X_test (numpy.ndarray): Test feature matrix
        y_train (pandas.Series): Training target values
        y_test (pandas.Series): Test target values
        
    Returns:
        tuple: (best_alpha, selected_features) Alpha parameter and indices of selected features
    """
    # Define the parameter grid for alpha values
    param_grid = {'alpha': np.logspace(2, -6, num=1000)}
    
    # Create a Lasso regressor
    lasso = Lasso()
    
    # Perform grid search with 5-fold cross-validation
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best alpha and best model
    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_
    
    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best alpha: {best_alpha:.6f}")
    print(f"Test R²: {r2:.4f}")
    
    # Get indices of non-zero coefficients (selected features)
    selected_features = np.where(best_model.coef_ != 0)[0]
    print(f"Selected {len(selected_features)} features with Lasso")
    
    return best_alpha, selected_features

def compute_feature_pvals(model, X, y):
    """
    Calculate p-values for linear regression coefficients.
    
    Parameters:
        model (sklearn.linear_model): Fitted linear regression model
        X (numpy.ndarray): Feature matrix
        y (pandas.Series): Target values
        
    Returns:
        numpy.ndarray: p-values for each coefficient
    """
    # Calculate the standard errors of the coefficients
    n = len(y)
    p = X.shape[1]
    dof = n - p - 1

    # Too few samples for the number of coefficients: the t-test is undefined
    if dof <= 0:
        return np.ones(p)

    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / dof

    # Use the Moore-Penrose pseudo-inverse rather than np.linalg.inv: with
    # collinear descriptors X.T @ X is singular and inv() raises LinAlgError,
    # which would abort the exhaustive search. pinv agrees with inv whenever the
    # matrix is well conditioned, so well-behaved models are unaffected.
    var_b = mse * (np.linalg.pinv(np.dot(X.T, X)).diagonal())

    # Clip tiny/negative variances so sqrt and the division below stay finite
    std_err = np.sqrt(np.clip(var_b, np.finfo(float).tiny, None))

    # Calculate the t-statistics and p-values
    t_stat = model.coef_ / std_err
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))

    return np.round(p_values, 3)

def exhaustive_mlr_search(X_train, X_test, y_train, y_test, selected_features, threshold=0.7, max_descs=7):
    """
    Perform exhaustive search to find the best MLR model.
    
    Parameters:
        X_train (numpy.ndarray): Training feature matrix
        X_test (numpy.ndarray): Test feature matrix
        y_train (pandas.Series): Training target values
        y_test (pandas.Series): Test target values
        selected_features (list): Indices of pre-selected features
        threshold (float): Minimum R² and Q² threshold
        max_descs (int): Maximum number of descriptors to include
        
    Returns:
        tuple: (best_model, best_features, best_score, best_p_values)
    """
    # Subset the data to only include pre-selected features
    X_train_subset = X_train[:, selected_features]
    X_test_subset = X_test[:, selected_features]
    
    best_score = -np.inf
    best_model = None
    best_p_values = None
    best_features = None
    best_combo = None
    
    print(f"Searching for best MLR model with up to {max_descs} descriptors...")
    
    # Generate all possible combinations of features
    searched_combinations = 0
    for r in range(2, min(max_descs + 1, len(selected_features) + 1)):
        for feature_combination in itertools.combinations(range(X_train_subset.shape[1]), r):
            searched_combinations += 1
            
            # Train a linear regression model
            model = LinearRegression()
            model.fit(X_train_subset[:, feature_combination], y_train)
            
            # Evaluate the model
            y_train_pred = model.predict(X_train_subset[:, feature_combination])
            y_test_pred = model.predict(X_test_subset[:, feature_combination])
            
            r2_train = r2_score(y_train, y_train_pred)
            q2_test = r2_score(y_test, y_test_pred)
            
            # Skip models that don't meet the R² and Q² thresholds
            if r2_train <= threshold or q2_test <= threshold:
                continue
            
            # Calculate p-values for coefficients
            p_values = compute_feature_pvals(model, X_train_subset[:, feature_combination], y_train)
            
            # Update best model if current model is better and all p-values are significant
            if q2_test > best_score and np.all(p_values < 0.05):
                best_score = q2_test
                best_model = model
                best_features = [selected_features[idx] for idx in feature_combination]
                best_p_values = p_values
                best_combo = feature_combination
    
    if best_model is None:
        print("No model found that meets the criteria. Consider lowering the threshold.")
        return None, None, None, None
    
    print(f"Searched {searched_combinations} combinations")
    print(f"Best Q² (test): {best_score:.4f}")
    print(f"Best feature combination: {best_combo}")
    
    return best_model, best_features, best_score, best_p_values

def visualize_williams_plot(model, X, y, test_indices, feature_indices, title):
    """
    Create a Williams plot for applicability domain assessment.
    
    Parameters:
        model (sklearn.linear_model): Fitted linear regression model
        X (numpy.ndarray): Feature matrix
        y (pandas.Series): Target values
        test_indices (list): Indices of test samples
        feature_indices (list): Indices of features used in the model
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: Figure with the Williams plot
    """
    # Select the features used in the model
    X_subset = X[:, feature_indices]
    
    # Calculate predictions and residuals
    y_pred = model.predict(X_subset)
    residuals = y - y_pred
    
    # Calculate leverage (hat matrix diagonal)
    leverage = np.diag(X_subset.dot(np.linalg.pinv(X_subset.T.dot(X_subset)).dot(X_subset.T)))
    
    # Calculate mean squared error
    mse = np.mean(residuals**2)
    
    # Calculate standardized residuals
    standardized_residuals = residuals / np.sqrt(mse * (1 - leverage))
    
    # Create a mask for test points
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_indices] = True
    
    # Plot Williams plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with different colors for train and test
    scp = plt.scatter(leverage, standardized_residuals, c=~test_mask, cmap='coolwarm', alpha=0.7, s=60)
    
    # Add warning lines
    plt.axhline(y=3, color='black', linestyle='--')
    plt.axhline(y=-3, color='black', linestyle='--')
    
    # Add leverage threshold line
    h_star = (3 * (len(feature_indices) + 1)) / len(y)
    plt.axvline(x=h_star, color='black', linestyle='--')
    
    # Add annotations
    plt.xlabel('Leverage', fontsize=12)
    plt.ylabel('Standardized Residuals', fontsize=12)
    plt.title(f'Williams Plot for {title}', fontsize=14)
    
    # Add legend
    plt.legend(handles=[scp.legend_elements()[0][0], scp.legend_elements()[0][-1]], 
               labels=['Test', 'Train'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'figures/ch06/williams_{title}.png', bbox_inches='tight', dpi=600)
    plt.savefig(f'figures/ch06/williams_{title}.pdf', bbox_inches='tight', dpi=600)
    
    return plt.gcf()

def qsar_modeling_workflow(X_train, X_test, y_train, y_test, test_indices, descriptor_names, target_name, threshold=0.7, max_descs=7):
    """
    Complete QSAR modeling workflow with feature selection and model evaluation.
    
    Parameters:
        X_train (numpy.ndarray): Training feature matrix
        X_test (numpy.ndarray): Test feature matrix
        y_train (pandas.Series): Training target values
        y_test (pandas.Series): Test target values
        test_indices (list): Indices of test samples
        descriptor_names (list): Names of descriptors
        target_name (str): Name of the target variable
        threshold (float): Minimum R² and Q² threshold
        max_descs (int): Maximum number of descriptors to include
        
    Returns:
        tuple: (best_model, best_features, best_score, feature_names)
    """
    print(f"\n===== QSAR Modeling for {target_name} =====")
    
    # Step 1: Lasso feature selection
    print("\nStep 1: Lasso Feature Selection")
    best_alpha, selected_features = lasso_feature_selection(X_train, X_test, y_train, y_test)
    
    # Step 2: Exhaustive MLR search
    print("\nStep 2: Exhaustive MLR Search")
    best_model, best_features, best_score, best_p_values = exhaustive_mlr_search(
        X_train, X_test, y_train, y_test, selected_features, threshold, max_descs)
    
    if best_model is None:
        return None, None, None, None
    
    # Get feature names
    feature_names = [descriptor_names[i] for i in best_features]
    
    # Print model details
    print("\nQSAR Model Details:")
    print(f"Features: {feature_names}")
    print(f"Coefficients: {best_model.coef_}")
    print(f"Intercept: {best_model.intercept_}")
    print(f"P-values: {best_p_values}")
    
    # Create Williams plot for applicability domain assessment
    williams_fig = visualize_williams_plot(
        best_model, np.concatenate([X_train, X_test]), 
        pd.concat([y_train, y_test]), test_indices, best_features, target_name)
    
    return best_model, best_features, best_score, feature_names


# %% [markdown]
# Let's run the QSAR workflow:
#
# **😱 Long Running Code Block**
#
# The following code block may take a long time to run (up to 25 minutes). This is a great chance to read the corresponding section in the book! But if you want to save time, you can skip this cell and review the results printed out for each model. The sections after this cell do not depend on it running to completion and consider different models and tasks.

# %%
# Run QSAR modeling for each target
descriptor_names = df_list[0].drop(columns=target_cols[0]).columns.tolist()

# For lnkon
lnkon_model, lnkon_features, lnkon_score, lnkon_feature_names = qsar_modeling_workflow(
    X_lnkon_train, X_lnkon_test, y_lnkon_train, y_lnkon_test, 
    lnkon_test_indices, descriptor_names, "lnkon")

# For lnkoff
lnkoff_model, lnkoff_features, lnkoff_score, lnkoff_feature_names = qsar_modeling_workflow(
    X_lnkoff_train, X_lnkoff_test, y_lnkoff_train, y_lnkoff_test, 
    lnkoff_test_indices, descriptor_names, "lnkoff", threshold=0.6)

# For lnKD
lnkd_model, lnkd_features, lnkd_score, lnkd_feature_names = qsar_modeling_workflow(
    X_lnkd_train, X_lnkd_test, y_lnkd_train, y_lnkd_test, 
    lnkd_test_indices, descriptor_names, "lnKD", max_descs=4)

# %% [markdown]
# The results of running the QSAR workflow above (in case you decide to skip the cell) are as follows:
#
# ===== QSAR Modeling for lnkon =====
#
# Step 1: Lasso Feature Selection
#
# Best alpha: 0.079598
#
# Test R²: 0.1610
#
# Selected 21 features with Lasso
#
# Step 2: Exhaustive MLR Search
#
# Searching for best MLR model with up to 7 descriptors...
#
# Searched 198418 combinations
#
# Best Q² (test): 0.7924
#
# Best feature combination: (3, 7, 8, 14, 16, 17, 19)
#
# QSAR Model Details:
#
# Features: ['E_oop', 'GCUT_PEOE_0', 'h_pKb', 'vsa_other', 'vsurf_DD23', 'vsurf_DW12', 'vsurf_IW3']
#
# Coefficients: [-0.87498913 -0.66337576  0.56893965 -1.72977477  0.50321844  0.5693903
#  -1.29009861]
#
# Intercept: 9.242536363277775
#
# P-values: [0.01  0.017 0.032 0.    0.033 0.037 0.   ]
#
# ===== QSAR Modeling for lnkoff =====
#
# Step 1: Lasso Feature Selection
#
# Best alpha: 0.211507
#
# Test R²: 0.3124
#
# Selected 12 features with Lasso
#
# Step 2: Exhaustive MLR Search
#
# Searching for best MLR model with up to 7 descriptors...
#
# Searched 3289 combinations
#
# Best Q² (test): 0.6711
#
# Best feature combination: (0, 1, 8)
#
# QSAR Model Details:
#
# Features: ['a_don', 'a_nN', 'vsurf_DD12']
#
# Coefficients: [-0.80040191 -1.04135819  0.55291156]
#
# Intercept: -2.665462416750001
#
# P-values: [0.001 0.    0.015]
#
# ===== QSAR Modeling for lnKD =====
#
# Step 1: Lasso Feature Selection
#
# Best alpha: 0.013308
#
# Test R²: 0.5749
#
# Selected 28 features with Lasso
#
# Step 2: Exhaustive MLR Search
#
# Searching for best MLR model with up to 4 descriptors...
#
# Searched 24129 combinations
#
# Best Q² (test): 0.8259
#
# Best feature combination: (7, 8, 13, 27)
#
# QSAR Model Details:
#
# Features: ['E_rsol', 'GCUT_PEOE_1', 'MNDO_HF', 'vsurf_IW4']
#
# Coefficients: [-0.76894516  1.09958228 -1.21625891  0.83105829]
#
# Intercept: -11.799190567666665
#
# P-values: [0.006 0.    0.    0.002]

# %% [markdown]
# The above run also produced and saved Williams plots for each target to `figures/ch06`. The Applicability Domain (AD) defines the chemical space where the QSAR model's predictions are considered reliable. The Williams plot visualizes the AD using standardized residuals (prediction error) versus leverage (a measure of a compound's influence on the model, related to its distance from the center of the training data). Compounds outside the AD (high leverage or large residual) should be treated with caution.

# %% [markdown]
# ### <b> <font color='#A20025'> Advanced Modeling with Gradient Boosting
#

# %% [markdown]
# GBMs (like XGBoost, LightGBM, CatBoost) are powerful ensemble methods often outperforming linear models, especially for complex relationships. They build models sequentially, with each new model correcting the errors of the previous ones. We'll perform a basic hyperparameter grid search for an XGBoost regressor as an example alternative to linear regression for QSAR modeling:
#

# %%
def train_gradient_boosting_model(X_train, X_test, y_train, y_test, target_name):
    """
    Train a gradient boosting regression model with hyperparameter tuning.
    
    Parameters:
        X_train (numpy.ndarray): Training feature matrix
        X_test (numpy.ndarray): Test feature matrix
        y_train (pandas.Series): Training target values
        y_test (pandas.Series): Test target values
        target_name (str): Name of the target variable
        
    Returns:
        tuple: (best_model, best_score, best_params)
    """
    print(f"\n===== Gradient Boosting for {target_name} =====")
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 250, 500],
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 4, 5, 7]
    }
    
    # Create XGBoost regressor
    model = XGBRegressor(random_state=RANDOM_SEED)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_score = r2_score(y_test, y_pred)
    
    print(f"Best parameters: {best_params}")
    print(f"Test R²: {test_score:.4f}")
    
    return best_model, test_score, best_params

# %%
# Train gradient boosting models for lnKD target
gb_lnkd_model, gb_lnkd_score, gb_lnkd_params = train_gradient_boosting_model(
    X_lnkd_train, X_lnkd_test, y_lnkd_train, y_lnkd_test, "lnKD")

# %% [markdown]
# ## <b> <font color='#A20025'> 4️⃣ Gradient Boosting Machines

# %% [markdown]
# While GBMs can be powerful predictors, understanding *why* they make certain predictions is crucial for gaining scientific insights. We'll explore methods to interpret a GBM model trained to distinguish known RNA binders from general protein binders.

# %% [markdown]
# ### <b> <font color='#A20025'> Informing the RNA-Binding Chemical Space

# %% [markdown]
# Let's use gradient boosting to explore the broader RNA-binding chemical space. We'll compare RNA binders with protein binders. However, we first need to spend some time downloading and preparing the data set that we will be using.
#
# You may either download the "data" folder from https://figshare.com/articles/dataset/Machine_Learning_Informs_RNA-Binding_Chemical_Space/20401974?file=36477873 and move "Mordred_BindingDB_3D.csv" and "Mordred_ROBIN_RNA_Binder_3D.csv" to "data/ch06" on your locale machine or in Colab.
#
# Or, alternatively, you can run the following cell, which will download the files and move the relevant files to "data"/cho06" (may take ~5 minutes).

# %% tags=["skip-execution"]
# Download the entire dataset from Figshare
# This downloads the zip file containing all the data
# !wget "https://figshare.com/ndownloader/files/36477873" -O dataset.zip

# Unzip the downloaded file
# !unzip -q dataset.zip -d temp_data

# Find and move only the files you want to keep
import os
import shutil

# Define the target files
target_files = [
    "Mordred_BindingDB_3D.csv",
    "Mordred_ROBIN_RNA_Binder_3D.csv"
]

# Search for the target files in the extracted directory
for root, dirs, files in os.walk("temp_data"):
    for file in files:
        if file in target_files:
            source_path = os.path.join(root, file)
            dest_path = os.path.join("data/ch06/", file)
            print(f"Moving {source_path} to {dest_path}")
            shutil.copy2(source_path, dest_path)

# Clean up the temporary directory
# !rm -rf temp_data
# !rm dataset.zip

# %% [markdown]
# After downloading and moving the two required data files to "data/ch06", run the following cell to load, process, and combine the data files into the Pandas DataFrame that we will be working with for the remainder of the chapter. The following code should take 2 to 3 minutes to run:

# %%
def prepare_binder_comparison_dataset():
    """
    Prepare a dataset comparing RNA binders and protein binders.
    
    Returns:
        pandas.DataFrame: Combined dataset of RNA and protein binders
    """
    print("Loading RNA and protein binder datasets...")
    
    # Read in ROBIN RNA binders and BindingDB protein binders
    df_robin_rna = pd.read_csv("data/ch06/Mordred_ROBIN_RNA_Binder_3D.csv")
    df_bindingdb = pd.read_csv("data/ch06/Mordred_BindingDB_3D.csv")
    
    # Remove the name column
    df_robin_rna = df_robin_rna.drop('name', axis=1)
    df_bindingdb = df_bindingdb.drop('name', axis=1)
    
    # Delete duplicate rows
    df_robin_rna.drop_duplicates(keep="first", inplace=True)
    df_bindingdb.drop_duplicates(keep="first", inplace=True)
    
    # Add library names to the dataframes
    df_robin_rna["Library"] = "RNA Binding"
    df_bindingdb["Library"] = "Protein Binding"
    
    # Concat the dataframes and convert feature columns to numeric
    df_combined = pd.concat([df_robin_rna, df_bindingdb])
    feature_columns = df_combined.columns.tolist()[:-1]
    df_combined[feature_columns] = df_combined[feature_columns].apply(pd.to_numeric, errors='coerce')
    
    print(f"Combined dataset: {df_combined.shape[0]} compounds, {df_combined.shape[1]} features")
    
    # Delete columns that have feature values missing in more than 40% of rows
    df_combined = df_combined.dropna(thresh=df_combined.shape[0]*0.6, axis=1)
    
    # Replace missing and infinity values by median of each column
    df_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_combined = df_combined.fillna(df_combined.median(numeric_only=True))
    
    # Reset the index
    df_combined = df_combined.reset_index(drop=True)
    
    print(f"After cleaning: {df_combined.shape[0]} compounds, {df_combined.shape[1]} features")
    
    # Optionally, save the cleaned and combined dataset
    # df_combined.to_csv("data/ch06/CH06_ROBIN_RNA_vs_BindingDB.csv", index=False)
    
    return df_combined

# Prepare the dataset
binder_df = prepare_binder_comparison_dataset()

# Sanity checks: both classes present, no NaN/inf left in the feature matrix
assert set(binder_df["Library"].unique()) == {"RNA Binding", "Protein Binding"}
assert binder_df.drop(columns=["Library"]).isna().sum().sum() == 0, "features still contain NaNs"
print(binder_df["Library"].value_counts())

# %% [markdown]
# #### <b> <font color='#A20025'> Training a Classifier and Assessing Feature Importance

# %% [markdown]
# We train an XGBoost classifier to distinguish between RNA and protein binders using 10-fold stratified cross-validation to get a robust estimate of performance (using Area Under the Precision-Recall Curve, AUPRC, suitable for imbalanced datasets). We then examine the feature importances provided by XGBoost.

# %% [markdown]
# **😱 Long Running Code Block**
#
# The following code block may take a long time to run (up to 25 minutes) when running with `cross_validate=True`. If you don't want to wait, turn the 10-fold cross validation off by passing in `cross_validate=False` when running the `build_rna_protein_binder_classifier`. This will then just train the final model directly, which will take ~2 minutes.

# %%
import pickle

def build_rna_protein_binder_classifier(df, cross_validate=True):
    """
    Build a classifier to distinguish RNA binders from protein binders.
    
    Parameters:
        df (pandas.DataFrame): Combined dataset of RNA and protein binders
        
    Returns:
        tuple: (best_model, X, y, auprc_scores)
    """
    print("Preparing ML dataset...")
    
    # Prepare features and labels
    X = df.values[:, :-1].astype('float64')
    
    # Convert target to binary (1 for RNA binding, 0 for protein binding)
    y = df["Library"].map({"RNA Binding": 1, "Protein Binding": 0}).values
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define model
    model = XGBClassifier(random_state=RANDOM_SEED)
    
    # Use stratified k-fold cross-validation
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import average_precision_score
    
    cv = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)
    auprc_scores = []
    
    if cross_validate:
        print("Training classifier with 10-fold cross-validation...")
        for idx, (train, test) in enumerate(cv.split(X_scaled, y)):
            # Clone a FRESH, unfitted estimator for every fold. Re-calling .fit()
            # on one shared object makes each fold's score depend on the object's
            # prior state, so folds are no longer independent; clone() copies the
            # hyperparameters only and guarantees a clean model per fold.
            fold_model = clone(model)
            fold_model.fit(X_scaled[train], y[train])

            # Predict probabilities
            y_pred = fold_model.predict_proba(X_scaled[test])[:, 1]

            # Calculate AUPRC (area under precision-recall curve)
            auprc = average_precision_score(y[test], y_pred)
            auprc_scores.append(auprc)

            print(f"Fold {idx+1} AUPRC: {auprc:.4f}")
    else:
        print("Skipping cross-validation, loading 10-fold results from artifacts/ch06/binder_data.pkl...")
        with open("artifacts/ch06/binder_data.pkl", "rb") as f:
            auprc_scores = pickle.load(f)
        print("Skipping cross-validation, training on all data...")
    
    # Train final model on all data
    final_model = XGBClassifier(random_state=RANDOM_SEED)
    final_model.fit(X_scaled, y)
    
    # Add feature names for later interpretation
    final_model.get_booster().feature_names = df.drop(columns=['Library']).columns.tolist()
    
    print(f"10-Fold Average AUPRC: {np.mean(auprc_scores):.4f} ± {np.std(auprc_scores):.4f}")
    
    return final_model, X_scaled, y, auprc_scores

# Build RNA-protein binder classifier
binder_model, X_binder, y_binder, auprc_scores = build_rna_protein_binder_classifier(binder_df, cross_validate=True)

# %% [markdown]
# **Code for Figure 6.14.** Feature importance plots from the trained XGBoost model. XGBoost provides several importance metrics:
# *   **Gain:** Average gain (improvement in accuracy/reduction in loss) across all splits where the feature was used.
# *   **Cover:** Average coverage (number of samples affected by splits) across all splits where the feature was used.
# *   **Weight (Frequency):** Number of times the feature was used to split a node.

# %%
def visualize_feature_importance(model):
    """
    Visualize feature importance for the XGBoost model.
    
    Parameters:
        model (xgboost.XGBClassifier): Trained XGBoost model
        
    Returns:
        matplotlib.figure.Figure: Figure with the visualization
    """
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Common arguments for plot_importance
    format_args = {"max_num_features": 10, "grid": False, "values_format": "{v:.0f}"}
    
    # 1. Gain (improvement in accuracy brought by a feature)
    xgb.plot_importance(model, importance_type='gain', ax=axs[0], **format_args)
    axs[0].set_title('Feature Importance by Gain', fontsize=14)
    
    # 2. Coverage (relative quantity of observations affected by a feature)
    xgb.plot_importance(model, importance_type='cover', ax=axs[1], **format_args)
    axs[1].set_title('Feature Importance by Coverage', fontsize=14)
    
    # 3. Frequency (number of times a feature appears in trees)
    xgb.plot_importance(model, importance_type='weight', ax=axs[2], **format_args)
    axs[2].set_title('Feature Importance by Frequency', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('figures/ch06/xgb_plot_importances.png', bbox_inches='tight', dpi=600)
    plt.savefig('figures/ch06/xgb_plot_importances.pdf', bbox_inches='tight', dpi=600)
    
    return fig

# Visualize feature importance
importance_fig = visualize_feature_importance(binder_model)

# %% [markdown]
# ### <b> <font color='#A20025'> Model-agnostic Interpretation

# %% [markdown]
# While model-specific importances are useful, model-agnostic methods like Partial Derivatives and SHAP (SHapley Additive exPlanations) can provide deeper insights applicable to various models.

# %% [markdown]
# #### <b> <font color='#A20025'> Partial Derivative Analysis

# %% [markdown]
# Partial derivatives estimate how sensitive the model's output (prediction probability) is to small changes in a specific input feature. A large positive derivative suggests increasing the feature value increases the probability of the positive class (RNA binding).
#
# > **⚠️ A caveat on finite differences for tree ensembles.** A gradient-boosted tree is a *piecewise-constant* function: its prediction only changes when an input crosses a split threshold. Its true derivative is therefore zero almost everywhere, and undefined (a step) exactly at the thresholds. The central difference we compute below with a finite `delta` is really asking *"does nudging this feature by ±delta push the sample across a split?"* — so a non-zero value flags a feature whose split boundary happens to lie within `delta` of this particular sample, and the magnitude depends on the arbitrary choice of `delta`. That makes this a useful *local sensitivity* probe, but **not** a gradient in the sense it would be for a smooth model (e.g. a neural network or a linear model). For a principled, model-agnostic attribution on tree ensembles, prefer the SHAP values computed in the next section.

# %%
def analyze_partial_derivatives(model, X, feature_names, threshold=1e-2):
    """
    Calculate partial derivatives to identify influential features.
    
    Parameters:
        model (xgboost.XGBClassifier): Trained model
        X (numpy.ndarray): Feature matrix
        feature_names (list): Names of features
        threshold (float): Threshold for including features
        
    Returns:
        matplotlib.figure.Figure: Figure with the visualization
    """
    def partial_derivative(model, X, feature_idx, delta=5e-2):
        """Central finite-difference derivative of one feature, for a single row."""
        X_plus = X.copy()
        X_plus[:, feature_idx] += delta
        y_pred_plus = model.predict_proba(X_plus)[:, 1]

        X_minus = X.copy()
        X_minus[:, feature_idx] -= delta
        y_pred_minus = model.predict_proba(X_minus)[:, 1]

        # predict_proba returns one probability per ROW, so this difference is a
        # length-1 array for the single-row X passed in below. Index it to return
        # a genuine scalar: assigning a (1,)-shaped array into a scalar slot of
        # `partial_derivatives` relies on NumPy's deprecated array-to-scalar coercion.
        return ((y_pred_plus - y_pred_minus) / (2 * delta))[0]

    print("Calculating partial derivatives...")

    # Calculate partial derivatives for first sample (an RNA binder)
    partial_derivatives = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        partial_derivatives[i] = partial_derivative(model, X[:1], i)

    # Filter features by threshold
    mask = partial_derivatives > threshold
    if not mask.any():
        # A gradient-boosted tree is piecewise-constant, so most finite differences
        # are exactly zero (see the caveat above). Bail out rather than draw an
        # empty bar chart.
        print(f"No feature has a partial derivative above {threshold}; nothing to plot.")
        return None
    selected_features = feature_names[mask]
    selected_derivatives = partial_derivatives[mask]
    
    # Sort by derivative value
    sort_idx = np.argsort(selected_derivatives)[::-1]
    selected_features = selected_features[sort_idx]
    selected_derivatives = selected_derivatives[sort_idx]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(selected_derivatives)), selected_derivatives, tick_label=selected_features)
    plt.xlabel('Descriptor', fontsize=12)
    plt.ylabel('Partial Derivative', fontsize=12)
    plt.xticks(rotation=60)
    plt.title('Partial Derivative Analysis for RNA Binding', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('figures/ch06/partial_derivatives_analysis.png', bbox_inches='tight', dpi=600)
    plt.savefig('figures/ch06/partial_derivatives_analysis.pdf', bbox_inches='tight', dpi=600)
    
    return plt.gcf()

# Analyze partial derivatives
feature_names = np.array(binder_df.drop(columns=['Library']).columns)
derivatives_fig = analyze_partial_derivatives(binder_model, X_binder, feature_names)

# %% [markdown]
# #### <b> <font color='#A20025'>  SHAP Value Analysis

# %% [markdown]
# SHAP (SHapley Additive exPlanations) is a game theory-based approach to explain individual predictions. It assigns each feature an importance value (SHAP value) representing its contribution to pushing the prediction away from the baseline (average prediction). Positive SHAP values push towards the positive class (RNA binding), negative values push towards the negative class (protein binding).

# %% [markdown]
# Common SHAP visualizations we'll consider:
# *   **Beeswarm Plot:** Shows SHAP value distribution for each feature. Color indicates feature value (high/low). Reveals global importance and feature effects.
# *   **Bar Plot:** Shows mean absolute SHAP value per feature (global importance). Can be split by class (cohorts).
# *   **Scatter/Dependence Plot:** Shows SHAP value for a single feature versus its value. Color can represent interaction with another feature.
# *   **Waterfall Plot:** Explains a *single* prediction by showing how each feature contributes to moving from the baseline to the final output.

# %% [markdown]
# **😱 Long Running Code Block** 
#
# Potentially Long Running Cell: Calculating SHAP values can be computationally intensive,
# especially for large datasets or complex models.

# %%
def analyze_with_shap(model, X, feature_names, max_display=10, sample_size=5000):
    """
    Apply SHAP analysis to interpret the model.
    
    Parameters:
        model (xgboost.XGBClassifier): Trained model
        X (numpy.ndarray): Feature matrix
        feature_names (list): Names of features
        max_display (int): Maximum number of features to display
        sample_size (int): Number of samples to use for SHAP analysis
        
    Returns:
        tuple: (explainer, shap_values)
    """
    print("Computing SHAP values...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model, data=None)
    
    # Compute SHAP values for a subset of samples (for efficiency)
    X_sample = X[:sample_size] if X.shape[0] > sample_size else X
    shap_values = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value

    # Normalize the SHAP output shape across SHAP/XGBoost versions before building
    # the Explanation. For a BINARY classifier, some versions return a single
    # (n_samples, n_features) array while others return a per-class list of two
    # arrays, or a 3-D (n_samples, n_features, n_classes) array. Reduce every case
    # to the positive class so the plots below always get a 2-D array.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[-1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[..., -1]
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.asarray(expected_value)[-1]

    assert shap_values.shape == X_sample.shape, (
        f"Expected SHAP values shaped like X {X_sample.shape}, got {shap_values.shape}"
    )
    print(f"SHAP values shape: {shap_values.shape} (matches X)")

    # Create SHAP explanation object with feature names
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X_sample,
        feature_names=list(feature_names)
    )
    
    # 1. Beeswarm plot (global feature importance)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, 
                      feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Plot: Global Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/ch06/shap_beeswarmplot.png', bbox_inches='tight', dpi=600)
    plt.savefig('figures/ch06/shap_beeswarmplot.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    
    # Get binder types for the samples
    binder_type = ["RNA Binding" if y == 1 else "Protein Binding" for y in y_binder[:sample_size]]
    
    # 2. Bar plot (cohort analysis)
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_explanation.cohorts(binder_type).abs.mean(0), show=False)
    plt.title("SHAP Bar Plot: Feature Importance by Binder Type", fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/ch06/shap_barplot.png', bbox_inches='tight', dpi=600)
    plt.savefig('figures/ch06/shap_barplot.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    
    # 3. Scatter plot (dependence plot for most important feature)
    # Rank features with plain NumPy: the mean |SHAP| over samples is unambiguous,
    # whereas the semantics of Explanation.argsort() have shifted between SHAP
    # releases (and returning an Explanation, not an int, breaks the indexing below).
    plt.figure(figsize=(12, 8))
    most_important_feature = int(np.argmax(np.abs(shap_values).mean(axis=0)))
    shap.plots.scatter(shap_explanation[:, most_important_feature], color=shap_explanation, show=False)
    plt.title(f"SHAP Scatter Plot: Effect of {feature_names[most_important_feature]}", fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/ch06/shap_scatterplot.png', bbox_inches='tight', dpi=600)
    plt.savefig('figures/ch06/shap_scatterplot.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    
    # 4. Waterfall plot (local explanation for a single example)
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_explanation[1], max_display=max_display, show=False)
    plt.title("SHAP Waterfall Plot: Feature Contributions for a Single Molecule", fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/ch06/shap_waterfall.png', bbox_inches='tight', dpi=600)
    plt.savefig('figures/ch06/shap_waterfall.pdf', bbox_inches='tight', dpi=600)
    plt.close()
    
    print("SHAP analysis completed and visualizations saved.")
    
    return explainer, shap_values

# Run SHAP analysis
shap_explainer, shap_values = analyze_with_shap(binder_model, X_binder, feature_names)

# %% [markdown]
# ## <b> <font color='#A20025'> Summary

# %% [markdown]
# In this chapter, we conducted a case study involving small molecule binding to an RNA target (HIV TAR) and a classification task distinguishing RNA from protein binders. Key steps included:
#
# 1.  **Data Preparation:** Loading data, handling molecular structures (protomers, tautomers, conformers), calculating Boltzmann-weighted 3D descriptors.
# 2.  **Feature Refinement & Splitting:** Log-transforming targets, removing redundant/correlated features, using PCA for dimensionality reduction/visualization, and applying the Kennard-Stone algorithm for representative train/test splitting.
# 3.  **QSAR Modeling:**
#      *   Built MLR models using LASSO feature selection and exhaustive search.
#      *   Assessed Applicability Domain using Williams plots.
#      *   Developed a more complex XGBoost regression model with hyperparameter tuning.
# 4.  **Classification & Interpretation:**
#      *   Trained an XGBoost classifier to distinguish RNA vs. protein binders using pre-calculated descriptors.
#      *   Interpreted the classifier using model-specific feature importances (gain, cover, weight) and model-agnostic methods (Partial Derivatives, SHAP values).
#      *   Visualized SHAP results (beeswarm, bar, scatter, waterfall plots) to understand global and local feature contributions.
#
# This case study demonstrates a comprehensive workflow from data processing and representation to modeling and interpretation, highlighting advanced techniques relevant for structure-based and ligand-based drug discovery tasks, especially in scenarios with complex structural considerations or the need for model interpretability.
#

# %% [markdown]
# ## <b> <font color='#A20025'> Interactive Exploration & Further Study

# %% [markdown]
# 1.  **Conformer Generation Parameters:** How do changes in `numConfs`, the energy window in `generate_conformers_for_states`, or the force field used for optimization affect the final descriptors and model performance?
# 2.  **Dimensionality Reduction:** Explore other DR techniques like t-SNE or UMAP for visualizing the chemical space. How do they compare to PCA? Does using DR features directly in the models (instead of just for splitting) improve performance?
# 3.  **KSA Variants:** Implement or investigate KSA using different distance metrics (e.g., Tanimoto distance on fingerprints if 2D descriptors were used). How does the choice of distance metric influence the split representativeness?
# 4.  **QSAR Models:** Compare the MLR and XGBoost results more thoroughly. Try other regression algorithms (e.g., Support Vector Regression, Random Forest Regressor). How do their performances and interpretations differ?
# 5.  **GBM Interpretation:** For the classification task, use SHAP interaction values (`shap.TreeExplainer(model).shap_interaction_values(X)`) to analyze feature interactions. Which features show strong synergistic or antagonistic effects?
# 6.  **Descriptor Impact:** Re-run the QSAR or classification workflows using only 2D descriptors. How does this impact performance and interpretation compared to using the complex 3D descriptors? When might 3D descriptors be most beneficial?
# 7.  **Error Analysis:** For the regression models, analyze the predictions on the test set. Are there specific chemical classes or regions of the applicability domain where the models perform poorly? Why might that be?
# 8.  **Mordred BindingDB Test Compounds** Evaluate our final model on the Mordred Test Compounds csv. What are the results?
#
# *See book for additional exercises.*

# %% [markdown]
# ## <b> <font color='#A20025'> References

# %% [markdown]
# [1] Cai, Z., Zafferani, M., Akande, O. M., Hargrove, A. E. (2022). Quantitative Structure–Activity Relationship (QSAR) Study Predicts Small-Molecule Binding to RNA Structure. J. Med. Chem., 65(10). Doi: 10.1021/acs.jmedchem.2c00254
#
# [2] Yazdani, K., Jordan, D., Yang, M., et al. (2023). Machine Learning Informs RNA-Binding Chemical Space. Angewandte Chemie (International ed. in English), 62(11). doi:10.1002/anie.202211358
#
# [3] Ropp, P. J., Kaminsky, J. C., Yablonski, S., Durrant, J. D. (2019) Dimorphite-DL: An open-source program for enumerating the ionization states of drug-like small molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9
#
# [4] Sitzmann, M., et al., “Tautomerism in Large Databases.”, JCAMD 24:521 (2010) https://doi.org/10.1007/s10822-010-9346-4
#
# [5] Landrum, G. (2023). Working with conformers. RDKit blog. https://greglandrum.github.io/rdkit-blog/posts/2023-02-04-working-with-conformers.html
#
# [6] Riniker, S.; Landrum, G. A. “Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation” J. Chem. Inf. Comp. Sci. 55:2562-74 (2015) https://doi.org/10.1021/acs.jcim.5b00654
#
# [7] Landrum, G. (2022). Variability of PMI descriptors. RDKit blog. https://greglandrum.github.io/rdkit-blog/posts/2022-06-22-variability-of-pmi-descriptors.html
#
# [8] Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for Machine Learning. Chapter 10. Cambridge: Cambridge University Press.
#
# [9] Bagheri, A. Principal Component Analysis (PCA) from Scratch. https://bagheri365.github.io/blog/Principal-Component-Analysis-from-Scratch/
#
# [10] Hansch, C. and Fujita, T. (1964) J. Am. Chem. Soc., 86, 1616–1626
#
# [11] Free, S.M. and Wilson, J.W. (1964) J. Med. Chem., 7, 395–399
#
# [12] Polishchuk, P. (2017) Interpretation of Quantitative Structure-Activity Relationship Models: Past, Present, and Future. J. Chem. Inf. 57(11). doi:10.1021/acs.jcim.7b00274
#
# [13] Moriwaki, H., Tian, YS., Kawashita, N. et al. Mordred: a molecular descriptor calculator. J Cheminform 10, 4 (2018). https://doi.org/10.1186/s13321-018-0258-y
#
# [14] Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS 2022. 
#
# [15] Molnar, C. Interpretable Machine Learning. https://christophm.github.io/interpretable-ml-book/
#
# [16] Rhiju Das, H Wayment-Steele, Do Soon Kim, Christian Choe, Bojan Tunguz, Walter Reade, Maggie Demkin. (2020). OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction. Kaggle. https://kaggle.com/competitions/stanford-covid-vaccine
#
# [17] Rhiju Das, Shujun He, Rui Huang, Jill Townley, Rachael Kretsch, Thomas Karagianes, John Nicol, Grace Nye, Christian Choe, Jonathan Romano, Maggie Demkin, Walter Reade, and Eterna players. (2023). Stanford Ribonanza RNA Folding. Kaggle. https://kaggle.com/competitions/stanford-ribonanza-rna-folding
#
# [18] Bagnolini, G., Luu, T. B., Hargrove, A., E. (2024). Recognizing the power of machine learning and other computational methods to accelerate progress in small molecule targeting of RNA. RNA 20: 473-488. Doi:10.1261/rna.079497.122


