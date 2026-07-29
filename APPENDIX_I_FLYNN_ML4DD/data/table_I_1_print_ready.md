# Table I.1 — Production models used in Appendix I

Licences read from their distribution points on 2026-07-27; code and weights read separately.
Runtimes are observed wall clock from the runs that produced this appendix's figures, never estimates.
`not recorded` means no timing survives in the artifacts or job ledger; it is not an omission of a known value.

| Model | Task | Code | Weights | Observed runtime | Device |
|---|---|---|---|---|---|
| FlowMol3 | unconditional 3D molecule generation | MIT | **unstated** | 1240 s (full timestep sweep total) | local CPU |
| DiffSBDD | pocket-conditioned ligand generation | MIT | CC-BY-4.0 | 286 s (denovo 20 ligands) | local CPU |
| RFdiffusion | protein backbone generation | BSD | BSD (covers weights explicitly) | 887 s (scaling 24 backbones 60 90 120 150res T50) | Modal A100 |
| ProteinMPNN | inverse folding (sequence design) | MIT | MIT (in-repo) | not recorded | Modal A100 |
| ESMFold | single-sequence structure prediction | MIT | MIT | not recorded | Modal A100 |
| Boltz-2 | complex structure + binding affinity | MIT | MIT | 161 s (msa backed run recorded in artifact) | Modal A100 |
| DiffDock | blind ligand docking | MIT | MIT (repo weights) | not recorded | not run |
| Boltz-2 (hosted NIM) | complex structure + affinity (hosted) | n/a (service) | NVIDIA service terms | 37 s (smoke test) | NVIDIA-hosted |
| DiffDock (hosted NIM) | blind ligand docking (hosted) | n/a (service) | NVIDIA service terms | 29 s (redock 20 ligands 10 poses each) | NVIDIA-hosted |

## Licence notes that matter

1. {"model": "FlowMol3", "code": "MIT", "weights": "UNSTATED", "severity": "high", "why": "Absence of a grant is not a permissive grant. A reader redistributing or building on the checkpoints has no stated permission to rely on. Strictly more restrictive in effect than the MIT code licence, because MIT's explicit grant does not reach these files.", "evidence": "bits.csb.pitt.edu/files/FlowMol/trained_models_v3.1/readme.md fetched this session: 2283 chars, zero licence/copyright/terms matches."}
2. {"model": "DiffSBDD", "code": "MIT", "weights": "CC-BY-4.0", "severity": "medium", "why": "CC-BY-4.0 adds an attribution-and-indicate-changes obligation on redistribution of the weights and adaptations that MIT does not impose in the same form. Both remain commercial-friendly.", "evidence": "Zenodo REST record 8183747 metadata.license = {'id': 'cc-by-4.0'}; metadata.rights = null."}
3. {"model": "Boltz-2 via hosted NIM", "code": "MIT (repo)", "weights": "NVIDIA service terms (API Trial ToS + Community/Open Model licence)", "severity": "high", "why": "The hosted path is materially more restrictive than the repo: testing-and-evaluation-only without a Subscription, and a prohibition on using outputs to develop or improve other AI models without written approval. The repo grants academic and commercial use outright.", "evidence": "NVIDIA API Trial Terms of Service PDF, 9 pages, read this session; NVIDIA Community Model License text read this session."}
4. {"model": "DiffDock via hosted NIM", "code": "MIT (repo)", "weights": "NVIDIA licence over NVIDIA-trained weights", "severity": "high", "why": "Not a licence divergence over the same artefact but something a reader is more likely to get wrong: different weights entirely, under NVIDIA terms, served over a trial API that is not for production absent a subscription.", "evidence": "build.nvidia.com/mit/diffdock model card: 'the model v2.2.0 is trained by NVIDIA using PLINDER and SAIR'."}

Full per-model detail, including every licence source URL, per-timestep runtimes and the six
`null` runtimes with their reasons, is in `assets/model_inventory.json`.