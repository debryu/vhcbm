# VH-CBM: Vision-plus-Human-guided Concept Bottleneck Models

Code for *Concepts Worth Having: Refining VLM-Guided Concept Bottleneck Models with Minimal Annotations* (NeurIPS 2026).

VH-CBM places one Gaussian Process per concept on top of a frozen vision encoder (CLIP ViT-L/14, DINOv3 or ResNet18), fits it on a small set of expert concept annotations (optionally selected actively from the GP uncertainty), and trains a sequential linear task layer on the GP concept predictions.

## Repository layout

```
argus/            VH-CBM (the Python package is called `argus`)
  OMCours.py        GP concept layer training + active/random acquisition loop
  train_argocbm*.py task layer (CBM head) on top of a trained GP
  eval_argocbm*.py  evaluation: F1(Y), F1(C), ROC-AUC(C), DCI disentanglement
  eval_interventions*.py  test-time concept interventions
  models/ metrics/ utils/ datasets/
  scripts/          embeddings, linear probes, calibration, baselines, plotting
  *.slurm           launch scripts used on the cluster (SLURM arrays)
CQA/              baselines: CBM (ResNet18), LaBo, LF-CBM, VLG-CBM, datasets and concept metrics
concepts/         concept vocabularies used for every dataset
requirements.txt  exact environment used for the experiments
```

## Installation

Python 3.11, CUDA 12.6.

```
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs both the `argus` and `CQA` packages.

## Paths and environment variables

The code was run on a SLURM cluster and contains hardcoded paths. Every such line is marked with a comment starting with `# NOTE: cluster-specific`:

```
grep -rn "NOTE: cluster-specific" .
```

Replace them with your own locations. In particular:

- `CQA/config.py`: dataset folders (`DATASETS_FOLDER_PATHS`), concept vocabularies (`CONCEPT_SETS`, point them to `concepts/`), saved baseline models and the shared activation folder.
- `*.slurm`: SLURM account, log directories and Python environment.

Environment variables read by the scripts:

| Variable | Meaning |
|---|---|
| `SCRATCH` | root for GP checkpoints (`$SCRATCH/models`) and GP result JSONs (`$SCRATCH/results/GPs/GPs`) |
| `FAST` | root for activations (`$FAST/activations`) and evaluation results (`$FAST/results/...`) |
| `VLM` | backbone: `ViT-L/14` (CLIP), `dinov3`, or `resnet18_cub` |
| `DS_NAME` | `shapes3d`, `celeba`, `dermamnist`, `cub` |
| `SEED` | random seed |
| `ACQ_FN` | `activeours` (active concept selection) or `randomc` (random) |
| `KERNEL` | GP kernel (`RBF` in all experiments) |
| `LENGTHSCALE`, `OUTPUTSCALE`, `LEARN_KERNEL` | GP initialisation overrides (sensitivity study only; defaults 1.41 / 1.0 / learned) |

## Pipeline

1. **Embeddings.** Compute and cache the frozen-encoder embeddings for all splits: `argus/scripts/save_activations.py` (see `save_activations.slurm`).
2. **GP concept layer.** `cd argus && SEED=... DS_NAME=... VLM=... ACQ_FN=activeours KERNEL=RBF python OMCours.py`. This starts from 40 annotated images and adds concept annotations for 60 images per iteration (the active variant spreads that budget over a pool of `int(60/0.633) = 94` images, see `OMCours.py`). It writes one checkpoint per iteration and a JSON log with the concept metrics. `train_gp.slurm` runs the seed/dataset grid.
3. **Task layer.** `python train_argocbm.py -dataset <ds> -model_path <GP checkpoint> -save_dir <dir> -seed <seed>` (grid: `train_argocbm.slurm`).
4. **Evaluation.** `python eval_argocbm.py -load_dir <task layer dir>` writes `metrics.txt` (F1(Y), F1(C), ROC-AUC(C), DIS) (grid: `eval_argocbm.slurm`).
5. **Calibration.** `python scripts/calibration.py -dataset <ds> -model_path <GP checkpoint> -save_dir <dir>` writes per-concept ECCE-R (`kuiper`), ECCE-MAD (`ks`), ECE1 and ECE2. The scripts `scripts/*_calibration.py` do the same for each baseline, and `scripts/compute_calibration.py` loops over all models.

Reported numbers are averaged over concepts within a run, then the mean and the sample standard deviation are taken across runs.

## Reproducing the paper

| Paper | Code |
|---|---|
| Fig. 2, Fig. 3, Fig. 4 (concept F1 vs. supervision; active vs. random acquisition) | GP JSON logs from step 2, aggregated with `scripts/newplotting.py` and `scripts/collect.py`, plotted with `scripts/plotting.ipynb` |
| LP-@ and LP-All baselines | `VLM=<backbone> python scripts/lp.py` |
| Zero-shot CLIP scores | `scripts/clip_scores.py` |
| CBM, LaBo, LF-CBM, VLG-CBM | `CQA/train.py` (examples in `CQA/experiments/experiments.md`), evaluated with `scripts/baselines.py` / `evaluate_baselines.slurm` |
| Table 2, Table 7 (main results, active / random acquisition) | steps 2-5; random acquisition uses `ACQ_FN=randomc` (DINO task heads: `randomc_cbm_dino.slurm`) |
| Table 3 (GP initialisation) | `sensitivity.slurm` (sets `LENGTHSCALE` / `OUTPUTSCALE` for `OMCours.py`) |
| Table 4 (frozen-backbone CBM) | `scripts/fair_cbm.py` (`fair_cbm.slurm`); ECCE-R and DIS with `scripts/faircbm_run_all.py` |
| Table 5 (ResNet18 backbone, CUB) | `resnet_gp.slurm`, `resnet_cbm.slurm`, `rn18_calibration.slurm`; frozen ResNet18 CBM from `fair_cbm.slurm` and `scripts/rebuttal_probe_metrics.py` |
| Table 6 (SSCBM) | `scripts/sscbm_knn.py` (`sscbm_knn.slurm`), calibration `scripts/sscbm_calibration.py`, DIS `scripts/rebuttal_probe_metrics.py` |
| Table 8 (all calibration metrics) | step 5 for every method |
| Table 9 (concept interventions) | `eval_interventions.py`, `eval_interventions_baseline.py`, `eval_interventions_coop.py` (CooP policy), `scripts/plot_coop.py`; `argus/data/_gp_ckpt_index.json` lists the GP runs used |

### Seeds

| Runs | Seeds |
|---|---|
| VH-CBM, CLIP (active) | 132, 213, 312 |
| VH-CBM, DINO (active and random) | 111, 222, 333 (DermaMNIST: 222, 333, 444) |
| VH-CBM, CLIP (random) | 123, 231, 321 |
| VH-CBM, ResNet18 (CUB) | 132, 213, 312 |
| Frozen-backbone CBM, SSCBM | 132, 213, 312 |
| LP-@ | the seed loop in `scripts/lp.py` |

## Notes

- GP training runs on cached embeddings and does not fine-tune the encoder.
- The GPs use a fixed number of inducing points (40), initialised with MiniBatchKMeans on the embeddings.
- Some scripts write their outputs into a `rebuttal_drafts/` folder; these paths are marked as cluster-specific and can be changed freely.
