"""Fill the remaining probe-path metric gaps for the rebuttal tables.

Two jobs, both inference-only on already-saved artifacts (no retraining):

  B1  SSCBM-KNN (+ self-train ablation) @N=340: disentanglement (DIS).
      ECCE was already computed by scripts/sscbm_calibration.py -> sscbm_calibration_N340.jsonl.
  B2  CBM-frozen ResNet18, cub @N=340: ECCE-R *and* DIS.
      Its metrics.txt only carries confidence-ECE (metrics/ece.py), a different quantity.

Protocol is copied verbatim from $FAST/results/FAIRCBM_metrics_work/faircbm_compute_metrics.py
(the script that produced the published FairCBM clip/dinov3 numbers) so the new rows are
comparable to the existing ones:
  - ECCE-R = mean over concept-class columns of _cumulative(...)[0]  (kuiper), flat column average
  - DIS    = DCI_wrapper on (test concept probs -> GT concepts), train_test_ratio=0.7
  - embeddings: z-score, plus L2 only for dinov3 (matches lp.py / fair_cbm.py / sscbm_knn.py)

Output: rebuttal_drafts/probe_metrics_gap.jsonl  (append-only, resumable)
"""
import sys, os, json, time, glob

# NOTE: cluster-specific path, replace with your own location
sys.path.insert(0, '/leonardo_scratch/fast/IscrC_ARGO/argobackup')
sys.path.insert(0, '/leonardo/home/userexternal/ndebole0/projects/CQA')

import torch
import torch.nn.functional as F
import numpy as np
from argus.models.linear import load_trainer
from argus.metrics.ecce import _cumulative
from argus.metrics.dci import DCI_wrapper
from CQA.datasets import GenericDataset

FAST = os.environ["FAST"]
ACT = os.path.join(FAST, "activations")
# NOTE: cluster-specific path, replace with your own location
OUT = '/leonardo_scratch/fast/IscrC_ARGO/argobackup/argus/rebuttal_drafts/probe_metrics_gap.jsonl'
TMP = '/leonardo_scratch/fast/IscrC_ARGO/argobackup/argus/figs/ece'
os.makedirs(TMP, exist_ok=True)

N = 340
SEEDS = [132, 213, 312]
# backbone tag -> activation model name
BACKBONE_ACT = {'clip': 'ViT-L/14', 'dinov3': 'dinov3', 'resnet18_cub': 'resnet18_cub'}


def load_embeddings(ds_name, split, backbone):
    model_name = BACKBONE_ACT[backbone]
    sanitized = model_name.replace('/', '%')
    emb = torch.load(os.path.join(ACT, f'{ds_name}_{split}_{sanitized}.pth'), map_location='cpu')
    emb = (emb - torch.mean(emb, dim=0, keepdim=True)) / torch.std(emb, dim=0, keepdim=True)
    if model_name in ['dinov3']:
        emb = F.normalize(emb, p=2, dim=-1)
    return emb


_concepts = {}


def test_concepts(ds_name):
    if ds_name not in _concepts:
        d = GenericDataset(ds_name=ds_name, split='test')
        _concepts[ds_name] = torch.stack([d[i][1] for i in range(len(d))], dim=0)
    return _concepts[ds_name]


def compute_ecer(test_probs, concept_gts, tag):
    """Flat average of ECCE-R over every concept-class column (the paper's convention)."""
    kuipers = []
    plot = os.path.join(TMP, f'_gap_{tag}.pdf')
    for i in range(test_probs.shape[1]):
        # _cumulative mutates its probs argument -> pass a fresh clone per column
        kuiper, ks, lenscale = _cumulative(probs=test_probs[:, i].detach().clone().cpu(),
                                           labels=concept_gts[:, i].clone().cpu(),
                                           majorticks=5, minorticks=20, filename=plot)
        kuipers.append(kuiper)
    if os.path.exists(plot):
        os.remove(plot)
    return float(np.mean(kuipers))


def compute_dis(test_probs, concept_gts, train_test_ratio=0.7):
    n = len(test_probs)
    idx = torch.randperm(n)
    ntr = int(n * train_test_ratio)
    tr, va = idx[:ntr], idx[ntr:]
    dci = DCI_wrapper(test_probs[tr], concept_gts[tr], test_probs[va], concept_gts[va], 'WARNING')
    return float(dci['disentanglement'])


def eval_probe(run_dir, ds_name, backbone, seed, tag, want_ecer, want_dis):
    torch.manual_seed(seed)
    probe = load_trainer(os.path.join(run_dir, 'probe.pth'))
    gts = test_concepts(ds_name)
    emb = load_embeddings(ds_name, 'test', backbone)
    with torch.no_grad():
        probs, _ = probe.eval(emb, gts)
    probs = probs.detach().cpu()
    gts = gts.cpu()
    out = {}
    if want_ecer:
        out['ecer'] = compute_ecer(probs, gts, tag)
    if want_dis:
        out['disentanglement'] = compute_dis(probs, gts)
    return out


def already_done():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r['model'], r['backbone'], r['dataset'], r['seed']))
    return done


def emit(row):
    with open(OUT, 'a') as f:
        f.write(json.dumps(row) + '\n')
    keys = [k for k in ('ecer', 'disentanglement') if k in row]
    vals = '  '.join(f"{k}={row[k]:.4f}" for k in keys)
    print(f"{row['model']:<18}{row['backbone']:<13}{row['dataset']:<11}s{row['seed']}  {vals}", flush=True)


def build_joblist():
    """One entry per row. Order matters: the rows the tables actually need come first, so a
    truncated run still yields a complete table (knn DIS, then ResNet18, then the self-train
    ablation)."""
    jobs = []
    for backbone in ['clip', 'dinov3']:
        for run in sorted(glob.glob(f'{FAST}/results/SSCBM/{backbone}/*-{N}-knn-*')):
            base = os.path.basename(run)
            jobs.append(('sscbm-knn', backbone, base.split('-')[0],
                         int(base.split('SEED=')[1]), run, False, True))
    for seed in SEEDS:
        jobs.append(('cbm-frozen', 'resnet18_cub', 'cub', seed,
                     f'{FAST}/results/FAIRCBM/resnet18_cub/cub-{N}-random-SEED={seed}', True, True))
    for backbone in ['clip', 'dinov3']:
        for run in sorted(glob.glob(f'{FAST}/results/SSCBM/{backbone}/*-{N}-pseudo-*')):
            base = os.path.basename(run)
            jobs.append(('sscbm-selftrain', backbone, base.split('-')[0],
                         int(base.split('SEED=')[1]), run, False, True))
    # Jul 30: SSCBM on resnet18_cub (cub only) — fills the DIS cells of the E1-symmetric ResNet18
    # table. APPENDED LAST ON PURPOSE: task indices are positional, so inserting these earlier
    # would renumber the existing 51 rows and invalidate the array indices in the slurm file.
    for tag, model in [('knn', 'sscbm-knn'), ('pseudo', 'sscbm-selftrain')]:
        for run in sorted(glob.glob(f'{FAST}/results/SSCBM/resnet18_cub/*-{N}-{tag}-*')):
            base = os.path.basename(run)
            jobs.append((model, 'resnet18_cub', base.split('-')[0],
                         int(base.split('SEED=')[1]), run, True, True))
    return jobs


def run_one_index(idx):
    """SLURM-array entry point: compute a single row and write it to its own file, so parallel
    tasks never race on a shared jsonl (the reference job used a process pool for the same reason)."""
    jobs = build_joblist()
    if idx >= len(jobs):
        print(f"index {idx} >= {len(jobs)} jobs; nothing to do")
        return
    model, backbone, ds, seed, run_dir, want_ecer, want_dis = jobs[idx]
    part_dir = os.path.join(os.path.dirname(OUT), 'probe_metrics_parts')
    os.makedirs(part_dir, exist_ok=True)
    part = os.path.join(part_dir, f'{model}_{backbone}_{ds}_{seed}.json')
    if os.path.exists(part):
        print(f"already done: {part}")
        return
    if not os.path.isdir(run_dir):
        print(f"MISSING {run_dir}")
        return
    t0 = time.time()
    m = eval_probe(run_dir, ds, backbone, seed, f'{model}_{ds}_{backbone}_{seed}',
                   want_ecer=want_ecer, want_dis=want_dis)
    row = {'model': model, 'backbone': backbone, 'dataset': ds, 'seed': seed, 'N': N, **m,
           'elapsed': round(time.time() - t0, 1)}
    with open(part, 'w') as f:
        json.dump(row, f)
    print(f"{model:<18}{backbone:<13}{ds:<11}s{seed}  "
          + '  '.join(f"{k}={row[k]:.4f}" for k in ('ecer', 'disentanglement') if k in row)
          + f"  ({row['elapsed']}s)", flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'index':
        run_one_index(int(sys.argv[2]))
        sys.exit(0)
    if sys.argv[1] == 'count':
        print(len(build_joblist()))
        sys.exit(0)
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    done = already_done()
    print(f"resuming: {len(done)} rows already present", flush=True)
    n = 0

    # ---- B1: SSCBM disentanglement (ECCE already done elsewhere) ----
    if which in ('all', 'b1'):
        for model, tag in [('sscbm-knn', 'knn'), ('sscbm-selftrain', 'pseudo')]:
            for backbone in ['clip', 'dinov3']:
                for run in sorted(glob.glob(f'{FAST}/results/SSCBM/{backbone}/*-{N}-{tag}-*')):
                    base = os.path.basename(run)
                    ds = base.split('-')[0]
                    seed = int(base.split('SEED=')[1])
                    if (model, backbone, ds, seed) in done:
                        continue
                    t0 = time.time()
                    try:
                        m = eval_probe(run, ds, backbone, seed, f'{model}_{ds}_{backbone}_{seed}',
                                       want_ecer=False, want_dis=True)
                    except Exception as e:
                        print(f"FAIL {run}: {type(e).__name__}: {e}", flush=True)
                        continue
                    emit({'model': model, 'backbone': backbone, 'dataset': ds, 'seed': seed,
                          'N': N, **m, 'elapsed': round(time.time() - t0, 1)})
                    n += 1

    # ---- B2: CBM-frozen ResNet18 (cub) — ECCE-R + DIS ----
    if which in ('all', 'b2'):
        for seed in SEEDS:
            run = f'{FAST}/results/FAIRCBM/resnet18_cub/cub-{N}-random-SEED={seed}'
            if ('cbm-frozen', 'resnet18_cub', 'cub', seed) in done:
                continue
            if not os.path.isdir(run):
                print(f"MISSING {run}", flush=True)
                continue
            t0 = time.time()
            try:
                m = eval_probe(run, 'cub', 'resnet18_cub', seed, f'rn18_cub_{seed}',
                               want_ecer=True, want_dis=True)
            except Exception as e:
                print(f"FAIL {run}: {type(e).__name__}: {e}", flush=True)
                continue
            emit({'model': 'cbm-frozen', 'backbone': 'resnet18_cub', 'dataset': 'cub', 'seed': seed,
                  'N': N, **m, 'elapsed': round(time.time() - t0, 1)})
            n += 1

    print(f"\nWrote {n} new rows to {OUT}", flush=True)
