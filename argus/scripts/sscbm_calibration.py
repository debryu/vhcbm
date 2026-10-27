"""Compute concept calibration for SSCBM runs: ECCE-R, ECCE-MAD, ECE.

Mirrors the procedure used for every other method (scripts/linear_probe_calibration.py,
scripts/calibration.py): load the saved concept probe, recompute test concept probs, then
per concept-class column run _cumulative -> (kuiper=ECCE-R, ks=ECCE-MAD) and _equiprob ->
(ece1, ece2). Flat-average over all columns (the paper's convention). Also records the
confidence-ECE (metrics/ece.py) for completeness.

Normalization matches scripts/sscbm_knn.py exactly (z-score; +L2 for dinov3) so the reloaded
probe sees the embeddings it was trained on.

Output: rebuttal_drafts/sscbm_calibration_N340.jsonl (one row per run) + a printed summary.
"""
import os, sys, json, glob
# NOTE: cluster-specific path, replace with your own location
sys.path.insert(0, '/leonardo/home/userexternal/ndebole0/projects/argus')
sys.path.insert(0, '/leonardo/home/userexternal/ndebole0/projects/CQA')
import torch
import torch.nn.functional as F
import numpy as np
from argus.models.linear import load_trainer
from argus.metrics.ecce import _cumulative, _equiprob
from argus.metrics.ece import ece_from_probs
from CQA.datasets import GenericDataset

# NOTE: cluster-specific path, replace with your own location
FAST = os.environ.get('FAST', '/leonardo_scratch/fast/IscrC_ARGO')
ACT = os.path.join(FAST, 'activations')
# NOTE: cluster-specific path, replace with your own location
OUT = '/leonardo_scratch/fast/IscrC_ARGO/argobackup/argus/rebuttal_drafts/sscbm_calibration_N340.jsonl'
os.makedirs('./figs/ece', exist_ok=True)   # _cumulative/_equiprob write plots there

# Jul 30: resnet18_cub added (cub only) so the E1-symmetric ResNet18 table gets a real ECCE-R for
# its SSCBM rows instead of `--`. Same normalisation contract as rebuttal_probe_metrics.py:
# standardise always, extra L2 for dinov3 only.
VLM = {'clip': 'ViT-L/14', 'dinov3': 'dinov3', 'resnet18_cub': 'resnet18_cub'}

def load_embeddings(ds_name, split, backbone):
    model_name = VLM[backbone]
    emb = torch.load(os.path.join(ACT, f'{ds_name}_{split}_{model_name.replace("/","%")}.pth'),
                     map_location='cpu')
    emb = (emb - emb.mean(0, keepdim=True)) / emb.std(0, keepdim=True)
    if backbone == 'dinov3':
        emb = F.normalize(emb, p=2, dim=-1)
    return emb

# cache test concepts per dataset
_concepts = {}
def test_concepts(ds_name):
    if ds_name not in _concepts:
        d = GenericDataset(ds_name=ds_name, split='test')
        c = torch.stack([d[i][1] for i in range(len(d))], dim=0)
        _concepts[ds_name] = c
    return _concepts[ds_name]

def calibrate_run(run_dir, ds_name, backbone):
    probe = load_trainer(os.path.join(run_dir, 'probe.pth'))
    gts = test_concepts(ds_name)
    emb = load_embeddings(ds_name, 'test', backbone)
    with torch.no_grad():
        probs, _ = probe.eval(emb, gts)
    preds = probs.detach().cpu()
    gts_cpu = gts.cpu()
    eccer, eccemad, ece1 = [], [], []
    for i in range(preds.shape[1]):
        # _cumulative MUTATES its probs arg (tie-break noise) -> pass a fresh clone each call
        k, ks, _ = _cumulative(probs=preds[:, i].clone(), labels=gts_cpu[:, i],
                               majorticks=5, minorticks=20,
                               filename=f'./figs/ece/sscbm_{ds_name}_{backbone}_{i}_ECCE.pdf')
        e1, _ = _equiprob(probs=preds[:, i].clone(), labels=gts_cpu[:, i], nbins=20,
                          filename=f'./figs/ece/sscbm_{ds_name}_{backbone}_{i}_ECE.pdf')
        eccer.append(k); eccemad.append(ks); ece1.append(e1)
    conf_ece = ece_from_probs(preds, gts_cpu)['ece']
    return {
        'eccer': float(np.mean(eccer)),        # ECCE-R (paper's ECCE column)
        'eccemad': float(np.mean(eccemad)),    # ECCE-MAD
        'ece': float(np.mean(ece1)),           # binned ECE (equiprob, l1) — same as other methods
        'conf_ece': float(conf_ece),           # confidence-ECE (metrics/ece.py), for reference
    }

if __name__ == '__main__':
    # Append incrementally + skip already-done runs, so a kill/restart loses nothing.
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            line = line.strip()
            if line:
                r = json.loads(line)
                done.add((r['variant'], r['backbone'], r['dataset'], r['seed']))
        print(f"resuming: {len(done)} runs already done")
    n = 0
    for variant, tag in [('knn', 'knn'), ('selftrain', 'pseudo')]:
        for backbone in ['clip', 'dinov3', 'resnet18_cub']:
            for run in sorted(glob.glob(f'{FAST}/results/SSCBM/{backbone}/*-340-{tag}-*')):
                base = os.path.basename(run)
                ds = base.split('-')[0]
                seed = int(base.split('SEED=')[1])
                if (variant, backbone, ds, seed) in done:
                    continue
                try:
                    cal = calibrate_run(run, ds, backbone)
                except Exception as e:
                    print(f"FAIL {run}: {e}", flush=True); continue
                row = {'variant': variant, 'backbone': backbone, 'dataset': ds, 'seed': seed, **cal}
                with open(OUT, 'a') as f:            # append + flush per run
                    f.write(json.dumps(row) + '\n')
                n += 1
                print(f"{variant:9s} {backbone:6s} {ds:10s} s{seed}  "
                      f"ECCE-R={cal['eccer']:.3f}  ECCE-MAD={cal['eccemad']:.3f}  "
                      f"ECE={cal['ece']:.3f}  confECE={cal['conf_ece']:.3f}", flush=True)
    print(f"\nWrote {n} new rows to {OUT}")
