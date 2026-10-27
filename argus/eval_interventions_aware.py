"""D1c - is the non-monotonic intervention curve fixable by intervention-aware head training?

A reviewer will notice that VH-CBM's intervention curve peaks at 50% and *falls* when every concept
is replaced by ground truth (cub: 0.805 -> 0.916 @50% -> 0.815 @100%, reproducibly in all 3 seeds).

The concept layer cannot be the cause: at 100% intervention the concept vector *is* the ground
truth, so the concept predictor is perfect by construction. The whole drop sits in the task head,
which was fitted on the GP's smooth posterior probabilities and is then evaluated on two-valued
(5th/95th-percentile) inputs -- the sequential-CBM distribution shift documented by Shin et al.

If that diagnosis is right, exposing the head to interventions *during* its training should remove
the decline without changing the concept layer at all. This script tests exactly that: it retrains
only the final linear layer on the same saved concept means, with a random fraction of concept
groups replaced by their ground-truth encoding in each batch, and re-measures the curve.

Nothing about the GP is retrained; the concept layer, its budget and its metrics are untouched.

Usage: python eval_interventions_aware.py -dataset cub -model_path <ckpt.pt>
Output: appends to rebuttal_drafts/D1_interventions_aware.jsonl
"""
import sys, os, json, argparse, time

# NOTE: cluster-specific path, replace with your own location
sys.path.insert(0, '/leonardo_scratch/fast/IscrC_ARGO/argobackup')
sys.path.insert(0, '/leonardo/home/userexternal/ndebole0/projects/CQA')

import torch
import numpy as np
from loguru import logger
from CQA.datasets import GenericDataset
from argus.models.argocbm import ArgoBackbone

from eval_interventions import (FRACTIONS, N_RANDOM_REPS, CONCEPT_STRUCTURES, group_columns,
                               build_gt_encoding, to_empirical, group_scores_ucp,
                               group_scores_gpstd, intervene, eval_preds)

# NOTE: cluster-specific path, replace with your own location
OUT = '/leonardo_scratch/fast/IscrC_ARGO/argobackup/argus/rebuttal_drafts/D1_interventions_aware.jsonl'
EPOCHS, LR, BS, PATIENCE = 300, 1e-3, 1024, 20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-dataset', required=True)
    p.add_argument('-model_path', required=True)
    p.add_argument('-logger', default='WARNING')
    a = p.parse_args()
    logger.remove()
    logger.add(sys.stderr, filter=lambda r: r["level"].no >= logger.level(a.logger).no)
    return a


def labels(ds, split):
    d = GenericDataset(ds_name=ds, split=split)
    return torch.stack([d[i][2] for i in range(len(d))], dim=0).long()


def train_head(X, Y, Xv, Yv, groups, E=None, Ev=None, aware=False, seed=0, device='cuda'):
    """Linear head, same recipe as models/argocbm.py:188-266 (Adam + early stop on val loss).
    aware=True: each batch, replace a random number of concept groups (uniform in 0..G) with their
    GT encoding, per sample. The head therefore sees both un-intervened and fully-intervened inputs.
    """
    torch.manual_seed(seed)
    G = len(groups)
    n_cls = int(Y.max().item()) + 1
    head = torch.nn.Linear(X.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    lossf = torch.nn.CrossEntropyLoss()
    X, Y, Xv, Yv = X.to(device), Y.to(device), Xv.to(device), Yv.to(device)
    if E is not None:
        E = E.to(device)
    best, best_state, bad = float('inf'), None, 0
    for ep in range(EPOCHS):
        head.train()
        perm = torch.randperm(X.shape[0], device=device)
        for i in range(0, X.shape[0], BS):
            idx = perm[i:i + BS]
            bx, by = X[idx], Y[idx]
            if aware:
                # per-sample random number of intervened groups, uniform over 0..G
                k = torch.randint(0, G + 1, (bx.shape[0],), device=device)
                scores = torch.rand(bx.shape[0], G, device=device)
                thresh = scores.sort(dim=1, descending=True).values.gather(
                    1, (k.clamp(max=G - 1)).unsqueeze(1))
                gmask = (scores >= thresh) & (k.unsqueeze(1) > 0)
                colmask = torch.zeros_like(bx, dtype=torch.bool)
                for gid, (cols, _) in enumerate(groups):
                    colmask[:, cols] = gmask[:, gid].unsqueeze(1)
                bx = torch.where(colmask, E[idx], bx)
            opt.zero_grad()
            loss = lossf(head(bx), by)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vl = lossf(head(Xv), Yv).item()
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    return head


def curve(head, x, E, scores, groups, y, device='cuda'):
    out = []
    G = len(groups)
    for frac in FRACTIONS:
        n = int(round(frac * G))
        xi = intervene(x, E, scores, n, groups)
        with torch.no_grad():
            pred = head(xi.to(device).float()).argmax(1).cpu()
        acc, f1 = eval_preds(pred, y)
        out.append({'frac': frac, 'label_acc': acc, 'label_f1': f1})
    return out


if __name__ == '__main__':
    a = parse_args()
    ds, mp = a.dataset, a.model_path
    seed = int(mp.split('SEED=')[1].replace('.pt', ''))
    cs = CONCEPT_STRUCTURES[ds]
    t0 = time.time()

    bk = ArgoBackbone(mp)
    tr, va, te = (bk.get_concept_preds(s) for s in ('train', 'val', 'test'))
    ytr, yva, yte = labels(ds, 'train'), labels(ds, 'val'), labels(ds, 'test')

    Xtr, Xva, Xte = (d['concept_means'].float() for d in (tr, va, te))
    groups, binary = group_columns(ds, cs, Xte.shape[1])

    # GT encodings, percentiles always taken from the TRAIN activations (as in eval_interventions)
    ref = Xtr
    Etr = to_empirical(build_gt_encoding(Xtr, tr['concept_gts'], groups, binary), ref)
    Eva = to_empirical(build_gt_encoding(Xva, va['concept_gts'], groups, binary), ref)
    Ete = to_empirical(build_gt_encoding(Xte, te['concept_gts'], groups, binary), ref)
    gpstd = group_scores_gpstd(te['concept_stds'].float(), groups)

    # Reproduce the published task-head recipe (train_argocbm.py:211-253): the head is fitted on
    # 30 Monte-Carlo samples of the GP posterior, mean + std*eps -- the variance is NOT an input
    # feature (that variant exists at train_argocbm.py:250 but is commented out), it enters as the
    # spread of the training inputs. Validation/test use the deterministic means.
    # Both arms below get this same MC augmentation, so the only difference between them remains
    # the intervention augmentation.
    NUM_MC = 30
    torch.manual_seed(seed)
    Xtr_mc = torch.cat([Xtr + tr['concept_stds'].float() * torch.randn_like(Xtr)
                        for _ in range(NUM_MC)], dim=0)
    ytr_mc = ytr.repeat(NUM_MC)
    Etr_mc = Etr.repeat(NUM_MC, 1)      # GT encoding is deterministic; tile to match
    print(f"head training set: {tuple(Xtr_mc.shape)} ({NUM_MC} MC samples of the GP posterior)",
          flush=True)

    rows = {}
    for aware in (False, True):
        head = train_head(Xtr_mc, ytr_mc, Xva, yva, groups,
                          E=Etr_mc if aware else None, aware=aware, seed=seed)
        rows['aware' if aware else 'standard'] = curve(head, Xte, Ete, gpstd, groups, yte)
        tag = 'intervention-aware' if aware else 'standard'
        accs = ' '.join(f"{r['label_acc']:.4f}" for r in rows['aware' if aware else 'standard'])
        pk = max(rows['aware' if aware else 'standard'], key=lambda r: r['label_acc'])['frac']
        print(f"  {tag:<20} {accs}   peak@{pk}", flush=True)

    row = {'dataset': ds, 'seed': seed, 'model_path': mp, 'encoding': 'empirical',
           'ordering': 'gpstd', **rows, 'elapsed': round(time.time() - t0, 1)}
    with open(OUT, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(f"-> appended to {OUT}", flush=True)
