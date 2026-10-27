"""D1b - concept interventions on the CBM-frozen baseline, under the IDENTICAL protocol used for
VH-CBM in eval_interventions.py.

Why this exists: reporting intervention curves only for our own model invites the objection
"the baseline would improve just as much" (or "you picked the encoding that flatters you").
This runs the same interventions on the frozen-probe CBM so the comparison is like-for-like.

Everything is inference on artifacts that already exist in each FAIRCBM run dir
(probe.pth + W_g.pt + b_g.pt) - no retraining.

Protocol imported verbatim from eval_interventions.py: same group mapping, same one-hot and
empirical-percentile GT encodings, same top-k group selection, same fractions.
Orderings: random (5 permutations) and ucp (predictive entropy). NOT gpstd - the frozen probe has
no predictive uncertainty, which is precisely the asymmetry we are pointing at.

Usage: python eval_interventions_baseline.py -dataset cub -backbone clip -seed 132
Output: appends one JSON row to rebuttal_drafts/D1_interventions_baseline.jsonl
"""
import sys, os, json, argparse

# NOTE: cluster-specific path, replace with your own location
sys.path.insert(0, '/leonardo_scratch/fast/IscrC_ARGO/argobackup')
sys.path.insert(0, '/leonardo/home/userexternal/ndebole0/projects/CQA')

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from CQA.datasets import GenericDataset
from argus.models.linear import load_trainer

# reuse the exact protocol
from eval_interventions import (FRACTIONS, N_RANDOM_REPS, CONCEPT_STRUCTURES,
                                group_columns, build_gt_encoding, to_empirical,
                                group_scores_ucp, intervene, eval_preds)

FAST = os.environ['FAST']
ACT = os.path.join(FAST, 'activations')
# NOTE: cluster-specific path, replace with your own location
OUT = '/leonardo_scratch/fast/IscrC_ARGO/argobackup/argus/rebuttal_drafts/D1_interventions_baseline.jsonl'
N = 340
BACKBONE_ACT = {'clip': 'ViT-L/14', 'dinov3': 'dinov3', 'resnet18_cub': 'resnet18_cub'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-dataset', type=str, required=True)
    p.add_argument('-backbone', type=str, default='clip', choices=list(BACKBONE_ACT))
    p.add_argument('-seed', type=int, required=True)
    p.add_argument('-logger', type=str, default='WARNING')
    a = p.parse_args()
    logger.remove()
    logger.add(sys.stderr, filter=lambda r: r["level"].no >= logger.level(a.logger).no)
    return a


def load_embeddings(ds_name, split, backbone):
    model_name = BACKBONE_ACT[backbone]
    emb = torch.load(os.path.join(ACT, f'{ds_name}_{split}_{model_name.replace("/", "%")}.pth'),
                     map_location='cpu')
    emb = (emb - emb.mean(0, keepdim=True)) / emb.std(0, keepdim=True)
    if model_name in ['dinov3']:
        emb = F.normalize(emb, p=2, dim=-1)
    return emb


def collect(ds_name, split):
    d = GenericDataset(ds_name=ds_name, split=split)
    cs, ys = [], []
    for i in range(len(d)):
        _, c, y = d[i]
        cs.append(c); ys.append(y)
    return torch.stack(cs, 0), torch.stack(ys, 0)


def to_task_input(probs, concept_structure):
    """Same layout the baseline's task head was trained on (scripts/fair_cbm.py:84-94):
    binary -> [1-p, p] pairs, multiclass -> softmax block as-is."""
    cols = []
    for _, (start, end) in concept_structure.items():
        if end - start + 1 == 1:
            p = probs[:, start].unsqueeze(-1)
            cols.append(torch.cat([1 - p, p], dim=1))
        else:
            cols.append(probs[:, start:end + 1])
    return torch.cat(cols, dim=1)


if __name__ == '__main__':
    args = parse_args()
    ds, bb, seed = args.dataset, args.backbone, args.seed
    cstruct = CONCEPT_STRUCTURES[ds]
    run_dir = os.path.join(FAST, 'results/FAIRCBM', bb, f'{ds}-{N}-random-SEED={seed}')
    assert os.path.isdir(run_dir), f'missing {run_dir}'

    torch.manual_seed(seed)
    np.random.seed(seed)

    probe = load_trainer(os.path.join(run_dir, 'probe.pth'))
    W_g = torch.load(os.path.join(run_dir, 'W_g.pt'), map_location='cpu')
    b_g = torch.load(os.path.join(run_dir, 'b_g.pt'), map_location='cpu')

    test_c, test_y = collect(ds, 'test')
    train_emb = load_embeddings(ds, 'train', bb)
    # The train split is only needed for the empirical-percentile reference distribution, i.e. for
    # the *activations*. probe.eval uses `labels` solely for the loss it returns (models/linear.py:
    # 87-124) - probs come from the logits - so pass zeros and skip iterating the raw train images
    # (celeba/cub would otherwise load >100k images just to read concept vectors we never use).
    dummy = torch.zeros(train_emb.shape[0], test_c.shape[1])
    with torch.no_grad():
        test_probs, _ = probe.eval(load_embeddings(ds, 'test', bb), test_c)
        train_probs, _ = probe.eval(train_emb, dummy)

    x = to_task_input(test_probs.detach().cpu().float(), cstruct)
    train_means = to_task_input(train_probs.detach().cpu().float(), cstruct)
    gts = test_c.cpu()

    groups, binary = group_columns(ds, cstruct, x.shape[1])

    def predict(inp):
        with torch.no_grad():
            return (inp.float() @ W_g.T + b_g).argmax(dim=1).cpu()

    base_acc, base_f1 = eval_preds(predict(x), test_y)
    print(f'[{ds}/{bb}/s{seed}] unintervened: acc={base_acc:.4f} f1={base_f1:.4f}', flush=True)

    E_onehot = build_gt_encoding(x, gts, groups, binary)
    E_emp = to_empirical(E_onehot, train_means)
    ucp = group_scores_ucp(x, groups, binary)

    G = len(groups)
    results = []
    for enc_name, E in [('onehot', E_onehot), ('empirical', E_emp)]:
        for frac in FRACTIONS:
            n_int = int(round(frac * G))
            entry = {'encoding': enc_name, 'frac': frac, 'n_intervened': n_int}
            # random ordering: mean +- std over N_RANDOM_REPS permutations
            accs, f1s = [], []
            for rep in range(N_RANDOM_REPS):
                S = torch.rand(x.shape[0], G)
                a, f = eval_preds(predict(intervene(x, E, S, n_int, groups)), test_y)
                accs.append(a); f1s.append(f)
            entry['random'] = {'label_acc': float(np.mean(accs)), 'label_acc_std': float(np.std(accs)),
                               'label_f1': float(np.mean(f1s))}
            a, f = eval_preds(predict(intervene(x, E, ucp, n_int, groups)), test_y)
            entry['ucp'] = {'label_acc': a, 'label_f1': f}
            # gpstd deliberately absent: the frozen probe has no predictive uncertainty
            results.append(entry)
            print(f'  {enc_name:<10} frac={frac:<5} random={entry["random"]["label_acc"]:.4f} '
                  f'ucp={a:.4f}', flush=True)

    row = {'model': 'cbm-frozen', 'dataset': ds, 'backbone': bb, 'seed': seed, 'N': N,
           'n_groups': G, 'base_acc': base_acc, 'base_f1': base_f1, 'results': results}
    with open(OUT, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(f'-> appended to {OUT}', flush=True)
