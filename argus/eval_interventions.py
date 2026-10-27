"""
Test-time concept interventions on trained VH-CBM runs (Table 9).

Procedure follows Shin et al., "A Closer Look at the Intervention Procedure of
Concept Bottleneck Models" (ICML 2023): our task layer is trained SEQuentially on
SOFT (probability) concept activations, so intervention replaces the selected
groups' columns of `concept_means` with the ground-truth encoding in probability
space (one-hot). Because SEQ task layers never see exact one-hots at train time
(Shin et al., Sec. 4.4), we also report an 'empirical' encoding that maps GT to the
per-column 5th/95th percentile of the *train-split* concept activations.

Orderings (per sample, group-level selection = mean of member-concept scores,
Shin et al.'s G+S setting):
  - random: 5 random permutations, mean +- std
  - ucp:    entropy of the predicted concept distribution (available to any CBM)
  - gpstd:  GP predictive std (our model's uncertainty; the novel criterion)

Usage: python eval_interventions.py -load_dir <run dir with W_g.pt/b_g.pt>
Writes <load_dir>/interventions.json. Sanity: frac=0 must reproduce metrics.txt label_f1.
"""
from argus.models.argocbm import ArgoCBM
from CQA.utils.args_utils import load_args
import argparse
from loguru import logger
import sys
import json, os
import torch
from sklearn.metrics import classification_report
import numpy as np
from CQA.datasets import GenericDataset

FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_RANDOM_REPS = 5

CONCEPT_STRUCTURES = {
    'celeba': {f'{i+1}': (i, i) for i in range(39)},
    'cub': {f'{i}': (i, i) for i in range(112)},
    'dermamnist': {'finding': (0, 6)},
    'shapes3d': {
        'floor color': (0, 9),
        'wall color': (10, 19),
        'object color': (20, 29),
        'object size': (30, 37),
        'shape': (38, 41),
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-logger', type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("-load_dir", type=str, default=None, help="Trained ARGOCBM run dir")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, filter=lambda r: r["level"].no >= logger.level(args.logger).no)
    return args


def group_columns(dataset, concept_structure, n_cols):
    """Map each concept group to (means-columns, gt-columns).
    celeba/cub: 2 means-cols per binary concept [2i, 2i+1], gt col i.
    shapes3d/dermamnist: means-cols == gt-cols == the group's index range."""
    groups = []
    binary = dataset in ['celeba', 'cub']
    for gid, (name, (start, end)) in enumerate(concept_structure.items()):
        if binary:
            groups.append(([2 * gid, 2 * gid + 1], [gid]))
        else:
            cols = list(range(start, end + 1))
            groups.append((cols, cols))
    used = sum(len(c) for c, _ in groups)
    assert used == n_cols, f"column mapping mismatch: {used} != {n_cols}"
    return groups, binary


def build_gt_encoding(x, gts, groups, binary):
    """Fully-intervened input in probability space (one-hot GT)."""
    E = x.clone()
    for gid, (cols, gt_cols) in enumerate(groups):
        if binary:
            g = gts[:, gt_cols[0]].float()
            E[:, cols[0]] = 1.0 - g
            E[:, cols[1]] = g
        else:
            E[:, cols] = gts[:, gt_cols].float()
    return E


def to_empirical(E_onehot, train_means):
    """Map one-hot GT encoding onto the empirically observed activation range:
    1 -> per-column 95th pct of train activations, 0 -> 5th pct (Shin et al. Sec 4.4 caveat)."""
    p5 = torch.quantile(train_means, 0.05, dim=0)
    p95 = torch.quantile(train_means, 0.95, dim=0)
    return torch.where(E_onehot >= 0.5, p95.expand_as(E_onehot), p5.expand_as(E_onehot))


def group_scores_ucp(x, groups, binary, eps=1e-12):
    """Per-sample per-group entropy of the predicted concept distribution.

    Computed in float64 on purpose. With the previous float32 + eps=1e-8, `1 - eps` rounds to
    EXACTLY 1.0 (float32 resolution near 1 is ~1.2e-7), so a confident concept kept p = 1.0 and
    the second term became 0 * log(0) = 0 * -inf = NaN. torch.topk ranks NaN as largest, so every
    NaN group was queried FIRST -- i.e. the ordering became anti-uncertainty, selecting the most
    confident concepts. On CelebA/CLIP this hit 43573/778518 = 5.6% of (sample, group) entries.
    float64 makes 1 - 1e-12 representable, so p is strictly inside (0, 1) and the entropy is finite.
    """
    N = x.shape[0]
    S = torch.zeros(N, len(groups), dtype=torch.float64)
    xd = x.double()
    for gid, (cols, _) in enumerate(groups):
        if binary:
            p = xd[:, cols[1]].clamp(eps, 1 - eps)
            S[:, gid] = -(p * p.log() + (1 - p) * (1 - p).log())
        else:
            p = xd[:, cols].clamp_min(eps)
            p = p / p.sum(dim=1, keepdim=True)
            S[:, gid] = -(p * p.log()).sum(dim=1)
    assert torch.isfinite(S).all(), "non-finite entropy in group_scores_ucp"
    return S


def group_scores_gpstd(stds, groups):
    """Per-sample per-group mean GP predictive std."""
    N = stds.shape[0]
    S = torch.zeros(N, len(groups))
    for gid, (cols, _) in enumerate(groups):
        S[:, gid] = stds[:, cols].mean(dim=1)
    return S


def intervene(x, E, scores, n_int, groups):
    """Replace the n_int highest-scoring groups (per sample) with their E columns."""
    if n_int == 0:
        return x
    top = torch.topk(scores, n_int, dim=1).indices          # [N, n_int]
    gmask = torch.zeros(x.shape[0], len(groups), dtype=torch.bool)
    gmask.scatter_(1, top, True)
    colmask = torch.zeros_like(x, dtype=torch.bool)
    for gid, (cols, _) in enumerate(groups):
        colmask[:, cols] = gmask[:, gid].unsqueeze(1)
    return torch.where(colmask, E, x)


def predict(model, x, device):
    with torch.no_grad():
        logits = model.final_layer(x.to(device).float())
    return logits.argmax(dim=1).cpu()


def eval_preds(preds, test_y):
    cr = classification_report(test_y.numpy(), preds.numpy(), output_dict=True, zero_division=0)
    return float(cr['accuracy']), float(cr['macro avg']['f1-score'])


if __name__ == "__main__":
    args = parse_args()
    args_model = load_args(args)
    model = ArgoCBM(args_model)
    model.load()
    device = args_model.device

    ds_name = args_model.dataset
    concept_structure = CONCEPT_STRUCTURES[ds_name]

    test = model.backbone.get_concept_preds('test')
    train = model.backbone.get_concept_preds('train')

    test_y = []
    data_test = GenericDataset(ds_name=ds_name, split='test')
    for i in range(len(data_test)):
        test_y.append(data_test[i][2])
    test_y = torch.stack(test_y, dim=0).long()

    x = test['concept_means'].float()
    stds = test['concept_stds'].float()
    gts = test['concept_gts']
    G = len(concept_structure)
    groups, binary = group_columns(ds_name, concept_structure, x.shape[1])

    E_onehot = build_gt_encoding(x, gts, groups, binary)
    E_emp = to_empirical(E_onehot, train['concept_means'].float())

    ucp = group_scores_ucp(x, groups, binary)
    gpstd = group_scores_gpstd(stds, groups)

    # frac=0 sanity check against stored metrics
    base_acc, base_f1 = eval_preds(predict(model, x, device), test_y)
    stored = None
    mpath = os.path.join(args.load_dir, "metrics.txt")
    if os.path.exists(mpath):
        stored = json.load(open(mpath)).get('label_f1')
        if stored is not None and abs(stored - base_f1) > 1e-6:
            logger.warning(f"frac=0 label_f1 {base_f1:.6f} != stored {stored:.6f}")
        else:
            logger.info(f"frac=0 sanity OK: label_f1={base_f1:.6f} matches stored")

    results = []
    for enc_name, E in [('onehot', E_onehot), ('empirical', E_emp)]:
        for frac in FRACTIONS:
            n_int = int(round(frac * G))
            entry = {'encoding': enc_name, 'frac': frac, 'n_groups_intervened': n_int}
            # random ordering: mean +- std over reps
            accs, f1s = [], []
            for rep in range(N_RANDOM_REPS):
                gen = torch.Generator().manual_seed(1000 + rep)
                rnd = torch.rand(x.shape[0], G, generator=gen)
                preds = predict(model, intervene(x, E, rnd, n_int, groups), device)
                a, f = eval_preds(preds, test_y)
                accs.append(a); f1s.append(f)
            entry['random'] = {'label_acc': float(np.mean(accs)), 'label_acc_std': float(np.std(accs)),
                               'label_f1': float(np.mean(f1s)), 'label_f1_std': float(np.std(f1s))}
            # deterministic orderings
            for oname, S in [('ucp', ucp), ('gpstd', gpstd)]:
                preds = predict(model, intervene(x, E, S, n_int, groups), device)
                a, f = eval_preds(preds, test_y)
                entry[oname] = {'label_acc': a, 'label_f1': f}
            logger.info(entry)
            results.append(entry)

    out = {
        'run': os.path.basename(os.path.normpath(args.load_dir)),
        'dataset': ds_name,
        'n_groups': G,
        'baseline_label_acc': base_acc,
        'baseline_label_f1': base_f1,
        'stored_label_f1': stored,
        'n_random_reps': N_RANDOM_REPS,
        'results': results,
    }
    with open(os.path.join(args.load_dir, "interventions.json"), "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved {os.path.join(args.load_dir, 'interventions.json')}")
