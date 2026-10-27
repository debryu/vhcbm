"""Test-time intervention curves under the ICBM (Chauhan et al., AAAI 2023) CooP policy (Table 9).

ICBM's contribution is a *policy* for choosing which concept to query next, and the GP of VH-CBM
provides predictive uncertainty, which is such a policy. We run their policy and ours on the same
models and compare query-efficiency.

CooP (their Algorithm 2):      score_i = a*CPU(c_i) + b*CIS(c_i) - g*q_i
  CPU  concept prediction uncertainty = entropy of p(c_i | x)          [= our existing `ucp`]
  CIS  concept importance             = | E_v~p(c_i|x)[ p(y=k | c_i=v, c_S, ch) ] - p(y=k | c_S, ch) |
                                        k = the label predicted in the PREVIOUS round; the score
                                        conditions on c_S, the already-revealed concepts.
                                        NB the |.| is OUTSIDE the expectation.
  q_i  acquisition cost
Each term is min-max normalised on the policy-learning split; a, b are tuned on a holdout.
Their Algorithm 1 applies this GREEDILY: acquire one concept, reveal it, re-score, repeat.

Adaptations, stated because they matter for what may be claimed:
  (i)   ALL concepts cost the same in our datasets, so q_i is constant and -g*q_i cannot change a
        per-sample ranking. The cost term is therefore inert here, not omitted. ICBM's cost-aware
        results have no counterpart in our setting.
  (ii)  We tune (a, b) on the VALIDATION split by maximising the area under the validation
        intervention curve, then freeze them and report test curves. ICBM tunes on a holdout too.
  (iii) Intervention uses the empirical-percentile encoding (Shin et al. Sec 4.4), identical to the
        one already used for every other ordering, so the policies differ and nothing else does.
  (iv)  Groups, not raw columns, are queried (a human corrects a concept, not a logit). For CUB and
        CelebA a group IS a single binary concept, so this matches ICBM's per-concept querying
        exactly; it would differ only on the multiclass-factor datasets, which are not run here.
  (v)   CooP and CIS are run GREEDILY with re-scoring after every acquisition, per their Algorithm 1.
        GP std, concept entropy and random are static functions of the image, so for those repeated
        greedy top-1 is provably the same set as one-shot top-k -- they are not disadvantaged.
This is ICBM's *policy* evaluated on our models. It is not a retrained ICBM; their model-side
contributions (the intervention-aware training of the underlying CBM) are not reproduced.

Usage: python eval_interventions_coop.py -dataset cub -backbone clip -seed 132 -model {vh,fair}
Output: one JSON row -> rebuttal_drafts/D6_coop.jsonl
"""
import sys, os, json, argparse, time

# NOTE: cluster-specific path, replace with your own location
sys.path.insert(0, '/leonardo_scratch/fast/IscrC_ARGO/argobackup')
sys.path.insert(0, '/leonardo/home/userexternal/ndebole0/projects/CQA')

import torch
import numpy as np
from loguru import logger
from CQA.datasets import GenericDataset

from eval_interventions import (CONCEPT_STRUCTURES, group_columns, build_gt_encoding,
                                to_empirical, group_scores_ucp, group_scores_gpstd,
                                intervene, eval_preds)

FAST = os.environ['FAST']
# NOTE: cluster-specific path, replace with your own location
OUT = '/leonardo_scratch/fast/IscrC_ARGO/argobackup/argus/rebuttal_drafts/D6_coop.jsonl'
IDX = json.load(open('/leonardo_scratch/fast/IscrC_ARGO/argobackup/argus/rebuttal_drafts/_gp_ckpt_index.json'))
BB_ACT = {'clip': 'ViT-L/14', 'dinov3': 'dinov3'}
N_RANDOM_REPS = 5
N_FAIR = 340

# budget grids: fine near 0 where the query-efficiency argument lives
KGRID = {'cub':    [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 28, 32, 48, 56, 64, 84, 112],
         'celeba': [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 29, 32, 36, 39],
         'shapes3d': [0, 1, 2, 3, 4, 5]}   # G=5 -> full grid
ALPHAS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]   # b = 1 - a; only the ratio matters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-dataset', required=True, choices=list(KGRID))
    p.add_argument('-backbone', default='clip', choices=list(BB_ACT))
    p.add_argument('-seed', type=int, required=True)
    p.add_argument('-model', required=True, choices=['vh', 'fair'])
    p.add_argument('-load_dir', default=None, help='VH-CBM run dir (required for -model vh)')
    p.add_argument('-logger', default='WARNING')
    a = p.parse_args()
    logger.remove(); logger.add(sys.stderr, level=a.logger)
    assert a.model == 'fair' or a.load_dir, '-model vh requires -load_dir'
    return a


def auc_norm(ys, ks):
    """Trapezoidal area under the curve, normalised by the budget range."""
    return float(sum((ys[i] + ys[i + 1]) / 2 * (ks[i + 1] - ks[i])
                     for i in range(len(ks) - 1)) / (ks[-1] - ks[0]))


def minmax(v, lo, hi):
    """ICBM normalises each term to [0,1] on the policy-learning split so the two are commensurate.
    Done in float64: in float32 the affine map merges distinct CIS values into ties, and topk then
    breaks those ties by column index, which silently changes the ranking (observed: CooP with a
    tuned alpha=0 did NOT reproduce the pure-CIS curve)."""
    v = v.double()
    return (v - lo.double()) / (hi.double() - lo.double()).clamp_min(1e-12)


def group_scores_cis(x_cur, x0, W, b, groups, binary, p5, p95, logits=None):
    """Concept Importance Score, ICBM Algorithm 2:

        CIS(c_i; S, k) = | E_{v ~ p_th(c_i|x)}[ p_ph(y=k | c_i=v, c_S, ch_Sbar\\{i}) ]
                          - p_ph(y=k | c_S, ch_Sbar) |

    Note the absolute value is OUTSIDE the expectation -- E|dp| (what an earlier version of this
    computed) is a different, larger quantity that cannot cancel across values.

    `x_cur` is the CURRENT concept vector (already-revealed concepts hold their GT encoding), which
    is what the score conditions on; `x0` is the ORIGINAL prediction, which is what p_th(c_i|x)
    comes from. `k` is the label predicted from x_cur, i.e. "the previous round". Exact for a linear
    head via a rank-1 logit delta, so no re-forward is needed.
    """
    if logits is None:
        logits = x_cur @ W.T + b
    k = logits.argmax(1)
    pk = torch.softmax(logits, 1).gather(1, k[:, None])[:, 0]
    S = torch.zeros(x_cur.shape[0], len(groups), dtype=torch.float64, device=x_cur.device)

    for gid, (cols, _) in enumerate(groups):
        Wc = W[:, cols]
        xc = x_cur[:, cols]
        n_vals = 2 if binary else len(cols)
        if binary:                                  # p(c_g = v | x), from the ORIGINAL prediction
            p1 = x0[:, cols[1]].clamp(0, 1)
            pv = torch.stack([1 - p1, p1], 1)
        else:
            pv = x0[:, cols].clamp_min(0)
            pv = pv / pv.sum(1, keepdim=True).clamp_min(1e-12)
        exp_p = torch.zeros_like(pk)
        for v in range(n_vals):
            # value v of this group, in the same encoding an actual intervention would write
            repl = torch.tensor([p95[c] if j == v else p5[c] for j, c in enumerate(cols)],
                                device=x_cur.device, dtype=x_cur.dtype)
            delta = (repl - xc) @ Wc.T
            p_new = torch.softmax(logits + delta, 1).gather(1, k[:, None])[:, 0]
            exp_p = exp_p + pv[:, v] * p_new
        S[:, gid] = (exp_p - pk).abs().double()
    return S


def curve(x, E, scores, groups, predict, y, kgrid):
    """One-shot top-k by a STATIC score. For a score that does not depend on what has already been
    revealed (GP std, concept entropy, random), repeated greedy top-1 selects exactly this set, so
    this is not an approximation for those policies -- only for CIS/CooP, which is why those get
    `curve_greedy` instead."""
    return [float(eval_preds(predict(intervene(x, E, scores, k, groups)), y)[0]) for k in kgrid]


def curve_greedy(x, E, W, b, groups, binary, p5, p95, cpu_n, lo_c, hi_c, alpha, y, kgrid,
                 return_masks=False):
    """CooP as ICBM actually specifies it (their Algorithm 1): acquire ONE concept, reveal its
    ground truth, RECOMPUTE the scores on the updated concept vector, repeat.

    This matters because CIS conditions on the already-revealed set c_S and on k, the label
    predicted in the previous round -- both of which change after every acquisition. Scoring once
    up front and taking top-k is a strictly weaker, non-adaptive version of their policy, and it
    understates it. Each sample follows its own trajectory.
    """
    N, G = x.shape[0], len(groups)
    colmat = torch.zeros(G, x.shape[1], dtype=torch.bool, device=x.device)
    for gid, (cols, _) in enumerate(groups):
        colmat[gid, cols] = True

    x_cur = x.clone()
    revealed = torch.zeros(N, G, dtype=torch.bool, device=x.device)
    idx = torch.arange(N, device=x.device)
    out, kset, masks = [], set(kgrid), {}

    for step in range(max(kgrid) + 1):
        logits = x_cur @ W.T + b
        if step in kset:
            out.append(float(eval_preds(logits.argmax(1), y)[0]))
            if return_masks:
                masks[step] = revealed.clone()
        if step == max(kgrid):
            break
        cis = group_scores_cis(x_cur, x, W, b, groups, binary, p5, p95, logits=logits)
        score = alpha * cpu_n + (1 - alpha) * minmax(cis, lo_c, hi_c)
        score = score.masked_fill(revealed, float('-inf'))
        pick = score.argmax(1)                                  # one concept, per sample
        revealed[idx, pick] = True
        x_cur = torch.where(colmat[pick], E, x_cur)
    return (out, masks) if return_masks else out


if __name__ == '__main__':
    a = parse_args()
    ds, bb, seed = a.dataset, a.backbone, a.seed
    cstruct = CONCEPT_STRUCTURES[ds]
    G = len(cstruct)
    kgrid = KGRID[ds]
    t0 = time.time()

    # ---------- load the model and produce x / E / stds for test AND val ----------
    if a.model == 'vh':
        # same loading path as eval_interventions.py: the PUBLISHED frozen task head
        from argus.models.argocbm import ArgoCBM
        from CQA.utils.args_utils import load_args
        args_model = load_args(argparse.Namespace(load_dir=a.load_dir, logger=a.logger))
        model = ArgoCBM(args_model)
        model.load()
        assert args_model.dataset == ds, f"{args_model.dataset} != {ds}"
        P = {s: model.backbone.get_concept_preds(s) for s in ('train', 'val', 'test')}
        X = {s: P[s]['concept_means'].float() for s in P}
        GT = {s: P[s]['concept_gts'] for s in P}
        STD = {s: P[s]['concept_stds'].float() for s in ('val', 'test')}
        W = model.final_layer.weight.detach().cpu().float()
        b = model.final_layer.bias.detach().cpu().float()
    else:
        from argus.models.linear import load_trainer
        from eval_interventions_baseline import load_embeddings, collect, to_task_input
        rd = os.path.join(FAST, 'results/FAIRCBM', bb, f'{ds}-{N_FAIR}-random-SEED={seed}')
        assert os.path.isdir(rd), f'missing {rd}'
        probe = load_trainer(os.path.join(rd, 'probe.pth'))
        W = torch.load(os.path.join(rd, 'W_g.pt'), map_location='cpu').float()
        b = torch.load(os.path.join(rd, 'b_g.pt'), map_location='cpu').float()
        X, GT = {}, {}
        for s in ('train', 'val', 'test'):
            emb = load_embeddings(ds, s, bb)
            c, _ = collect(ds, s)
            with torch.no_grad():
                probs, _ = probe.eval(emb, c)
            X[s] = to_task_input(probs.detach().cpu().float(), cstruct)
            GT[s] = c
        STD = None

    groups, binary = group_columns(ds, cstruct, X['test'].shape[1])
    p5 = torch.quantile(X['train'], 0.05, dim=0)
    p95 = torch.quantile(X['train'], 0.95, dim=0)

    Y = {}
    for s in ('val', 'test'):
        d = GenericDataset(ds_name=ds, split=s)
        Y[s] = torch.stack([d[i][2] for i in range(len(d))], 0).long()

    def predict(inp):
        with torch.no_grad():
            return (inp.float() @ W.T + b).argmax(1)

    # ---------- scores on both splits ----------
    Sc = {}
    for s in ('val', 'test'):
        E = to_empirical(build_gt_encoding(X[s], GT[s], groups, binary), X['train'])
        ucp = group_scores_ucp(X[s], groups, binary)
        cis = group_scores_cis(X[s], X[s], W, b, groups, binary, p5, p95)   # step-0 CIS
        Sc[s] = dict(E=E, ucp=ucp, cis=cis)
        if STD is not None:
            Sc[s]['gpstd'] = group_scores_gpstd(STD[s], groups)

    # min-max normalise both terms on the policy-learning (validation) split, per ICBM
    lo_u, hi_u = Sc['val']['ucp'].min(), Sc['val']['ucp'].max()
    lo_c, hi_c = Sc['val']['cis'].min(), Sc['val']['cis'].max()
    for s in Sc:
        Sc[s]['ucp_n'] = minmax(Sc[s]['ucp'], lo_u, hi_u)
        Sc[s]['cis_n'] = minmax(Sc[s]['cis'], lo_c, hi_c)

    # ---------- tune (alpha, beta) on VALIDATION: maximise area under the val curve ----------
    best, best_alpha = -1.0, None
    val_search = {}
    for al in ALPHAS:
        cv = curve_greedy(X['val'], Sc['val']['E'], W, b, groups, binary, p5, p95,
                          Sc['val']['ucp_n'], lo_c, hi_c, al, Y['val'], kgrid)
        auc = auc_norm(cv, kgrid)
        val_search[al] = {'auc': auc, 'curve': cv}
        if auc > best:
            best, best_alpha = auc, al
    logger.info(f"tuned alpha={best_alpha} (beta={1-best_alpha}) val AUC={best:.4f}")

    # ---------- test curves ----------
    Et, res = Sc['test']['E'], {}
    # CooP and CIS are ADAPTIVE -> greedy, re-scored after every acquisition (ICBM Algorithm 1).
    res['coop'] = curve_greedy(X['test'], Et, W, b, groups, binary, p5, p95,
                               Sc['test']['ucp_n'], lo_c, hi_c, best_alpha, Y['test'], kgrid)
    res['cis'] = curve_greedy(X['test'], Et, W, b, groups, binary, p5, p95,
                              Sc['test']['ucp_n'], lo_c, hi_c, 0.0, Y['test'], kgrid)
    # entropy / GP std / random are STATIC: greedy top-1 repeated == one-shot top-k, exactly.
    res['ucp'] = curve(X['test'], Et, Sc['test']['ucp_n'], groups, predict, Y['test'], kgrid)
    res['coop_oneshot'] = curve(  # kept to quantify what the non-adaptive version cost CooP
        X['test'], Et, best_alpha * Sc['test']['ucp_n'] + (1 - best_alpha) * Sc['test']['cis_n'],
        groups, predict, Y['test'], kgrid)
    if STD is not None:
        res['gpstd'] = curve(X['test'], Et, Sc['test']['gpstd'].double(), groups, predict,
                             Y['test'], kgrid)
    rnd = []
    for rep in range(N_RANDOM_REPS):
        gen = torch.Generator().manual_seed(1000 + rep)
        rnd.append(curve(X['test'], Et, torch.rand(X['test'].shape[0], G, generator=gen),
                         groups, predict, Y['test'], kgrid))
    res['random'] = list(np.mean(rnd, 0))
    res['random_std'] = list(np.std(rnd, 0))

    # Consistency check. At alpha=1 CooP is pure CPU, which is a STATIC score, so the greedy loop
    # must select the same set as one-shot top-k. It is not required to be bit-identical: entropy
    # has exact ties across concepts, and repeated argmax and topk break ties differently, which
    # can swap two equally-scored concepts. Report the deviation and only fail if it is large
    # enough to be something other than tie-breaking.
    # Consistency check on the greedy machinery. At alpha=1 CooP is pure CPU, a STATIC score, so
    # greedy must select a valid top-k set at every budget. It need NOT pick the same INDICES as
    # topk: after the entropy fix every concept at the clamp boundary shares the identical minimum
    # entropy, so there are large tie blocks and the two tie-break differently. The invariant that
    # does hold regardless of tie-breaking is the SELECTED SCORE MASS, so that is what we assert.
    sc_static = Sc['test']['ucp_n']
    gd, masks = curve_greedy(X['test'], Et, W, b, groups, binary, p5, p95,
                             sc_static, lo_c, hi_c, 1.0, Y['test'], kgrid, return_masks=True)
    consistency = 0.0
    for kk in kgrid:
        if kk == 0:
            continue
        g_mass = float((sc_static * masks[kk]).sum())
        t_mass = float(sc_static.topk(kk, dim=1).values.sum())
        consistency = max(consistency, abs(g_mass - t_mass) / max(abs(t_mass), 1e-9))
    curve_gap = float(np.abs(np.array(gd) - np.array(res['ucp'])).max())
    logger.info(f"greedy audit: max rel. score-mass error={consistency:.3g}; "
                f"curve gap vs one-shot={curve_gap:.4g} (ties)")
    assert consistency < 1e-9, (
        f"greedy does not select a valid top-k set (score-mass error {consistency:.3g})")

    row = dict(dataset=ds, backbone=bb, seed=seed, model=a.model, G=G, kgrid=kgrid,
               coop_alpha=best_alpha, coop_beta=1 - best_alpha, val_auc=best,
               val_search={str(k): v['auc'] for k, v in val_search.items()},
               greedy_scoremass_err=consistency, greedy_vs_oneshot_ties=curve_gap,
               curves={k: [round(float(z), 5) for z in v] for k, v in res.items()},
               elapsed=round(time.time() - t0, 1))
    with open(OUT, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(f"{ds}/{bb}/{a.model}/s{seed}  alpha={best_alpha}  " +
          "  ".join(f"{k}@0={v[0]:.3f},max={max(v):.3f}" for k, v in res.items() if k != 'random_std'),
          flush=True)
