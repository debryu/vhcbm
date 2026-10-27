"""D6 figure + table: query-efficiency of intervention policies (ICBM CooP vs GP uncertainty).

Reads rebuttal_drafts/D6_coop.jsonl (written by eval_interventions_coop.py) and produces
  rebuttal_drafts/D6_intervention_curves.pdf / .png   two panels, CUB and CelebA
  rebuttal_drafts/D6_coop_table.md                    the numbers behind the figure

Encoding (per the dataviz method):
  colour  = POLICY   (categorical slots 1-3 of the reference palette, fixed order, all-pairs
                      validated in both modes: CVD dE 9.2 / normal-vision 24.0)
  line    = MODEL    (VH-CBM solid, CBM-fair dashed)
so series identity never rests on colour alone. One y-axis per panel; the two panels are small
multiples of the same measure, not a dual axis.
"""
import json, os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# NOTE: cluster-specific path, replace with your own location
HERE = os.environ.get('D6_OUTDIR', '/leonardo_scratch/fast/IscrC_ARGO/argobackup/argus/rebuttal_drafts')
SRC = os.environ.get('D6_SRC', os.path.join(HERE, 'D6_coop.jsonl'))

# reference palette, categorical slots 1-3 (light mode)
C = {'gpstd': '#2a78d6', 'coop': '#eb6834', 'random': '#1baf7a'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
STYLE = {'vh': ('-', 2.0), 'fair': ((0, (5, 2)), 1.8)}
LABEL = {('vh', 'gpstd'): 'VH-CBM · GP uncertainty (ours)',
         ('vh', 'coop'): 'VH-CBM · CooP (ICBM policy)',
         ('vh', 'random'): 'VH-CBM · random',
         ('fair', 'coop'): 'CBM-fair · CooP (ICBM policy)',
         ('fair', 'random'): 'CBM-fair · random'}
SERIES = [('vh', 'gpstd'), ('vh', 'coop'), ('vh', 'random'), ('fair', 'coop'), ('fair', 'random')]
PANELS = ['cub', 'celeba']
TITLE = {'cub': 'CUB  (112 concept groups)', 'celeba': 'CelebA  (39 concept groups)'}


def load():
    rows = [json.loads(l) for l in open(SRC)]
    agg = defaultdict(list)
    for r in rows:
        agg[(r['dataset'], r['model'])].append(r)
    return rows, agg


def series(agg, ds, model, policy):
    """mean and std across seeds; None if that arm was not run."""
    rs = agg.get((ds, model), [])
    rs = [r for r in rs if policy in r['curves']]
    if not rs:
        return None
    k = rs[0]['kgrid']
    M = np.array([r['curves'][policy] for r in rs])
    return np.array(k), M.mean(0), M.std(0), len(rs)


def queries_to(k, y, target):
    """Smallest budget whose accuracy reaches `target` (linear interp between grid points).
    0 if the curve already starts at or above the target; None if it never reaches it."""
    if y[0] >= target:
        return 0.0
    for i in range(1, len(k)):
        if y[i] >= target:
            if y[i] == y[i - 1]:
                return float(k[i])
            f = (target - y[i - 1]) / (y[i] - y[i - 1])
            return float(k[i - 1] + f * (k[i] - k[i - 1]))
    return None


def main():
    rows, agg = load()
    panels = [d for d in PANELS if any(k[0] == d for k in agg)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 3.9))
    if len(panels) == 1:
        axes = [axes]

    table = ['# D6 — intervention curves under the ICBM CooP policy', '',
             'Task accuracy vs number of concept groups revealed per test image (mean over 3 seeds).',
             'CooP = Chauhan et al. (AAAI 2023) Algorithm 2, alpha*CPU + beta*CIS, cost term inert',
             '(uniform concept cost), alpha/beta tuned on validation. Empirical-percentile encoding',
             'throughout, so the policies differ and nothing else does.', '']

    for ax, ds in zip(axes, panels):
        best_fair, curves = -1, {}
        for model, pol in SERIES:
            s = series(agg, ds, model, pol)
            if s is None:
                continue
            k, m, sd, n = s
            curves[(model, pol)] = (k, m, sd, n)
            ls, lw = STYLE[model]
            ax.plot(k, m, color=C[pol], linestyle=ls, linewidth=lw, marker='o', markersize=3.4,
                    markeredgecolor='white', markeredgewidth=0.6, label=LABEL[(model, pol)],
                    zorder=3 if model == 'vh' else 2)
            ax.fill_between(k, m - sd, m + sd, color=C[pol], alpha=0.10, linewidth=0, zorder=1)
            if model == 'fair':
                best_fair = max(best_fair, m.max())

        # the query-efficiency headline: where does VH reach the baseline's BEST accuracy?
        if best_fair > 0 and ('vh', 'gpstd') in curves:
            k, m, _, _ = curves[('vh', 'gpstd')]
            ax.axhline(best_fair, color=INK2, linewidth=0.9, linestyle=':', zorder=1)
            q = queries_to(k, m, best_fair)
            if q is not None:
                lo, hi = ax.get_ylim()
                rng = hi - lo
                # a query count must round UP: 1.33 queries means you need 2
                nq = int(np.ceil(q))
                txt = ("VH-CBM starts above\nthe baseline's best" if q == 0
                       else f'{nq} quer{"y" if nq == 1 else "ies"} to exceed\nthe baseline\'s best')
                # keep the callout inside the axes: it collided with the x-label otherwise
                ytext = max(best_fair - 0.16 * rng, lo + 0.06 * rng)
                ax.annotate(txt, xy=(q, best_fair), xytext=(q + 0.09 * k[-1], ytext),
                            color=INK, fontsize=8.2, linespacing=1.35, va='bottom',
                            arrowprops=dict(arrowstyle='-', color=INK2, linewidth=0.8))
                ax.plot([q], [best_fair], marker='o', ms=5, color=INK, zorder=4)

        ax.set_title(TITLE.get(ds, ds), fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel('concept groups revealed per image', fontsize=9.5, color=INK2)
        ax.set_ylabel('task accuracy', fontsize=9.5, color=INK2)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5, length=3)

        # ---- table ----
        table.append(f'## {ds}')
        table.append('')
        hdr = curves[list(curves)[0]][0]
        table.append('| policy | ' + ' | '.join(f'k={int(v)}' for v in hdr) + ' |')
        table.append('|' + '---|' * (len(hdr) + 1))
        for key in SERIES:
            if key not in curves:
                continue
            k, m, sd, n = curves[key]
            table.append(f'| {LABEL[key]} | ' + ' | '.join(f'{v:.3f}' for v in m) + ' |')
        # the two CooP components, reported for completeness and because `ucp` is the corrected
        # version of a curve already published in D1 (the float32 NaN bug)
        for model in ('vh', 'fair'):
            for pol, nm in (('ucp', 'CPU / predictive entropy'), ('cis', 'CIS / concept importance'),
                            ('coop_oneshot', 'CooP, NON-adaptive one-shot (what our earlier bug measured)')):
                s = series(agg, ds, model, pol)
                if s is None:
                    continue
                table.append(f'| {"VH-CBM" if model == "vh" else "CBM-fair"} · {nm} | '
                             + ' | '.join(f'{v:.3f}' for v in s[1]) + ' |')
        table.append('')
        tuned = sorted({(r['model'], r['coop_alpha']) for r in rows if r['dataset'] == ds})
        table.append('Tuned CooP weights (alpha on CPU, beta = 1 - alpha on CIS), per run: '
                     + ', '.join(f'{m}={al}' for m, al in tuned) + '.')
        table.append('')
        table.append(f'Max across-seed std: {max(curves[key][2].max() for key in curves):.3f}. '
                     f'Baseline best accuracy at any budget: {best_fair:.3f}.')
        for key in curves:
            k, m, _, _ = curves[key]
            q = queries_to(k, m, best_fair)
            table.append(f'- {LABEL[key]}: reaches the baseline ceiling at '
                         + (f'**{int(np.ceil(q))}** queries' if q is not None else 'no budget')
                         + f'; peak {m.max():.3f} at k={int(k[int(np.argmax(m))])}.')
        table.append('')

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=8.8,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Concept interventions: which query policy buys accuracy fastest',
                 fontsize=11.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'D6_intervention_curves.{ext}'), dpi=200,
                    bbox_inches='tight', facecolor='white')
    open(os.path.join(HERE, 'D6_coop_table.md'), 'w').write('\n'.join(table) + '\n')
    print('\n'.join(table))
    print('\nwrote D6_intervention_curves.pdf/.png and D6_coop_table.md')


if __name__ == '__main__':
    main()
