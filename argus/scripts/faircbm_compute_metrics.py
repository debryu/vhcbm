import sys, os, json, time
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
RESULTS = os.path.join(FAST, "results/FAIRCBM")
N = 340

VLM_TO_TEXT = {'ViT-L/14': 'clip', 'dinov3': 'dinov3'}
TEXT_TO_VLM = {v: k for k, v in VLM_TO_TEXT.items()}


def load_embeddings(ds_name, split, model_name):
    sanitized = model_name.replace('/', '%')
    emb = torch.load(os.path.join(ACT, f'{ds_name}_{split}_{sanitized}.pth'), map_location='cpu')
    emb = (emb - torch.mean(emb, dim=0, keepdim=True)) / torch.std(emb, dim=0, keepdim=True)
    if model_name in ['dinov3']:
        emb = F.normalize(emb, p=2, dim=-1)
    return emb


def collect_concepts(dataset):
    concepts = []
    for i in range(len(dataset)):
        _, c, _ = dataset[i]
        concepts.append(c)
    return torch.stack(concepts, dim=0)


def compute_ecer(test_probs, concept_gts, tag):
    kuipers = []
    # NOTE: cluster-specific path, replace with your own location
    tmp_plot = f"/leonardo_scratch/fast/IscrC_ARGO/results/FAIRCBM_metrics_work/_ecce_tmp_{tag}.pdf"
    for i in range(test_probs.shape[1]):
        probs_i = test_probs[:, i].detach().clone().cpu()
        labels_i = concept_gts[:, i].clone().cpu()
        kuiper, ks, lenscale = _cumulative(probs=probs_i, labels=labels_i, majorticks=5, minorticks=20, filename=tmp_plot)
        kuipers.append(kuiper)
    if os.path.exists(tmp_plot):
        os.remove(tmp_plot)
    return float(np.mean(kuipers))


def DCI(reprs, train_test_ratio=0.7):
    n = len(reprs['concept_means'])
    indices = torch.randperm(n)
    train_size = int(n * train_test_ratio)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    representation_train = reprs['concept_means'][train_idx]
    representation_val = reprs['concept_means'][val_idx]
    concept_gt_train = reprs['concept_gts'][train_idx]
    concept_gt_val = reprs['concept_gts'][val_idx]
    dci = DCI_wrapper(representation_train, concept_gt_train, representation_val, concept_gt_val, 'WARNING')
    return dci['disentanglement']


def run_one(ds_name, backbone_text, seed, seed_offset=0):
    torch.manual_seed(seed + seed_offset)
    model_name = TEXT_TO_VLM[backbone_text]
    run_dir = os.path.join(RESULTS, backbone_text, f"{ds_name}-{N}-random-SEED={seed}")
    probe_path = os.path.join(run_dir, "probe.pth")
    trained_model = load_trainer(probe_path)

    test_data = GenericDataset(ds_name=ds_name, split='test')
    test_concepts = collect_concepts(test_data)
    test_embeddings = load_embeddings(ds_name, 'test', model_name)

    with torch.no_grad():
        test_probs, _ = trained_model.eval(test_embeddings, test_concepts)
    test_probs = test_probs.detach().cpu()

    tag = f"{ds_name}_{backbone_text}_{seed}"
    ecer = compute_ecer(test_probs, test_concepts, tag)

    reprs = {'concept_means': test_probs, 'concept_gts': test_concepts.cpu()}
    disent = DCI(reprs, train_test_ratio=0.7)

    with open(os.path.join(run_dir, "metrics.txt")) as f:
        base_metrics = json.load(f)

    result = {
        **base_metrics,
        'dataset': ds_name,
        'backbone': backbone_text,
        'seed': seed,
        'N': N,
        'ecer': ecer,
        'disentanglement': float(disent),
    }
    return result


if __name__ == "__main__":
    ds_name = sys.argv[1]
    backbone_text = sys.argv[2]
    seed = int(sys.argv[3])
    t0 = time.time()
    res = run_one(ds_name, backbone_text, seed)
    res['elapsed'] = time.time() - t0
    print(json.dumps(res, indent=2))
