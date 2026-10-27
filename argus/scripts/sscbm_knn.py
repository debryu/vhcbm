"""
E5 rebuttal experiment: SSCBM-faithful semi-supervised CBM over frozen embeddings.

Implements the pseudo-labeling of SSCBM (Hu et al., ICCV'25, Sec. 4.2) — which is also
exactly their own "CBM+SSL" baseline (Table 2): each unlabeled sample receives a SOFT
concept label equal to the weighted average of the ground-truth concept labels of its
k nearest labeled samples in the visual encoder's cosine space, with weights the
normalized reciprocals of the cosine distances. The concept head is then trained on
GT (labeled subset) + soft KNN pseudo-labels (rest) with a soft-target weighted CE
(MultiTaskTrainer hardens labels, models/linear.py:63-79, so we subclass it).

Deliberate adaptations, to be stated in the rebuttal:
- SSCBM's alignment loss needs spatial HxW feature maps vs concept embeddings; our
  pipeline operates on pooled frozen embeddings, so it does not apply. The paper itself
  reports KNN-only pseudo-labeling "has satisfactory performance" and uses it for its
  CBM+SSL/CEM+SSL baselines.
- Training is sequential (concept head -> task layer) as for every method we compare,
  instead of SSCBM's joint CEM training — controlled comparison at equal architecture.
- Their setting assumes task labels y are known for ALL samples; our sequential task
  layer is likewise trained on the full labeled train set.

The labeled subset draw is call-for-call identical to scripts/fair_cbm.py / scripts/lp.py
(set_seed -> random.sample), so SSCBM@N and FairCBM@N use the same annotations.

Env: DS_NAME, VLM (ViT-L/14 | dinov3), SEED, KNN_K (default 5), FAST, SCRATCH, DEVICE, SMOKE.
Output: $FAST/results/SSCBM/{clip|dinov3}/{ds}-{N}-knn-SEED={seed}/metrics.txt
"""
import torch
import torch.nn.functional as F
from argus.models.linear import MultiTaskTrainer
from argus.metrics.ece import ece_from_probs
from CQA.datasets import GenericDataset
import json, os
from loguru import logger
from sklearn.metrics import classification_report
import numpy as np
import random
import copy
from argus.utils.utils import set_seed
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DEVICE = str(os.environ.get("DEVICE", 'cuda'))
SEED = int(os.environ.get("SEED", 132))
FAST_STORAGE = os.environ["FAST"]
SCRATCH_STORAGE = os.environ["SCRATCH"]
MODEL_NAME = str(os.environ.get("VLM", 'ViT-L/14'))
ds_name = os.environ.get("DS_NAME", "shapes3d")
KNN_K = int(os.environ.get("KNN_K", 5))
SMOKE = os.environ.get("SMOKE", "0") == "1"
activations_folder = os.path.join(FAST_STORAGE, "activations")

VLM_TO_TEXT = {'ViT-L/14': 'clip', 'dinov3': 'dinov3', 'resnet18_cub': 'resnet18_cub'}
SAVE_ROOT = os.path.join(FAST_STORAGE, "results/SSCBM", VLM_TO_TEXT[MODEL_NAME])

STEPS = [40] if SMOKE else [40, 100, 160, 220, 280, 340, 400]   # same budgets as scripts/lp.py:166

if ds_name == 'celeba':
    n_concepts = 39
    concept_structure = {f'{i+1}': (i, i) for i in range(39)}
elif ds_name == 'dermamnist':
    n_concepts = 7
    concept_structure = {'finding': (0, 6)}
elif ds_name == 'cub':
    n_concepts = 112
    concept_structure = {f'{i}': (i, i) for i in range(112)}
elif ds_name == 'shapes3d':
    n_concepts = 42
    concept_structure = {
        'floor color': (0, 9),
        'wall color': (10, 19),
        'object color': (20, 29),
        'object size': (30, 37),
        'shape': (38, 41),
    }
else:
    raise ValueError(f"Unknown dataset {ds_name}")


def load_embeddings(split):
    # Same normalization as scripts/lp.py:132-155 (z-score; extra L2 for dinov3)
    sanitized = MODEL_NAME.replace('/', '%')
    emb = torch.load(os.path.join(activations_folder, f'{ds_name}_{split}_{sanitized}.pth'),
                     map_location=DEVICE)
    emb = (emb - torch.mean(emb, dim=0, keepdim=True)) / torch.std(emb, dim=0, keepdim=True)
    if MODEL_NAME in ['dinov3']:
        emb = F.normalize(emb, p=2, dim=-1)
    return emb


def collect_labels(dataset):
    concepts, ys = [], []
    for i in range(len(dataset)):
        _, c, l = dataset[i]
        concepts.append(c)
        ys.append(l)
    return torch.stack(concepts, dim=0), torch.stack(ys, dim=0).long()


def knn_soft_pseudo_labels(all_emb, labeled_idx, labeled_concepts, k):
    """SSCBM Sec 4.2: dist = 1 - cosine similarity to each labeled anchor; take the k
    nearest, weights = normalized reciprocal distances, pseudo-label = weighted average
    of the anchors' GT concept labels. Returns soft labels for ALL rows (labeled rows
    get overwritten with GT by the caller)."""
    k = min(k, len(labeled_idx))
    a = F.normalize(all_emb.float(), p=2, dim=-1)
    anchors = a[labeled_idx]                                   # (L, d)
    sim = a @ anchors.T                                        # (N, L) cosine similarity
    dist = (1.0 - sim).clamp_min(1e-8)
    topk = torch.topk(dist, k, dim=1, largest=False)
    w = 1.0 / topk.values                                      # (N, k) reciprocal distances
    w = w / w.sum(dim=1, keepdim=True)
    neigh = labeled_concepts.float()[topk.indices]             # (N, k, n_cols)
    return torch.einsum('nk,nkc->nc', w, neigh)


class SoftMultiTaskTrainer(MultiTaskTrainer):
    """Same heads/optimizer as MultiTaskTrainer, but trains on SOFT targets: per head,
    weighted soft cross-entropy matching the hard weighted-CE of train_epoch
    (models/linear.py:63-79) when targets are one-hot."""

    def _soft_targets(self, labels):
        targets = {}
        for task_name, (start, end) in self.concept_structure.items():
            if end - start + 1 == 1:
                p = labels[:, start].unsqueeze(-1)
                targets[task_name] = torch.cat([1 - p, p], dim=1)
            else:
                targets[task_name] = labels[:, start:end + 1]
        return targets

    def train_epoch_soft(self, embeddings, labels):
        self.model.train()
        self.optimizer.zero_grad()
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        logits = self.model(embeddings)
        targets = self._soft_targets(labels)
        total_loss = 0
        for task_name, t in targets.items():
            counts = t.sum(dim=0).clamp_min(1e-8)
            total = counts.sum()
            weights = total / (len(counts) * counts)           # same formula as linear.py:77
            logp = F.log_softmax(logits[task_name], dim=1)
            wt = weights.unsqueeze(0) * t
            loss = -(wt * logp).sum() / wt.sum()               # torch weighted-CE normalization
            total_loss += loss
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()


def train_probe_soft(embeddings, labels):
    # Same early-stopping pattern as scripts/lp.py:191-204
    trainer = SoftMultiTaskTrainer(embeddings.shape[1], concept_structure)
    pat = 0
    min_loss = 1000
    for e in range(10000):
        train_loss = trainer.train_epoch_soft(embeddings, labels)
        if train_loss < min_loss:
            min_loss = train_loss
        else:
            pat += 1
        if pat > 3 or train_loss < 0.01:
            break
        if e % 100 == 0:
            logger.debug(f"{train_loss} patience {pat}")
    return trainer


def to_task_input(probs):
    """Mirror the VH-CBM concept_means layout (cf. eval_argocbm.py:151-165)."""
    cols = []
    for task_name, (start, end) in concept_structure.items():
        if end - start + 1 == 1:
            p = probs[:, start].unsqueeze(-1)
            cols.append(torch.cat([1 - p, p], dim=1))
        else:
            cols.append(probs[:, start:end + 1])
    return torch.cat(cols, dim=1)


def train_task_layer(train_x, train_y, val_x, val_y):
    # Lifted from models/argocbm.py:188-266 (sequential CBM final layer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_x.shape[1]
    num_classes = int(train_y.max().item()) + 1
    model = torch.nn.Linear(input_dim, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=1024)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    for epoch in range(2000):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_x, v_y in val_loader:
                v_x, v_y = v_x.to(device), v_y.to(device)
                val_loss += criterion(model(v_x), v_y).item()
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= 10:
            logger.info(f"Task layer early stopping at epoch {epoch}")
            break
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


if __name__ == "__main__":
    logger.info(f"SSCBM-KNN: {ds_name} / {MODEL_NAME} / seed {SEED} / k={KNN_K}")
    train_embeddings = load_embeddings('train')
    val_embeddings = load_embeddings('val')
    test_embeddings = load_embeddings('test')

    train_data = GenericDataset(ds_name=ds_name, split='train')
    val_data = GenericDataset(ds_name=ds_name, split='val')
    test_data = GenericDataset(ds_name=ds_name, split='test')

    logger.info("Collecting labels (train/val/test)")
    train_concepts_full, train_y = collect_labels(train_data)
    val_concepts, val_y = collect_labels(val_data)
    test_concepts, test_y = collect_labels(test_data)

    for subset_size in STEPS:
        logger.info(f"=== budget N={subset_size} ===")
        # --- labeled subset: identical call order to lp.py:168-208 / fair_cbm.py ---
        set_seed(SEED)
        random_subset = random.sample(range(len(train_data)), int(subset_size))
        probing_train_data = torch.utils.data.Subset(train_data, random_subset)
        train_labels = []
        for i in probing_train_data:
            _, c, l = i
            train_labels.append(c)
        train_labels = torch.stack(train_labels, dim=0)
        test_labels = []
        for i in test_data:
            _, c, l = i
            test_labels.append(c)
        test_labels = torch.stack(test_labels, dim=0)

        # --- SSCBM Sec 4.2: KNN soft pseudo-labels from the labeled anchors ---
        soft = knn_soft_pseudo_labels(train_embeddings.cpu(), random_subset,
                                      train_labels.cpu(), KNN_K)
        soft[random_subset] = train_labels.float().cpu()       # GT wins on the labeled subset
        hard = torch.zeros_like(soft)
        for _, (s, e) in concept_structure.items():
            if e - s + 1 == 1:
                hard[:, s] = (soft[:, s] > 0.5).float()
            else:
                hard[torch.arange(soft.shape[0]), s + soft[:, s:e + 1].argmax(dim=1)] = 1.0
        pseudo_acc = (hard == train_concepts_full.float().cpu()).float().mean().item()
        logger.info(f"KNN pseudo-label (hardened) elementwise agreement with GT: {pseudo_acc:.4f}")

        # --- concept head trained on GT + soft pseudo-labels ---
        probe = train_probe_soft(train_embeddings, soft)

        test_probs, _ = probe.eval(test_embeddings, test_labels)
        test_probs = test_probs.detach().cpu()
        concept_metrics = probe.evaluate(test_probs, test_labels)
        ece = ece_from_probs(test_probs, test_labels.cpu())

        # --- sequential task layer on predicted concepts ---
        with torch.no_grad():
            train_probs, _ = probe.eval(train_embeddings, train_concepts_full)
            val_probs, _ = probe.eval(val_embeddings, val_concepts)
        train_x = to_task_input(train_probs.detach().cpu())
        val_x = to_task_input(val_probs.detach().cpu())
        test_x = to_task_input(test_probs)

        task_model = train_task_layer(train_x, train_y, val_x, val_y)
        task_model.eval()
        with torch.no_grad():
            preds = task_model(test_x.to(next(task_model.parameters()).device)).argmax(dim=1).cpu()
        cr_labels = classification_report(test_y.numpy(), preds.numpy(), output_dict=True)

        res = {
            'dataset': ds_name,
            'backbone': MODEL_NAME,
            'seed': SEED,
            'method': 'sscbm_knn',
            'knn_k': KNN_K,
            'training_pool_size': subset_size,
            'concept_used': subset_size * len(concept_structure),
            'pseudo_label_agreement': float(pseudo_acc),
            'f1_raw': float(concept_metrics['f1_raw']),
            'f1_cal': float(concept_metrics['f1_cal']),
            'roc_auc': float(concept_metrics['rocauc']),
            'pr_auc': float(concept_metrics['prauc']),
            'ece': float(ece['ece']),
            'label_f1': float(cr_labels['macro avg']['f1-score']),
            'label_acc': float(cr_labels['accuracy']),
        }
        logger.info(res)

        save_dir = os.path.join(SAVE_ROOT, f"{ds_name}-{subset_size}-knn-SEED={SEED}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            json.dump(res, f, indent=2)
        torch.save(task_model.weight.data.detach().cpu(), os.path.join(save_dir, "W_g.pt"))
        torch.save(task_model.bias.data.detach().cpu(), os.path.join(save_dir, "b_g.pt"))
        probe.save_trainer(os.path.join(save_dir, "probe.pth"))

    logger.info("### Done ###")
