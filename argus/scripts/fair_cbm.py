"""
E1 rebuttal experiment: fair frozen-backbone CBM baseline.

Frozen VLM embeddings -> concept head (MultiTaskTrainer, trained on N fully-annotated
samples, N in the same budget grid as scripts/lp.py) -> sequential linear task layer
(trained on the full train set's *predicted* concepts, mirroring the VH-CBM task layer).

The probe part replicates scripts/lp.py call-for-call (set_seed -> random.sample ->
label collection -> trainer) so concept metrics match the published LP-@ results.

Env: DS_NAME, VLM (ViT-L/14 | dinov3 | resnet18_cub), SEED, FAST, SCRATCH, DEVICE.
Output: $FAST/results/FAIRCBM/{clip|dinov3|resnet18_cub}/{ds}-{N}-random-SEED={seed}/metrics.txt
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
activations_folder = os.path.join(FAST_STORAGE, "activations")

VLM_TO_TEXT = {'ViT-L/14': 'clip', 'dinov3': 'dinov3', 'resnet18_cub': 'resnet18_cub'}
SAVE_ROOT = os.path.join(FAST_STORAGE, "results/FAIRCBM", VLM_TO_TEXT[MODEL_NAME])

STEPS = [40, 100, 160, 220, 280, 340, 400]   # same budgets as scripts/lp.py:166

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


def to_task_input(probs):
    """Mirror the VH-CBM concept_means layout (cf. eval_argocbm.py:151-165):
    binary concepts -> [1-p, p] column pairs; multiclass groups -> softmax block as-is."""
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
    logger.info(f"FairCBM: {ds_name} / {MODEL_NAME} / seed {SEED}")
    train_embeddings = load_embeddings('train')
    val_embeddings = load_embeddings('val')
    test_embeddings = load_embeddings('test')

    train_data = GenericDataset(ds_name=ds_name, split='train')
    val_data = GenericDataset(ds_name=ds_name, split='val')
    test_data = GenericDataset(ds_name=ds_name, split='test')

    # Everything the task layer needs is collected BEFORE the budget loop so the
    # RNG trajectory inside the loop stays identical to scripts/lp.py (smoke test S3).
    logger.info("Collecting labels (train/val/test)")
    train_concepts_full, train_y = collect_labels(train_data)
    val_concepts, val_y = collect_labels(val_data)
    _, test_y = collect_labels(test_data)

    for subset_size in STEPS:
        logger.info(f"=== budget N={subset_size} ===")
        # --- probe: identical call order to lp.py:168-208 ---
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

        trained_model = MultiTaskTrainer(train_embeddings[random_subset].shape[1], concept_structure)
        pat = 0
        min_loss = 1000
        for e in range(10000):
            train_loss = trained_model.train_epoch(train_embeddings[random_subset], train_labels)
            if train_loss < min_loss:
                min_loss = train_loss
            else:
                pat += 1
            if pat > 3 or train_loss < 0.01:
                break
            if e % 100 == 0:
                logger.debug(f"{train_loss} patience {pat}")

        test_probs, _ = trained_model.eval(test_embeddings, test_labels)
        test_probs = test_probs.detach().cpu()
        concept_metrics = trained_model.evaluate(test_probs, test_labels)
        ece = ece_from_probs(test_probs, test_labels.cpu())

        # --- sequential task layer on predicted concepts (full train set) ---
        with torch.no_grad():
            train_probs, _ = trained_model.eval(train_embeddings, train_concepts_full)
            val_probs, _ = trained_model.eval(val_embeddings, val_concepts)
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
            'training_pool_size': subset_size,
            'concept_used': subset_size * len(concept_structure),
            'f1_raw': float(concept_metrics['f1_raw']),
            'f1_cal': float(concept_metrics['f1_cal']),
            'roc_auc': float(concept_metrics['rocauc']),
            'pr_auc': float(concept_metrics['prauc']),
            'ece': float(ece['ece']),
            'label_f1': float(cr_labels['macro avg']['f1-score']),
            'label_acc': float(cr_labels['accuracy']),
        }
        logger.info(res)

        save_dir = os.path.join(SAVE_ROOT, f"{ds_name}-{subset_size}-random-SEED={SEED}")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            json.dump(res, f, indent=2)
        torch.save(task_model.weight.data.detach().cpu(), os.path.join(save_dir, "W_g.pt"))
        torch.save(task_model.bias.data.detach().cpu(), os.path.join(save_dir, "b_g.pt"))
        trained_model.save_trainer(os.path.join(save_dir, "probe.pth"))

    logger.info("### Done ###")
