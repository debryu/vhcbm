import torch
from argus.models.svm import SVM_Linear, SVM
from argus.models.linear import LinearProbe
from argus.models.mlp import MLPProbe
from CQA.datasets import GenericDataset
import json, os
from sklearn.metrics import precision_recall_curve, auc
from argus.metrics.auc import MacroAUC
from loguru import logger
from sklearn.metrics import classification_report
import numpy as np
import random
from argus.utils.utils import set_seed
import datetime
from tqdm import tqdm

time = datetime.datetime.now().strftime("%H_%M")
date = datetime.datetime.now().strftime("%Y_%m_%d")
time_date = f"{date}_{time}"
dict = {
    'macro auc'
    'training_pool_size'
}


def generate_name(date,ds_name,seed):
    return f"{date}-probe_{ds_name}__SEED={seed}.json"

SEED = int(os.environ.get("SEED", 42))
ds_name = os.environ.get("DS_NAME", "shapes3d")
set_seed(SEED)

KERNEL = os.environ.get("KERNEL", 'linear')
MODEL_NAME = str(os.environ.get("MODEL_NAME",'ViT-L/14'))
FAST_STORAGE = str(os.environ.get("FAST"))
DEVICE = str(os.environ.get("DEVICE",'cuda'))
SAVE_FOLDER = f"{FAST_STORAGE}/results/GPs/GPs"
activations_folder = os.path.join(FAST_STORAGE,"activations")
os.makedirs(os.path.join(activations_folder), exist_ok=True)

if ds_name == 'celeba':
    n_concepts = 39
elif ds_name == 'dermamnist':
    n_concepts = 7
elif ds_name == 'cub':
    n_concepts = 112
elif ds_name == 'shapes3d':
    n_concepts = 42
else:
    n_concepts = 0
    subset_size = 500
    
steps = [10,50,75,100,125,150,175,200,250,300,350,400,500]

def evaluate_clip_scores(n_concepts, clip_scores, dataset):
    aucs = []
    min_auc = []
    for c_id in range(n_concepts):
        concept_labels = []
        # Collect train labels
        for i in range(len(dataset)):
            concept_labels.append(dataset[i][1][c_id])
        concept_labels = torch.stack(concept_labels, dim=0).detach().cpu().numpy()
        # Final shape will be (1, train_size)
        
        predicted_concept = clip_scores[:,c_id].detach().cpu().numpy()
        
        res = MacroAUC(concept_labels,predicted_concept)
        aucs.append(res.macro_auc)
        min_auc.append(min(res.prauc0,res.prauc1))
        #logger.debug(f"Evaluating CLIP scores for concept {c_id}. PR-AUC={res.macro_auc}")
    #logger.info(np.mean(aucs))
    return {'auc':np.mean(aucs), 'minauc':np.mean(min_auc)}    
    
# Load Embeddings
MODEL_NAME_SANITIZED = MODEL_NAME.replace('/','%')

with torch.no_grad():
    train_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_train_{MODEL_NAME_SANITIZED}.pth'), map_location=DEVICE)
    # Normalization
    train_embeddings = (train_embeddings - torch.mean(train_embeddings, dim=0, keepdim=True))/torch.std(train_embeddings, dim=0, keepdim=True)
    
    test_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_test_{MODEL_NAME_SANITIZED}.pth'), map_location=DEVICE)
    # Normalization
    test_embeddings = (test_embeddings - torch.mean(test_embeddings, dim=0, keepdim=True))/torch.std(test_embeddings, dim=0, keepdim=True)
    
    val_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_val_{MODEL_NAME_SANITIZED}.pth'), map_location=DEVICE)
    # Normalization
    val_embeddings = (val_embeddings - torch.mean(val_embeddings, dim=0, keepdim=True))/torch.std(val_embeddings, dim=0, keepdim=True)
    
    text_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_concepts_{MODEL_NAME_SANITIZED}.pth'))
    # Normalization
    text_embeddings = (text_embeddings - torch.mean(text_embeddings, dim=0, keepdim=True))/torch.std(text_embeddings, dim=0, keepdim=True)
    
train_data = GenericDataset(ds_name=ds_name,split = 'train')
test_data = GenericDataset(ds_name=ds_name,split = 'test')
    
results = []
for i,subset_size in enumerate(tqdm(steps)):
    logger.debug(subset_size)
    random_subset = random.sample(range(len(train_data)), subset_size)
    probing_train_data = torch.utils.data.Subset(train_data, random_subset)
    train_subset_embeddings = train_embeddings[random_subset]
    # Skip SVMs because they do not handle subset in which there is only one class annotated
    '''
    # LINEAR_PROBE
    if KERNEL == 'linear':
        linear_model = SVM_Linear(n_concepts=n_concepts)
    else:
        linear_model = SVM(n_concepts=n_concepts)
    linear_model.train(train_embeddings[random_subset], probing_train_data)
    LINEAR_PROBE = linear_model.evaluate(test_embeddings, test_data)
    
    logger.info(f"{ds_name} probe: {LINEAR_PROBE}")
    name = generate_name(time_date, "SVM", ds_name, "random", KERNEL, n_concepts, SEED)
    
    results.append({
        "probe": "linearsvm",
        "backbone": "ViT-L/14",
        "roc auc": LINEAR_PROBE['roc_auc'],
        "pr auc": LINEAR_PROBE['pr_auc'],
        "f1_cal": LINEAR_PROBE['f1_cal'],
        "training_pool_size":subset_size,
        'kernel': KERNEL,
    })
    '''
    # =========================
    # LINEAR PROBE
    # =========================
    linear_model = LinearProbe(n_concepts=n_concepts, seed=SEED)
    linear_model.train(train_subset_embeddings, probing_train_data)
    LINEAR_PROBE = linear_model.evaluate(test_embeddings, test_data)

    logger.info(f"{ds_name} linear probe: {LINEAR_PROBE}")

    results.append({
        "backbone": "ViT-L/14",
        "probe": "linear",
        "roc auc": LINEAR_PROBE["roc_auc"],
        "pr auc": LINEAR_PROBE["pr_auc"],
        "f1_cal": LINEAR_PROBE["f1_cal"],
        "bss": LINEAR_PROBE["bss"],
        "training_pool_size": subset_size,
    })
    
    # =========================
    # MLP PROBE
    # =========================
    mlp_model = MLPProbe(
        n_concepts=n_concepts,
        seed=SEED,
        hidden_dim=min(64, 2 * subset_size)  # capacity control
    )

    mlp_model.train(train_subset_embeddings, probing_train_data)
    MLP_PROBE = mlp_model.evaluate(test_embeddings, test_data)

    #logger.info(f"{ds_name} MLP probe: {MLP_PROBE}")

    results.append({
        "backbone": "ViT-L/14",
        "probe": "mlp",
        "roc auc": MLP_PROBE["roc_auc"],
        "pr auc": MLP_PROBE["pr_auc"],
        "f1_cal": MLP_PROBE["f1_cal"],
        "bss": LINEAR_PROBE["bss"],
        "training_pool_size": subset_size,
    })
    
with open(f"{SAVE_FOLDER}/{generate_name(time_date, ds_name, SEED)}", "w") as f:
    json.dump(results, f, indent=4)
        
