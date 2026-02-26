import torch
from argus.models.svm import SVM_Linear, SVM
from CQA.datasets import GenericDataset
import json, os
from sklearn.metrics import precision_recall_curve, auc
from argus.metrics.auc import MacroAUC, RocAUC
from loguru import logger
from sklearn.metrics import classification_report
import numpy as np
import random
from argus.utils.utils import set_seed
import copy
from argus.models.base import train_LR_on_concepts

SEED = int(os.environ.get("SEED", 42))
DS_NAME = os.environ.get("DS_NAME", "dermamnist")
set_seed(SEED)
MODEL_NAME = str(os.environ.get("MODEL_NAME",'ViT-L/14'))
DEVICE = str(os.environ.get("DEVICE",'cuda'))
FAST_STORAGE = str(os.environ.get("FAST"))
SAVE_FOLDER = f"{FAST_STORAGE}/results/baselines"
activations_folder = os.path.join(FAST_STORAGE,"activations")
os.makedirs(os.path.join(activations_folder), exist_ok=True)

ds_name = DS_NAME
#for ds_name in ['dermamnist', 'shapes3d', 'cub', 'celeba']:
if ds_name == 'celeba':
    n_concepts = 39
    subset_size = 467
elif ds_name == 'dermamnist':
    n_concepts = 7
    subset_size = 227
elif ds_name == 'cub':
    n_concepts = 112
    subset_size = 153
elif ds_name == 'shapes3d':
    n_concepts = 42
    subset_size = 420
else:
    n_concepts = None
    subset_size = 500

    
def evaluate_clip_scores(n_concepts, clip_scores, dataset):
    min_pr_aucs = []
    pr_aucs = []
    roc_aucs = []
    f1 = []
    reports = []
    predictions = clip_scores.cpu()
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][1])
    labels = torch.stack(labels, dim=0)
    # Create a calibration set and find the best threshold
    N = predictions.shape[0]
    train_ratio = 0.2
    # Generate a random permutation of indices
    perm = torch.randperm(N)
    n_train = int(train_ratio * N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    # Apply the same indices to both tensors
    pred_train = predictions[train_idx]
    pred_test = predictions[test_idx]
    cf = copy.deepcopy(pred_test).to('cpu')
    
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    logger.info("Training Logistic Regression on All Concepts")
    
    W,B = train_LR_on_concepts(pred_train.cpu(), labels_train.cpu())
    cf *= W
    cf += B
    probs = torch.nn.functional.sigmoid(cf)      
    
    # The calibrated predictions are only on the test-test set
    preds_calibrated = (probs > 0.5).int()
    
    # The raw predictions are the entire test set
    preds_raw = (torch.nn.functional.sigmoid(predictions) > 0.5).int()
    
    # Store the concepts from all samples in a single tensor
    concept_pred_raw = []
    concept_pred_calibrated = []
    concept_gt_raw = []
    concept_pred = []
    # Collect calibrated X,y
    for i in range(labels.shape[1]):
        concept_pred.append(predictions[:,i].numpy())
        concept_pred_raw.append(preds_raw[:,i].numpy())
        concept_gt_raw.append(labels[:,i].numpy()) 
        
    # Collect calibrated X,y   
    concept_gt_calibrated = []
    for i in range(labels_test.shape[1]):
        concept_pred_calibrated.append(preds_calibrated[:,i].numpy())
        concept_gt_calibrated.append(labels_test[:,i].numpy())    
    
    f1_raw = []
    f1_calibrated = []
    acc = []
    rec = []
    prec = []
    reports = []
    pr_aucs = []
    roc_aucs = []
    logger.debug("Computing concept-wise auc and f1")
    # Compute concept-wise accuracy metrics
    for i in range(labels.shape[1]):
        cr_calibrated = classification_report(concept_gt_calibrated[i], concept_pred_calibrated[i], output_dict=True)
        cr_raw = classification_report(concept_gt_raw[i], concept_pred_raw[i], output_dict=True)
        
        f1_calibrated.append(cr_calibrated['macro avg']['f1-score'])  # type:ignore
        f1_raw.append(cr_raw['macro avg']['f1-score'])  # type:ignore
        reports.append((cr_calibrated, cr_raw))
        res1 = RocAUC(concept_gt_raw[i], concept_pred[i])
        res2 = MacroAUC(concept_gt_raw[i], concept_pred[i])
        logger.debug(f"ROC AUC: {res1.roc_auc} PR AUC: {res2.prauc0} MACRO AUC:{res2.macro_auc}")
        roc_aucs.append(res1.roc_auc)
        pr_aucs.append(res2.min_auc)
        
        
    res = { 'f1_cal': np.mean(f1_calibrated),
            'f1_raw': np.mean(f1_raw),
            'all_f1_cal': f1_calibrated,
            'all_f1_raw': f1_raw,
            'acc': np.mean(acc),
            'rec': np.mean(rec),
            'prec':np.mean(prec),
            'prauc':np.mean(pr_aucs),
            'rocauc':np.mean(roc_aucs),
            'all_pr_aucs':pr_aucs,
            'all_roc_aucs':roc_aucs,
            'reports':reports,
            }
    logger.info(f"Evaluation f1: raw={res['f1_raw']} cal={res['f1_cal']} - roc auc: {res['rocauc']} - pr auc:{res['prauc']}")
    return res
    
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
random_subset = random.sample(range(len(train_data)), subset_size)
probing_train_data = torch.utils.data.Subset(train_data, random_subset)
test_data = GenericDataset(ds_name=ds_name,split = 'test')


# CLIP SCORES BASELINE
text_normed = text_embeddings/(text_embeddings.norm(p=2, dim=-1, keepdim = True).float())
train_normed = train_embeddings/(train_embeddings.norm(p=2, dim=-1, keepdim = True).float())
test_normed = test_embeddings/(test_embeddings.norm(p=2, dim=-1, keepdim = True).float())
train_feats = train_embeddings @ text_embeddings.T
test_feats = test_embeddings @ text_embeddings.T
# Normalize for SVM training
CLIP = evaluate_clip_scores(n_concepts=n_concepts, clip_scores=test_feats, dataset=test_data)

# LINEAR BASELINE
linear_model = SVM_Linear(n_concepts=n_concepts)
linear_model.train(train_embeddings, train_data)
LINEAR = linear_model.evaluate(test_embeddings, test_data)

# NON-LINEAR BASELINE
model = SVM(n_concepts=n_concepts)
model.train(train_embeddings, train_data)
NONLINEAR = model.evaluate(test_embeddings, test_data)

os.makedirs(SAVE_FOLDER, exist_ok=True)
with open(os.path.join(SAVE_FOLDER, f"{ds_name}.json"), "w") as f:
    logger.info(f'CLIP:{CLIP}, LINEAR:{LINEAR}, NON-LINEAR:{NONLINEAR}')
    json.dump({'CLIP':CLIP, 'LINEAR':LINEAR, 'NON-LINEAR':NONLINEAR}, f, indent=4)
