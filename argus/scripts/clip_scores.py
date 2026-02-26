import torch
from argus.models.svm import SVM_Linear, SVM
from CQA.datasets import GenericDataset
import json, os
from sklearn.metrics import precision_recall_curve, auc
from loguru import logger
from sklearn.metrics import classification_report
import numpy as np
from argus.metrics.auc import MacroAUC, RocAUC
import datetime


DEVICE = str(os.environ.get("DEVICE",'cuda'))
KERNEL = os.environ.get("KERNEL", 'linear')
FAST_STORAGE = os.environ.get("FAST", "/mnt/external_storage")
SAVE_FOLDER = f"{FAST_STORAGE}/results/baselines"
MODEL_NAME = str(os.environ.get("MODEL_NAME",'ViT-L/14'))
activations_folder = os.path.join(FAST_STORAGE,"activations")


    
def generate_name(date):
    return f"{date}-CLIPSCORES.json"

def evaluate_clip_scores(n_concepts, clip_scores, dataset, text = ''):
    aucs = []
    prs = []
    f1_raw = []
    for c_id in range(n_concepts):
        concept_labels = []
        inverted_labels = []
        # Collect train labels
        for i in range(len(dataset)):
            concept_labels.append(dataset[i][1][c_id])
            inverted_labels.append(1-dataset[i][1][c_id])
        concept_labels = torch.stack(concept_labels, dim=0).detach().cpu().numpy()
        predicted_concept = clip_scores[:,c_id].detach().cpu().numpy()
        pr = MacroAUC(concept_labels,predicted_concept)
        prs.append(pr.prauc0)
        auc = RocAUC(concept_labels,predicted_concept)
        aucs.append(auc.roc_auc)
        boolean_predictions = (predicted_concept > 0.0)
        cr = classification_report(concept_labels,boolean_predictions, output_dict=True)
        f1_raw.append(cr['macro avg']['f1-score'])  #type:ignore
        logger.debug(f"Evaluating CLIP scores for concept {c_id}.")
    logger.info(np.mean(aucs))
    return {'rocauc':np.mean(aucs),
            'prauc':np.mean(prs),
            'f1_raw':np.mean(f1_raw),
            'all_f1_raw': f1_raw,
            'all_pr_aucs':prs,
            'all_roc_aucs':aucs,
            }    
    
savepath = './figs/clip'
results = []
for ds_name in ['shapes3d','celeba','dermamnist','cub']:
    if ds_name == 'celeba':
        n_concepts = 39
    elif ds_name == 'dermamnist':
        n_concepts = 7
    elif ds_name == 'cub':
        n_concepts = 112
    elif ds_name == 'shapes3d':
        n_concepts = 42
    else:
        n_concepts = None
        
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
    val_data = GenericDataset(ds_name=ds_name, split= 'val')
    test_data = GenericDataset(ds_name=ds_name,split = 'test')


    # CLIP SCORES BASELINE
    text_normed = text_embeddings/(text_embeddings.norm(p=2, dim=-1, keepdim = True).float())
    train_normed = train_embeddings/(train_embeddings.norm(p=2, dim=-1, keepdim = True).float())
    val_normed = val_embeddings/(val_embeddings.norm(p=2, dim=-1, keepdim = True).float())
    test_normed = test_embeddings/(test_embeddings.norm(p=2, dim=-1, keepdim = True).float())
    train_feats = train_normed @ text_normed.T
    test_feats = test_normed @ text_normed.T
    val_feats = val_normed @ text_normed.T
    # Normalize for SVM training
    #CLIP = evaluate_clip_scores(n_concepts=n_concepts, clip_scores=val_feats, dataset=val_data)
    CLIP = evaluate_clip_scores(n_concepts=n_concepts, clip_scores=test_feats, dataset=test_data, text = '_ensembled')
    time = datetime.datetime.now().strftime("%H_%M")
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    time_date = f"{date}_{time}"
    
    CLIP.update({
                "backbone": MODEL_NAME,
                "model":"clip_scores",
                "dataset": ds_name,
                })
    print(CLIP)
    results.append(CLIP)
    
name = generate_name(time_date)    
with open(f"{SAVE_FOLDER}/{name}", "w") as f:
        json.dump(results, f, indent=4)