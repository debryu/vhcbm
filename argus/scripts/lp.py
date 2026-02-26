import torch
from argus.models.svm import SVM_Linear, SVM
from argus.models.linear import MultiTaskTrainer
from argus.models.base import BaseModel
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
import torch.nn.functional as F

time = datetime.datetime.now().strftime("%H_%M")
date = datetime.datetime.now().strftime("%Y_%m_%d")
time_date = f"{date}_{time}"
dict = {
    'macro auc'
    'training_pool_size'
}


def generate_name(date,model,ds_name,acq_fn,kernel,K, seed):
    return f"{date}-{model}_{ds_name}_{acq_fn}_{kernel}_{K}__SEED={seed}.json"

SEED = int(os.environ.get("SEED", 222))
set_seed(SEED)

DEVICE = str(os.environ.get("DEVICE",'cuda'))
KERNEL = os.environ.get("KERNEL", 'linear')
FAST_STORAGE = os.environ.get("FAST", "/mnt/external_storage")
SAVE_FOLDER = f"{FAST_STORAGE}/results/baselines"
MODEL_NAME = str(os.environ.get("MODEL_NAME",'ViT-L/14'))
activations_folder = os.path.join(FAST_STORAGE,"activations")
os.makedirs(os.path.join(activations_folder), exist_ok=True)
for ds_name in ['celeba', 'cub', 'shapes3d', 'dermamnist']:
    if ds_name == 'celeba':
        n_concepts = 39
        concept_structure = {   '1':(0,0),
                            '2':(1,1),
                            '3':(2,2),
                            '4':(3,3),
                            '5':(4,4),
                            '6':(5,5),
                            '7':(6,6),
                            '8':(7,7),
                            '9':(8,8),
                            '10':(9,9),
                            '11':(10,10),
                            '12':(11,11),
                            '13':(12,12),
                            '14':(13,13),
                            '15':(14,14),
                            '16':(15,15),
                            '17':(16,16),
                            '18':(17,17),
                            '19':(18,18),
                            '20':(19,19),
                            '21':(20,20),
                            '22':(21,21),
                            '23':(22,22),
                            '24':(23,23),
                            '25':(24,24),
                            '26':(25,25),
                            '27':(26,26),
                            '28':(27,27),
                            '29':(28,28),
                            '30':(29,29),
                            '31':(30,30),
                            '32':(31,31),
                            '33':(32,32),
                            '34':(33,33),
                            '35':(34,34),
                            '36':(35,35),
                            '37':(36,36),
                            '38':(37,37),
                            '39':(38,38),
                        }
    elif ds_name == 'dermamnist':
        n_concepts = 7
        concept_structure = {   'finding':(0,6),
                        }
    elif ds_name == 'cub':
        n_concepts = 112
        concept_structure = {}
        for i in range(112):
            concept_structure[f'{i}'] = (i,i)
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
       
        subset_size = 500
    steps = range(n_concepts,700, n_concepts)
    
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
     
    name = generate_name(time_date, "linear", ds_name, "random", KERNEL, n_concepts, SEED)
    
    steps = [40,100,160,220,280,340,400] #, #len(train_embeddings)
    results = []
    for i,subset_size in enumerate(tqdm(steps)):
        logger.debug(subset_size)
        set_seed(SEED)
        random_subset = random.sample(range(len(train_data)), int(subset_size))
        probing_train_data = torch.utils.data.Subset(train_data, random_subset)
        train_labels = []
        for i in probing_train_data:
            _,c,l = i
            train_labels.append(c)
        train_labels = torch.stack(train_labels, dim=0)
        test_labels = []
        for i in test_data:
            _,c,l = i
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
                pat +=1
            if pat >3 or train_loss < 0.01:
                break
            if e%100==0:
                logger.debug(f"{train_loss} patience {pat}")
        
        probs, loss = trained_model.eval(test_embeddings, test_labels)
        
        logger.debug(f"Eval loss:{loss}")
        LINEAR_PROBE = trained_model.evaluate(probs, test_labels)
        trained_model.save_trainer(os.path.join(FAST_STORAGE,f"results/baselines/{name}.pth"))
        
        logger.info(f"{ds_name} probe: {LINEAR_PROBE}")
        name = generate_name(time_date, "SVM", ds_name, "random", KERNEL, n_concepts, SEED)
        
        results.append({
            "backbone": "ViT-L/14",
            "rocauc": LINEAR_PROBE['rocauc'],
            "prauc": LINEAR_PROBE['prauc'],
            "f1_raw": LINEAR_PROBE['f1_raw'],
            "f1_cal": LINEAR_PROBE['f1_cal'],
            "training_pool_size":subset_size,
            "concept_used":subset_size*len(concept_structure),
            "pool": 'random',
            'kernel': KERNEL,
        })
        
    # Optionally run a version with a custom subset
    subset = []
    if len(subset)>1:
        # LINEAR_PROBE
        if KERNEL == 'linear':
            linear_model = SVM_Linear(n_concepts=n_concepts)
        else:
            linear_model = SVM(n_concepts=n_concepts)
        linear_model.train(train_embeddings[subset], probing_train_data)
        LINEAR_PROBE = linear_model.evaluate(test_embeddings, test_data)
        
        logger.info(f"{ds_name} probe: {LINEAR_PROBE}")
        
        
        results.append({
            "backbone": "ViT-L/14",
            "macro auc": LINEAR_PROBE['auc'],
            "all_min_aucs": LINEAR_PROBE['all_min_aucs'],
            "min auc": LINEAR_PROBE['minauc'],
            "f1": LINEAR_PROBE['f1'],
            "training_pool_size":subset_size,
            "all_f1s": LINEAR_PROBE['all_f1s'],
            'kernel': KERNEL,
            'pool': 'fixed',
        })    
    
    with open(f"{SAVE_FOLDER}/{name}", "w") as f:
        json.dump(results, f, indent=4)
        
