import torch
from loguru import logger
from CQA.datasets import GenericDataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import copy
from typing import Any, Optional
import json
from abc import ABC, abstractmethod
import random 
from argus.datasets import get_augmentations
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, auc
import matplotlib.pyplot as plt
import math
from argus.metrics.auc import MacroAUC, RocAUC

def train_LR_on_concepts(conc_pred:torch.Tensor,conc_gt:torch.Tensor):
    n_concepts = conc_pred.shape[1]
    W = []
    B = []
    for i in tqdm(range(n_concepts), desc="Fitting Logistic Regression"):
        X = conc_pred[:,i].numpy().reshape(-1,1)# sklearn requires 2d input 
        
        y = conc_gt[:,i]
        LR = LogisticRegression(C=1,class_weight='balanced')
        LR.fit(X,y)
        w = LR.coef_[0][0]  # Slope
        b = LR.intercept_[0] # Intercept
        
        W.append(w)
        B.append(b)
        zeros = np.count_nonzero(LR.predict(X) == 0)
    return torch.tensor(W),torch.tensor(B)

class BaseModel():
    def __init__(self, args):
        self.concept_used = 0
        self.embedding_pool = None
        self.data = None
        self.masking = False
        self.num_concepts:Any = None
        self.num_one_hot_concepts:Any = None
        self.concept_counts = None
        self.label_counts = None
        self.args: Any = None
        self.text_embeddings: Any = None
        self.pool = []
        self.training_pool = []
        pass
    
    def get_name(self):
        return f"{self.args.dataset}-{len(self.training_pool)}-{self.args.acq_fn}-{self.args.kernel}-{self.args.seed}"
    
    def set_mask(self, embeddings, num_concepts):
        self.mask = torch.ones((len(embeddings),num_concepts))

    def get_probs(self, predictions: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.sigmoid(predictions)  
    
    def save(self, path: str):
        checkpoint = {
            "args": self.args,
            "concept_structure": self.concept_structure,
            "models_state_dict": [m.state_dict() for m in self.models],
            "likelihoods_state_dict": [l.state_dict() for l in self.likelihoods],
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device):
        checkpoint = torch.load(path, map_location=device,  weights_only=False)

        args = checkpoint["args"]
        args.n_inducing_points = int(path.split("-")[1])
        concept_structure = checkpoint["concept_structure"]

        model_obj = cls(args, concept_structure)

        for m, state in zip(model_obj.models, checkpoint["models_state_dict"]):
            m.load_state_dict(state)

        for l, state in zip(model_obj.likelihoods, checkpoint["likelihoods_state_dict"]):
            l.load_state_dict(state)

        return model_obj

    @staticmethod
    def evaluate_model(model, predictions, labels):
        return model.evaluate_new(predictions, labels)
    
    def evaluate_new(self, predictions, labels) -> dict: 
        # The raw predictions are the entire test set
        # Store the concepts from all samples in a single tensor
        #print(predictions.shape)
        mc_labels = []
        f1_mc = []
        preds_raw = []
        for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
            num_classes = -(idx_range[0]-idx_range[1])+1
            if num_classes == 1:
                lab = labels[:, idx_range[0]]
                #print(predictions[0])
                pre = (predictions[:, idx_range[0]:idx_range[1]+1] > 0.5).long()
                #print(lab[0:10])
                #print(pre[0:10])
                preds_raw.append(pre)
            else:
                lab = torch.argmax(labels[:, idx_range[0]:idx_range[1]+1], dim=1)
                pre = torch.argmax(predictions[:, idx_range[0]:idx_range[1]+1], dim=1)
                prediction = torch.nn.functional.one_hot(pre, num_classes=num_classes)
                preds_raw.append(prediction)
            cr = classification_report(lab, pre, output_dict=True)
            f1_mc.append(cr['macro avg']['f1-score'])  # type:ignore
        
        preds_raw = torch.cat(preds_raw, dim=1)
        f1_raw = []
        f1_calibrated = []
        acc = []
        rec = []
        prec = []
        reports = []
        pr_aucs = []
        roc_aucs = []
        concept_pred_raw = []
        concept_gt_raw = []
        concept_pred_float = []
        #print(preds_raw.shape)
        for i in range(labels.shape[1]):
            concept_pred_raw.append(preds_raw[:,i].long().numpy())
            concept_gt_raw.append(labels[:,i].numpy()) 
            concept_pred_float.append(predictions[:,i].numpy())
            
        logger.debug("Computing concept-wise auc and f1")
        # Compute concept-wise accuracy metrics
        for i in range(labels.shape[1]):
            #print(concept_gt_raw[i])
            #print(concept_pred_raw[i])
            
            cr_raw = classification_report(concept_gt_raw[i], concept_pred_raw[i], output_dict=True)
            #print(classification_report(concept_gt_raw[i], concept_pred_raw[i]))
            #input("..")
            f1_raw.append(cr_raw['macro avg']['f1-score'])  # type:ignore
            #reports.append(cr_raw)
            res1 = RocAUC(concept_gt_raw[i], concept_pred_float[i])
            res2 = MacroAUC(concept_gt_raw[i], concept_pred_float[i])
            logger.debug(f"ROC AUC: {res1.roc_auc} PR AUC: {res2.min_auc} F1: {f1_mc}")
            roc_aucs.append(res1.roc_auc)
            pr_aucs.append(res2.min_auc)
            
            
        res = { 'f1_cal': np.mean(f1_mc),
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
                'training_pool_size': len(self.training_pool),
                'training_pool': list(self.training_pool),
                'concept_used': self.concept_used,
                }
        logger.info(f"Evaluation f1: raw={res['f1_raw']} cal={res['f1_cal']} - roc auc: {res['rocauc']} - pr auc:{res['prauc']}")
        return res
        
    def set_pool(self, embeddings):
        self.pool = list(range(len(embeddings)))
        
    def set_training_pool(self, idx_list:list):
        for l in idx_list:
            print(l)
            self.pool.remove(l)
            self.training_pool.append(l)
            if self.masking:
                # All concepts must be taken into consideration (= not masked)
                self.mask[l] = torch.zeros(self.num_one_hot_concepts)
    
    def show_training_stats(self, dataset):
        concepts = []
        labels = []
        for sample in self.training_pool:
            _, c, l = dataset[sample]
            concepts.append(c)
            labels.append(l)
            
        concepts = torch.stack(concepts, dim=0)
        labels = torch.stack(labels, dim=0)
        concepts_freq = torch.sum(concepts, dim=0)/concepts.shape[0]
        labels_freq = torch.sum(labels)/concepts.shape[0]
        concepts_count = torch.sum(concepts, dim=0)
        labels_count = torch.sum(labels)
        delta_c = concepts_count.tolist()
        delta_l = concepts_count.tolist()
        if self.concept_counts is not None:
            delta_c = concepts_count -  self.concept_counts 
        if self.label_counts is not None:
            delta_l = labels_count - self.label_counts
        self.concept_counts = concepts_count
        self.label_counts = labels_count
        logger.warning(f"Concepts:\n{concepts_count}{delta_c}\n{concepts_freq}\nLabels:\n{labels_count}{delta_l} - {labels_freq}")
        
            
    
    def set_data(self, data):
        self.data = data
        
    def set_text_embeddings(self, text_embeddings):
        self.text_embeddings = text_embeddings
    
    def sanitize_args(self,args):
        args_dict = vars(args)
        sanitized = {}
        for k, v in args_dict.items():
            try:
                json.dumps(v)  # try serializing
                sanitized[k] = v
            except (TypeError, OverflowError):
                sanitized[k] = str(type(v))  # or custom placeholder
        return sanitized
    
    @abstractmethod
    def train(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def eval(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def latent_variance(self, *args, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def mean(self, *args, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def entropy(self, dataset) -> torch.Tensor:
        pass
    
    @abstractmethod
    def probs(self, dataset) -> torch.Tensor:
        pass
    
    def sample_random_c(self, embeddings, K=100):
        subset = random.sample(self.pool, K)
        search_space = embeddings[subset]
        
        selected_concepts = torch.ones(self.mask.shape[1]).bool()

        for i,idx in enumerate(subset):
            self.mask[idx,selected_concepts] = 0
            
        # Return indexes
        return [i for i in subset]
            
    def train_loop(self, embeddings, data, testx,testy, acq_fn = 'random', K = 100, variation_list = [''], **kwargs):
        # 1) Sample points
        if acq_fn == 'random':
            logger.debug(f"Added {K} random samples.")
            indexes = random.sample(self.pool, K)
        elif acq_fn == 'randomc':
            indexes = self.sample_random_c(embeddings, K)
        else:
            raise NotImplementedError()
            indexes = random.sample(self.pool, K)
        
        for i in indexes:
            self.training_pool.append(i)
            self.pool.remove(i)
        
        
        train_concept_labels = []
        for i in range(len(embeddings)):
            concepts = data[i][1]
            train_concept_labels.append(concepts)
        train_concept_labels = torch.stack(train_concept_labels, dim=0).to(self.args.device)
        
        
        train_labels = []
        for i in range(len(data)):
            labels = data[i][2]
            train_labels.append(labels)
        train_labels = torch.stack(train_labels, dim=0).to(self.args.device)
        train_embeddings = embeddings[self.training_pool].to(self.args.device)

        for aug in variation_list:
            if aug == '':
                continue
            logger.info(f'Adding {aug} variation')
            augmented_embs, augmented_labels = get_augmentations(self.args.dataset, indexes, aug, device=self.args.device)
            train_embeddings = torch.cat((train_embeddings,augmented_embs), dim=0)
            train_concept_labels = torch.cat((train_concept_labels,augmented_labels), dim=0)
                
        self.reset_models()  
        self.train(embeddings.to(self.device),train_concept_labels, patience=self.args.gp_patience, text_embeddings = kwargs.get("text_embeddings", None))
        out_dict = self.eval(testx.to(self.args.device))
              
        pred_probs = out_dict['probs']
        pred_vars = out_dict['std']
        dict = self.evaluate_new(pred_probs.cpu(),testy.cpu())
        dict.update(self.sanitize_args(self.args))
        dict.update({'discretization':'threshold'})
        return dict