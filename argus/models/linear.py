import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from typing import Dict, Tuple
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


class MultiTaskIndependentProbes(nn.Module):
    def __init__(self, input_dim: int, concept_structure: Dict[str, Tuple[int, int]]):
        super().__init__()
        self.concept_structure = concept_structure
        
        # Create a dictionary of independent linear layers
        self.heads = nn.ModuleDict()
        
        for task_name, (start, end) in concept_structure.items():
            # Calculate output dimension from the slice range (inclusive)
            output_dim = (end - start) + 1
            if output_dim == 1:
                output_dim=2
            # Each head is its own independent architectural layer
            self.heads[task_name] = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Dictionary to store independent logit outputs
        return {task_name: head(x) for task_name, head in self.heads.items()}

class MultiTaskTrainer:
    def __init__(self, input_dim: int, concept_structure: Dict[str, Tuple[int, int]], lr: float = 1e-2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.concept_structure = concept_structure
        self.model = MultiTaskIndependentProbes(input_dim, concept_structure).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(reduce='none')

    def train_epoch(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        embeddings: (N, input_dim)
        labels: (N, total_label_dim) - one-hot encoded according to concept_structure
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass: get dict of logits
        task_logits = self.model(embeddings)
        total_loss = 0

        for task_name, (start, end) in self.concept_structure.items():
            num_c = end - start + 1
            if num_c==1:
                task_ground_truth = labels[:, start]
                target_indices = (task_ground_truth > 0.5).long()
                counts = torch.bincount(target_indices, minlength=2)
            else:
                task_ground_truth = labels[:, start:end+1]
                target_indices = task_ground_truth.argmax(dim=1)
                counts = torch.bincount(target_indices, minlength=num_c)
            # Calculate loss for this specific head
            #print(counts)
            total = sum(counts)
            # Standard formula: w_i = total / (num_classes * count_i)
            weights = torch.tensor([total / (len(counts) * c) for c in counts]).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            loss = criterion(task_logits[task_name], target_indices)
            total_loss += loss
            
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def eval(self, embeddings, labels):
        """
        embeddings: (N, input_dim)
        labels: (N, total_label_dim) - one-hot encoded according to concept_structure
        """
        self.model.eval()
        embeddings = embeddings.to(self.device)
        task_logits = self.model(embeddings)
        
        labels = labels.to(self.device)
        total_loss = 0
        probs = []
        for task_name, (start, end) in self.concept_structure.items():
            num_c = end - start + 1
            if num_c == 1:
                # Slice the ground truth one-hot tensor for this specific task
                task_ground_truth = labels[:, start]
                # CrossEntropyLoss expects class indices (Long)
                # argmax converts one-hot [0, 0, 1, 0] -> 2
                target_indices = (task_ground_truth > 0.5).long()
            else:
                # Slice the ground truth one-hot tensor for this specific task
                task_ground_truth = labels[:, start:end+1]
                # CrossEntropyLoss expects class indices (Long)
                # argmax converts one-hot [0, 0, 1, 0] -> 2
                target_indices = task_ground_truth.argmax(dim=1)
            
            # Calculate loss for this specific head
            loss = self.criterion(task_logits[task_name], target_indices)
            if num_c == 1:
                probs.append(torch.nn.functional.softmax(task_logits[task_name], dim=1)[:,1].unsqueeze(-1))
            else:
                probs.append(torch.nn.functional.softmax(task_logits[task_name], dim=1))
            total_loss += loss
            
        probs = torch.cat(probs, dim=1)
        return probs, loss
    
    def evaluate(self, predictions, labels) -> dict: 
        # The raw predictions are the entire test set
        # Store the concepts from all samples in a single tensor
        #print(predictions.shape)
        
        predictions = predictions.cpu().detach()
        labels = labels.cpu().detach()
        mc_labels = []
        f1_mc = []
        preds_raw = []
        raw_predictions = []
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
            print(concept_gt_raw[i])
            print(concept_pred_raw[i])
            
            cr_raw = classification_report(concept_gt_raw[i], concept_pred_raw[i], output_dict=True)
            print(classification_report(concept_gt_raw[i], concept_pred_raw[i]))
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
                }
        logger.info(f"Evaluation f1: raw={res['f1_raw']} cal={res['f1_cal']} - roc auc: {res['rocauc']} - pr auc:{res['prauc']}")
        return res
    
    def save_trainer(self, filepath: str):
        """Saves the model, optimizer, and concept structure to a file."""
        checkpoint = {
            'input_dim': self.model.heads[next(iter(self.concept_structure))].in_features,
            'concept_structure': self.concept_structure,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Trainer saved to {filepath}")

   
def load_trainer(filepath: str, lr: float = 1e-3) -> MultiTaskTrainer:
    """Loads the checkpoint and reconstructs the MultiTaskTrainer."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    # 1. Reconstruct the trainer with the saved structure
    trainer = MultiTaskTrainer(
        input_dim=checkpoint['input_dim'],
        concept_structure=checkpoint['concept_structure'],
        lr=lr
    )
    
    # 2. Load weights and optimizer state
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Trainer loaded from {filepath}")
    return trainer