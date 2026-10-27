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
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        args = checkpoint["args"]
        concept_structure = checkpoint["concept_structure"]
        states = checkpoint["models_state_dict"]
        embedding_dim = states[0][
            'variational_strategy.base_variational_strategy.inducing_points'
        ].shape[2]

        # Build the container with dummy args, just to get self.likelihoods/self.models lists
        max_inducing = max(
            s['variational_strategy.base_variational_strategy.inducing_points'].shape[1]
            for s in states
        )
        args.n_inducing_points = max_inducing
        args.inducing_points = torch.zeros(max_inducing + 1, embedding_dim)
        model_obj = cls(args, concept_structure)

        # Rebuild each model using inducing points directly from checkpoint
        for i, (m, state) in enumerate(zip(model_obj.models, states)):
            inducing_points_ckpt = state[
                'variational_strategy.base_variational_strategy.inducing_points'
            ][0].to(device)  # take first latent -> shape: [n_inducing, embedding_dim]
            
            num_classes = m.variational_strategy.num_tasks
            
            # Rebuild the model with correct inducing points from checkpoint
            from argus.models.MCSVGPDO import MCSVGPModel
            new_model = MCSVGPModel(
                inducing_points_ckpt,
                args.kernel,
                args.learn_mean,
                num_classes=num_classes,
                num_latents=args.num_latents
            ).to(device)
            new_model.load_state_dict(state)
            model_obj.models[i] = new_model

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
    
    def evaluate_new_old(self, predictions, labels) -> dict: 
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
        probs = self.get_probs(cf)      
        
        # The calibrated predictions are only on the test-test set
        preds_calibrated = (probs > 0.5).int()
        
        # The raw predictions are the entire test set
        preds_raw = (self.get_probs(predictions) > 0.5).int()
        
        # Store the concepts from all samples in a single tensor
        concept_pred_raw = []
        concept_pred_calibrated = []
        concept_gt_raw = []
        concept_pred = []
        print(predictions.shape)
        print(labels.shape)
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
            logger.debug(f"ROC AUC: {res1.roc_auc} PR AUC: {res2.prauc0} F1 cal and raw:{np.mean(f1_calibrated)} {np.mean(f1_raw)}")
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
                'training_pool_size': len(self.training_pool),
                'training_pool': list(self.training_pool),
                }
        logger.info(f"Evaluation f1: raw={res['f1_raw']} cal={res['f1_cal']} - roc auc: {res['rocauc']} - pr auc:{res['prauc']}")
        return res
    
    def evaluate(self, predictions, labels, mode = 'lr') -> dict: 
        cf = copy.deepcopy(predictions).to('cpu')
        
        if mode == 'lr':
            logger.info("Training Logistic Regression on All Concepts")
        
            W,B = train_LR_on_concepts(predictions.cpu(), labels.cpu())
            
            cf *= W
            cf += B
            #
            
            probs = torch.nn.functional.sigmoid(cf)      
            preds = (probs > 0.5).int()
        
        else:
            preds = (predictions > 0.5).int()
        # Store the concepts from all samples in a single tensor
        concept_pred = []
        concept_pred_real_value = []
        concept_gt = []
        for i in range(labels.shape[1]):
            concept_pred_real_value.append(predictions[:,i].numpy())
            concept_pred.append(preds[:,i].numpy())
            concept_gt.append(labels[:,i].numpy())    
        
        f1 = []
        acc = []
        rec = []
        prec = []
        reports = []
        aucs = []
        macro_auc = []
        logger.debug("Computing concept-wise auc and f1")
        # Compute concept-wise accuracy metrics
        for i in range(labels.shape[1]):
            # COmpute AUC-ROC
            if mode == 'lr':
                precision, recall, thresholds = precision_recall_curve(concept_gt[i], concept_pred[i])
            else:
                precision, recall, thresholds = precision_recall_curve(concept_gt[i], concept_pred_real_value[i])
            
            logger.error(f"Concept:{i}   {concept_pred[i]} - {concept_gt[i]}")
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            logger.debug(f"Best threshold is {best_threshold} with f1={f1_scores[best_idx]}")
            # Compute PR AUC
            ress = MacroAUC(concept_gt[i], concept_pred_real_value[i])
            logger.debug(f"PR AUC: {ress.prauc0} MACRO AUC:{ress.macro_auc}")
            aucs.append(ress.prauc0)
            macro_auc.append(ress.macro_auc)
            '''
            # Plot PR curve
            plt.figure()
            plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            plt.legend(loc="lower left")

            # Save the figure
            plt.savefig(f"./results/images/pr_auc/C{i}-{self.get_name()}-M={mode}.png", dpi=300)  # you can change filename/format
            plt.close()  # close the figure if running in a loop
            '''
            cr = classification_report(concept_gt[i], concept_pred[i], output_dict=True)
            #print(cr)
            f1.append(cr['macro avg']['f1-score'])  # type:ignore
            acc.append(cr['accuracy'])  # type:ignore
            rec.append(cr['macro avg']['recall'])   # type:ignore
            prec.append(cr['macro avg']['precision'])   # type:ignore
            reports.append(cr)
            
        res = { 'f1': np.mean(f1),
                'acc': np.mean(acc),
                'rec': np.mean(rec),
                'prec':np.mean(prec),
                'reports':reports,
                'discretization':mode,
                'training_pool_size': len(self.training_pool),
                'auc':np.mean(aucs),
                'macro auc':np.mean(macro_auc),
                'all_macro_aucs': macro_auc,
                'training_pool': list(self.training_pool),
                }
        logger.info(f"Evaluation f1: {res['f1']} - auc: {res['auc']} - macro auc:{res['macro auc']}")
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
    
    def sample_active1(self, embeddings, K=100):
        subset = random.sample(self.pool, 2*K)
        search_space = embeddings[subset]
        output = self.eval(search_space)
        preds = output['preds']
        stds = output['std']
        probs = output['probs']
        conc = output['concentration']
        gt = self.train_labels
        mask = preds.bool()
        
        selected_stds = stds[mask].view(stds.shape[0], -1)
        unc_score = torch.sum(selected_stds, dim=1) 
        print(unc_score[0:10])
        
        values, indexes = torch.topk(unc_score, K, sorted=True, largest = True)
        
        for i in indexes:
            print(conc[i])
            logger.warning(f"was {torch.argmax(gt[self.pool[i]])} predicted {torch.argmax(preds[i])} with uncs {stds[i]}")
        # Return indexes
        return [self.pool[i] for i in indexes]
    
    def sample_active_c(self, embeddings, K=100):
        subset = random.sample(self.pool, 2*K)
        search_space = embeddings[subset]
        output = self.eval(search_space)
        preds = output['preds']
        stds = output['std']
        probs = output['probs']
        conc = output['concentration']
        gt = self.train_labels
        mask = preds.bool()
        
        selected_stds = stds[mask].view(stds.shape[0], -1)
        print("stds", selected_stds)
        selected_concepts = torch.argmin(selected_stds, dim=1)
        selected_concepts = torch.ones(selected_concepts.shape) - selected_concepts
        unc_score = torch.sum(selected_stds, dim=1) 
        print(unc_score[0:10])
        
        values, indexes = torch.topk(unc_score, K, sorted=True, largest = True)
        
        for i in indexes:
            print(conc[i])
            self.mask[self.pool[i],selected_concepts[i]] = 0
            selected_concept = selected_concepts[i]
            idx_range = list(self.concept_structure.values())[selected_concept]
            print(selected_concept, self.mask[self.pool[i]])
            logger.warning(f"was {torch.argmax(gt[self.pool[i]])} predicted {torch.argmax(preds[i])} with uncs {stds[i, idx_range[0]:idx_range[1]+1]}")
        # Return indexes
        
        return [self.pool[i] for i in indexes]
    
    def sample_active_cc(self, embeddings, K=100):
        subset = random.sample(self.pool, 2*K)
        search_space = embeddings[subset]
        output = self.eval(search_space)
        preds = output['preds']
        stds = output['std']
        probs = output['probs']
        conc = output['concentration']
        gt = self.train_labels
        mask = preds.bool()
        
        selected_stds = stds[mask].view(stds.shape[0], -1)
        print("stds", selected_stds)
        selected_concepts = torch.argmin(selected_stds, dim=1)
        selected_concepts = (torch.ones(selected_concepts.shape) - selected_concepts).long()
        unc_score = torch.sum(selected_stds, dim=1) 
        
        
        values, indexes = torch.topk(unc_score, K, sorted=True, largest = True)
        
        for i, idx in enumerate(indexes):
            self.mask[self.pool[subset[idx]],selected_concepts[idx]] = 0
            print(self.mask.shape)
            print(i, self.pool[i], self.pool[idx])
            
            selected_concept = selected_concepts[i]
            idx_range = list(self.concept_structure.values())[selected_concept]
            print(selected_concept, self.mask[self.pool[i]])
            logger.warning(f"was {torch.argmax(gt[self.pool[i]])} predicted {torch.argmax(preds[i])} with uncs {stds[i, idx_range[0]:idx_range[1]+1]}")
        # Return indexes
        return [self.pool[i] for i in indexes]
    
    def get_random_one_hot(self, num_rows, num_cols, device='cpu'):
        # 1. Generate random column indices: one for each row
        # Shape: (num_rows, 1)
        random_indices = torch.randint(0, num_cols, (num_rows, 1), device=device)
        
        # 2. Create a base tensor of zeros
        one_hot = torch.zeros(num_rows, num_cols, device=device)
        
        # 3. Scatter a '1' into the random index for each row
        # scatter_(dimension, index_tensor, value)
        one_hot.scatter_(1, random_indices, 1.0)
        
        return one_hot.long()#, random_indices.squeeze()
    
    def get_uniform_one_hot(self, num_rows, num_cols, device='cpu'):
        # 1. Generate random column indices: one for each row
        # Shape: (num_rows, 1)
        if num_rows % num_cols != 0:
            clip_after = num_rows
            num_rows = num_rows + num_rows % num_cols
            samples_per_idx = num_rows // num_cols
            identity = torch.eye(num_cols, device=device)
            one_hot_tensor = identity.repeat_interleave(samples_per_idx, dim=0)[:clip_after]
        else:
            samples_per_idx = num_rows // num_cols
            identity = torch.eye(num_cols, device=device)
            one_hot_tensor = identity.repeat_interleave(samples_per_idx, dim=0)
        
        return one_hot_tensor.long()#, random_indices.squeeze()
    
    def sample_random_c(self, embeddings, K=100):
        subset = random.sample(self.pool, K)
        search_space = embeddings[subset]

        #selected_concepts = torch.argmax(self.get_random_one_hot(K, len(self.concept_structure)), dim=1)
        selected_concepts = torch.ones(self.mask.shape[1]).bool()

        for i,idx in enumerate(subset):
            self.mask[idx,selected_concepts] = 0
            
            
        # Return indexes
        return [i for i in subset]
    
    def sample_random_uni_c(self, embeddings, K=100):
        subset = random.sample(self.pool, K)
        search_space = embeddings[subset]

        selected_concepts = torch.argmax(self.get_uniform_one_hot(K, len(self.concept_structure)), dim=1)
        
        for i,idx in enumerate(subset):
            print(len(self.pool))
            print(idx)
            print(self.mask.shape)
            print(i)
            print(selected_concepts.shape)
            self.mask[idx,selected_concepts[i]] = 0
            selected_concept = selected_concepts[i]
            idx_range = list(self.concept_structure.values())[selected_concept]
            print(selected_concept, self.mask[idx])
            
        # Return indexes
        return [i for i in subset]
    
    def sample_active_ours(self, embeddings, K=100, multiclass = True):
        subset = random.sample(self.pool, K)
        search_space = embeddings[subset]
        output = self.evalmc(search_space)
        preds = output['preds']
        means = output['means']
        stds = output['std']
        probs = output['probs']
        conc = output['concentration']
        gt = self.train_labels
        num_MC_samples = 30
        num_samples = means.shape[0]
    
        X = torch.zeros((num_samples*num_MC_samples, means.shape[1]))
        for i in range(num_MC_samples):
            start = i * num_samples
            end = (i + 1) * num_samples
            
            # Generate noisy samples
            X[start:end, :] = means + stds * torch.randn_like(means)
        
        # 30 samples of the same image, then 30 samples of the next image, etc.

        #Y = train_y.repeat(num_MC_samples, 1).flatten()
        
        if X.shape[1] != self.num_concepts:
            X_binary = torch.zeros((num_samples*num_MC_samples, self.num_concepts))
            for i in range(X_binary.shape[1]):
                X_binary[:,i] = torch.nn.functional.softmax(X[:,2*i:2*i+2], dim=1)[:,1]
            X = X_binary
        else:
            for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
                num_classes = len(range(idx_range[0],idx_range[1]+1))
                X[:,idx_range[0]:idx_range[1]+1] = torch.softmax(X[:, idx_range[0]:idx_range[1]+1], dim=1)   

        # Reshape the tensor in [number of samples, number of MC samples, number of concepts]
        # Current shape: (num_MC_samples * num_samples, num_concepts)
        # 1. Reshape to (num_MC_samples, num_samples, num_concepts)
        # 2. Permute/Transpose to (num_samples, num_MC_samples, num_concepts)
        X = X.view(num_MC_samples, num_samples, -1).permute(1, 0, 2)
        #print(X.shape)
        #print(X[0,:,24])
        preds = torch.mean(X, dim=1)
        # estimate uncertainty
        if preds.shape[-1] == len(self.concept_structure):
            unc = torch.zeros((preds.shape[0], self.num_concepts))
            for i in range(unc.shape[1]):
                unc[:,i] = 1 - torch.maximum(preds[:,i], 1-preds[:,i])
        else:
            unc = torch.zeros((preds.shape[0], len(self.concept_structure)))
            for task_id, (task_name, idx_range) in enumerate(self.concept_structure.items()):
                unc[:,task_id] = 1 - torch.max(preds[:, idx_range[0]:idx_range[1]+1], dim=1)[0]   
        
        unc2 = copy.deepcopy(unc)
        for i in range(unc2.shape[0]):
            unc2[i] = unc2[i]/torch.sum(unc2, dim=1)[i]
        #import pdb; pdb.set_trace()
        thresholds = torch.quantile(unc, q=0.367, dim=1)
        selected_concepts = (unc < thresholds.unsqueeze(-1))
        num = 60*len(self.concept_structure) - (~selected_concepts).sum()
        selected_concepts = (unc < thresholds.unsqueeze(-1)).long()
        
        if num > 0:
            # We need MORE 'False's (more annotated concepts). 
            # Pick the highest values that are currently '0'
            eligible_indices = (selected_concepts == 1).nonzero(as_tuple=True)
            values_to_check = unc[eligible_indices]
            
            # Get indices of the smallest values among those currently unselected
            _, move_idx = torch.topk(values_to_check, k=num, largest=True)
            
            # Update the mask
            selected_concepts[eligible_indices[0][move_idx], eligible_indices[1][move_idx]] = 0
        elif num < 0:
            # We need FEWER 'False's. 
            # Pick the largest values that are currently '0' (closest to the threshold)
            eligible_indices = (selected_concepts == 0).nonzero(as_tuple=True)
            values_to_check = unc[eligible_indices]
            
            # Get indices of the largest values among those currently selected
            _, move_idx = torch.topk(values_to_check, k=abs(num), largest=False)
            
            # Update the mask
            selected_concepts[eligible_indices[0][move_idx], eligible_indices[1][move_idx]] = 1

        for i,idx in enumerate(subset):
            # setting to 0 the mask of concepts I want supervision on
            self.mask[idx] = selected_concepts[i].float()
            #print(self.mask[idx])
            selected_concept = selected_concepts[i]
            #idx_range = list(self.concept_structure.values())[selected_concept]
            
            #print(selected_probs)
            #print(selected_stds)
            #print(median)
            #print(selected_concept, self.mask[idx])
            #print(probs[i, idx_range[0]:idx_range[1]+1])
            #logger.warning(f"was {torch.argmax(gt[self.pool[i]])} predicted {torch.argmax(preds[i])} with uncs {stds[i, idx_range[0]:idx_range[1]+1]}")
        logger.warning(f"fixed with {num} concepts in total")
        # Return indexes
        
        return [i for i in subset]
        
    def expected_cond_entropy_binary(self, embeddings, eps = 1e-8) -> torch.Tensor:
        samples = embeddings[self.pool]
        probs = self.probs(samples)
        print(probs)
        print(probs.shape)
        p = torch.clip(probs, eps, 1-eps)
        Hs = - (p * p.log() + (1-p) * (1-p).log())  # shape (S,)
        return torch.mean(Hs, dim=0).permute(1,0)

    def sample_entropy(self, embeddings, K = 100) -> list:
        # Evaluate input points from the pool
        search_space = embeddings[self.pool]
        
        # Select the indexes with highest entropy
        entropy_concepts = self.entropy(search_space)
        total_entropy = torch.sum(entropy_concepts, dim=1)
        values, indexes = torch.topk(total_entropy, K, sorted=True, largest = True)
        print(values)
        print(indexes)
        
        # Indexes are from 0 to n. embeddings, and refer to the embedding in the search space
        # So their number refers to the index in the pool to be used
        
        # Return indexes
        return [self.pool[i] for i in indexes]
    
    def sample_entropy_even(self, embeddings, K = 100) -> list:
        # Evaluate input points from the pool
        search_space = embeddings[self.pool]
        
        # Select the indexes with highest entropy
        entropy_concepts = self.entropy(search_space)
        total_entropy = torch.sum(entropy_concepts, dim=1)
        n_concepts = entropy_concepts.shape[1]
        training_indexes = []
        for i in range(n_concepts):
            values, indexes = torch.topk(entropy_concepts[:,i], math.floor(K/n_concepts), sorted=True, largest = True)
            print(torch.topk(entropy_concepts[:,i], math.floor(K), sorted=True, largest = True))
            for k in indexes:
                if k not in training_indexes:
                    training_indexes.append(k)
        
        # Indexes are from 0 to n. embeddings, and refer to the embedding in the search space
        # So their number refers to the index in the pool to be used
        
        '''
        # In the case there are less than K samples added, take the ones with highest entropy
        values, indexes = torch.topk(total_entropy, len(total_entropy), sorted=True, largest = True)
        for i,idx in enumerate(indexes):
            if len(training_indexes) >= K:
                break
            if idx not in training_indexes and len(training_indexes) < K:
                training_indexes.append(idx)
                logger.debug(f"Added sample with average entropy {values[i]/n_concepts}.")
        '''     
        # Return indexes
        return [self.pool[i] for i in training_indexes]
    
    def sample_combined_even(self, embeddings, K = 100) -> list:
        # Evaluate input points from the pool
        search_space = embeddings[self.pool]
        
        # Select the indexes with highest entropy
        entropy_concepts = self.entropy(search_space)
        total_entropy = torch.sum(entropy_concepts, dim=1)
        n_concepts = entropy_concepts.shape[1]
        training_indexes = []
        for i in range(n_concepts):
            values, indexes = torch.topk(entropy_concepts[:,i], math.floor(K/n_concepts), sorted=True, largest = True)
            for k in indexes:
                if k not in training_indexes:
                    training_indexes.append(k)
        # Indexes are from 0 to n. embeddings, and refer to the embedding in the search space
        # So their number refers to the index in the pool to be used
        
        # In the case there are less than K samples added, take the ones with highest entropy
        values, indexes = torch.topk(total_entropy, len(total_entropy), sorted=True, largest = True)
        for i,idx in enumerate(indexes):
            if len(training_indexes) >= K:
                break
            if idx not in training_indexes and len(training_indexes) < K:
                training_indexes.append(idx)
                logger.debug(f"Added sample with average entropy {values[i]/n_concepts}.")
     
        # Return indexes
        return [self.pool[i] for i in training_indexes]
    
    def sample_variance(self, embeddings, K = 100) -> list:
        # Evaluate input points from the pool
        search_space = embeddings[self.pool]
        
        # Select the indexes with highest entropy
        logits, vars = self.latent_variance(search_space)
        n_concepts = vars.shape[1]
        training_indexes = []
        for i in range(n_concepts):
            values, indexes = torch.topk(vars[:,i], math.floor(K/n_concepts), sorted=True, largest = True)
            for k in indexes:
                if k not in training_indexes:
                    training_indexes.append(k)
        
        # Return indexes
        return [self.pool[i] for i in training_indexes]
    
    def sample_BALD(self, embeddings, K=100, local = False):
        samples = embeddings[self.pool]
        entropy_concepts = self.entropy(samples)
        expected_conditional_entropy = self.expected_cond_entropy_binary(embeddings)
        n_concepts = entropy_concepts.shape[1]
        if local:
            training_indexes = []
            for i in range(n_concepts):
                local_predictive_entropy = entropy_concepts[:,i]
                local_expected_conditional_entropy = expected_conditional_entropy[:,i]
                # Compute BALD
                BALD = local_predictive_entropy - local_expected_conditional_entropy
                logger.warning(f"Predictive Entropy:{local_predictive_entropy} - Expected Entropy:{local_expected_conditional_entropy}")
                values, indexes = torch.topk(BALD, max(math.floor(K/n_concepts), 1), sorted=True, largest = True)
                for k in indexes:
                    if k not in training_indexes:
                        training_indexes.append(k)
        else:    
            global_predictive_entropy = torch.sum(entropy_concepts, dim=1)
            global_expected_conditional_entropy = torch.sum(expected_conditional_entropy, dim=1)
            
            BALD = global_predictive_entropy - global_expected_conditional_entropy
            
            values, training_indexes = torch.topk(BALD, K, sorted=True, largest = True)
        # Local Version
        '''
        mean = self.mean(samples)
        logits, vars = self.latent_variance(samples)
        n_concepts = vars.shape[1]
        if local:
            
        '''
        return [self.pool[i] for i in training_indexes]
    
    def sample_UCB_fixed(self, embeddings, K=100, alpha = 1):
        samples = embeddings[self.pool]
        mean = self.mean(samples)
        logits, vars = self.latent_variance(samples)
        n_concepts = vars.shape[1]
        training_indexes = []
        for i in range(n_concepts):
            c_variance = vars[:,i]
            c_mean = mean[:,i]
            # Compute UCB
            score = c_mean + alpha*c_variance
            logger.warning(f"Mean:{c_mean} - Variance:{c_variance}")
            values, indexes = torch.topk(score, max(math.floor(K/n_concepts), 1), sorted=True, largest = True)
            for k in indexes:
                if k not in training_indexes:
                    training_indexes.append(k)
        
        # If for some reasons the found samples are less than K, add some random
        if len(training_indexes) < K:
            logger.debug(f"Less, found only {len(training_indexes)}")
            random_samples = random.sample(range(len(samples)), len(samples))
            for i in random_samples:
                if i not in training_indexes:
                    training_indexes.append(i)
                if len(training_indexes) == K:
                    break
        # If for some other reasons the found samples are more than K, just select K randomly
        elif len(training_indexes) > K:
            logger.debug(f"More, found over {len(training_indexes)}")
            training_indexes = random.sample(training_indexes, K)
        
        return [self.pool[i] for i in training_indexes]

    def sample_UCB_GLOBAL(self, embeddings, K=100, alpha = 1):
        samples = embeddings[self.pool]
        mean = self.mean(samples)
        logits, vars = self.latent_variance(samples)
        n_concepts = vars.shape[1]
        training_indexes = []
        
        g_variance = torch.sum(vars, dim=1)
        g_mean = torch.sum(mean, dim=1)
        # Compute UCB
        score = g_mean + alpha*g_variance
        logger.warning(f"Mean:{g_mean} - Variance:{g_variance}")
        values, indexes = torch.topk(score, max(K, 1), sorted=True, largest = True)
        for k in indexes:
            if k not in training_indexes:
                training_indexes.append(k)
        return [self.pool[i] for i in training_indexes]
    
    def sample_UUCB(self, embeddings, K=100, alpha = 1, beta = 1):
        samples = embeddings[self.pool]
        mean = self.mean(samples)
        entropy_concepts = self.entropy(samples)
        logits, vars = self.latent_variance(samples)
        n_concepts = vars.shape[1]
        training_indexes = []
        for i in range(n_concepts):
            c_variance = vars[:,i]
            c_entropy = entropy_concepts[:,i]
            c_mean = mean[:,i]
            # Compute UCB
            score = c_mean + alpha*c_variance + beta*c_entropy
            logger.warning(f"Mean:{c_mean} - Variance:{c_variance} - Entropy:{c_entropy}")
            values, indexes = torch.topk(score, max(math.floor(K/n_concepts), 1), sorted=True, largest = True)
            for k in indexes:
                if k not in training_indexes:
                    training_indexes.append(k)
        return [self.pool[i] for i in training_indexes]
        
    def sample_clip(self, train_embeddings, K, CLIP_MODE = 'extremes'):
        #print(K,train_embeddings.shape)
        N_PER_CONCEPT = math.floor(K/self.num_concepts)
        #N_PER_CONCEPT = 2
        logger.debug(f"Adding {N_PER_CONCEPT} samples per concept")
        train_subset = []
        # Load Clip scores
        # For each concept, sort and choose samples from an uniform range
        with torch.no_grad():
            #clip_feats = (train_embeddings.norm(dim=-1, keepdim=True).float() @ text_embeddings.norm(dim=-1, keepdim=True).float().T)
            text_normed = self.text_embeddings/(self.text_embeddings.norm(p=2, dim=-1, keepdim = True).float())
            clip_normed = train_embeddings/(train_embeddings.norm(p=2, dim=-1, keepdim = True).float())
            clip_feats = train_embeddings @ self.text_embeddings.T

        # Limit the samples only to the one in the pool (remove the one already used for training)
        clip_feats = clip_feats[self.pool]
        
        for task in range(clip_feats.shape[1]):
            values, indexes = torch.sort(clip_feats[:,task], dim=0)
            #print(values)
            if CLIP_MODE == 'extremes':
                # Take the samples with lowest activation
                choosen = indexes[:N_PER_CONCEPT//2]
                for id in choosen:
                    #print("Added", indexes[id].item())
                    train_subset.append(indexes[id].item())
                    
                # Take samples with highest activation
                choosen = indexes[-N_PER_CONCEPT//2:]
                for id in choosen:
                    #print("Added", indexes[id].item())
                    train_subset.append(indexes[id].item())
            
            elif CLIP_MODE == 'uniform':
                choosen = torch.linspace(0, clip_feats.shape[1]-1, steps=N_PER_CONCEPT).long()
                for id in choosen.tolist():
                    train_subset.append(indexes[id].item())
            else:
                raise NotImplementedError()
                
        original_len = len(train_subset)
        train_subset = list(set(train_subset))
        removed = original_len - len(train_subset)
        logger.debug(f"Removed {removed} duplicates. Final size = {len(train_subset)}")
        # Add more samples if there are not enough due to duplicates
        if len(train_subset) < K:
            logger.warning(f"{len(train_subset)} is lower than {K} requested samples. Adding some more randomly.")
            some_random_samples = random.sample(range(len(self.pool)), len(self.pool))
            #print(self.pool)
            ii = 0
            while len(train_subset) < K:
                if some_random_samples[ii] not in train_subset:
                    train_subset.append(some_random_samples[ii])
                    #print("Added R", some_random_samples[ii])
                ii+=1
                   
        return [self.pool[i] for i in train_subset]
        
    def train_loop(self, embeddings, data, testx,testy, acq_fn = 'random', K = 100, variation_list = [''], **kwargs):
        if self.args.dataset in ['shapes3d', 'dermamnist']:
            multiclass = True
        else:
            multiclass = False
        # 1) Sample points
        if acq_fn == 'random':
            logger.debug(f"Added {K} random samples.")
            indexes = random.sample(self.pool, K)
        elif acq_fn == 'randomc':
            indexes = self.sample_random_c(embeddings, K)
        elif acq_fn == 'randomuni':
            indexes = self.sample_random_uni_c(embeddings, K)
        elif acq_fn == 'entropy' or acq_fn == 'combined':
            indexes = self.sample_entropy(embeddings, K=K)
        elif acq_fn == 'active1':
            indexes = self.sample_active1(embeddings, K=K)
        elif acq_fn == 'activec':
            indexes = self.sample_active_c(embeddings, K=K)
        elif acq_fn == 'activecc':
            indexes = self.sample_active_cc(embeddings, K=K)
        elif acq_fn == 'activeours':
            indexes = self.sample_active_ours(embeddings, K=K, multiclass=multiclass)
        elif acq_fn == 'shen':
            indexes = self.sample_shannon_entropy(embeddings, K=K)
        elif acq_fn == 'boh':
            indexes = self.sample_boh(embeddings, K=K)
        elif acq_fn == 'entropyeven':
            indexes = self.sample_entropy_even(embeddings, K=K)
        elif acq_fn == 'combinedeven':
            indexes = self.sample_combined_even(embeddings, K=K)
        elif acq_fn == 'variance':
            indexes = self.sample_variance(embeddings, K=K)
        elif acq_fn == 'bald':
            local = kwargs.get("local", False)
            indexes = self.sample_BALD(embeddings, local=local)
        elif acq_fn == 'ucb':
            alpha = kwargs.get("alpha", 1)
            if alpha != 1:
                logger.warning(f"Alpha set to {alpha}")
            indexes = self.sample_UCB(embeddings, alpha=alpha, K=K)
        elif acq_fn == 'ucbf':
            alpha = kwargs.get("alpha", 1)
            if alpha != 1:
                logger.warning(f"Alpha set to {alpha}")
            indexes = self.sample_UCB_fixed(embeddings, alpha=alpha, K=K)
        elif acq_fn == 'ucbg':
            alpha = kwargs.get("alpha", 1)
            if alpha != 1:
                logger.warning(f"Alpha set to {alpha}")
            indexes = self.sample_UCB_GLOBAL(embeddings, alpha=alpha, K=K)
        elif acq_fn == 'uucb':
            alpha = kwargs.get("alpha", 1)
            beta = kwargs.get("beta", 1)
            if alpha != 1:
                logger.warning(f"Alpha set to {alpha}")
            if beta != 1:
                logger.warning(f"Alpha set to {beta}")
            indexes = self.sample_UUCB(embeddings, alpha=alpha, beta=beta, K=K)
        elif acq_fn == 'clip':
            indexes = self.sample_clip(embeddings,K)
        elif acq_fn == 'copy':
            with open("./results/GPs/2025_09_24_23_08-SVGP_dermamnist_random_cos_7__LOGS.json", 'r') as f:
                original = json.load(f)
                for i,step in enumerate(original):
                    if step['training_pool_size'] == len(self.training_pool):
                        # print(i,step['training_pool_size'])
                        pool_to_copy = original[i+1]['training_pool']
                        indexes = []
                        for idx in pool_to_copy:
                            if idx not in list(self.training_pool):
                                indexes.append(idx)
                        # print("Added", len(indexes), " to copy asdasasd")
                        break
        else:
            raise NotImplementedError()
            indexes = random.sample(self.pool, K)
        
        # 2) Collect annotations
        # First update the training pool based on the acq fn
        for i in indexes:
            self.training_pool.append(i)
            self.pool.remove(i)
        
        # Create the training pool
        ## Retrieve labels (GT data)
        '''
        train_concept_labels = []
        for i in self.training_pool:
            concepts = data[i][1]
            train_concept_labels.append(concepts)
        train_concept_labels = torch.stack(train_concept_labels, dim=0).to(self.args.device)
        
        
        train_labels = []
        for i in self.training_pool:
            labels = data[i][2]
            train_labels.append(labels)
        train_labels = torch.stack(train_labels, dim=0).to(self.args.device)
        '''
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
        ## Retrive embeddings (input)
        train_embeddings = embeddings[self.training_pool].to(self.args.device)

        for aug in variation_list:
            if aug == '':
                continue
            logger.info(f'Adding {aug} variation')
            augmented_embs, augmented_labels = get_augmentations(self.args.dataset, indexes, aug, device=self.args.device)
            train_embeddings = torch.cat((train_embeddings,augmented_embs), dim=0)
            train_concept_labels = torch.cat((train_concept_labels,augmented_labels), dim=0)
                
        print(len(self.pool))
        print(len(self.training_pool))  

        self.reset_models()  
        
        if self.kernel in ['cosy', 'hadamard']:
            # 3) Train
            self.train(train_embeddings,train_concept_labels, patience=self.args.gp_patience, labels=train_labels)
            # 4) Evaluate
            out_dict = self.eval(testx.to(self.args.device), labels=train_labels)
        else:
            # 3) Train
            print(embeddings.shape)
            self.train(embeddings.to(self.device),train_concept_labels, patience=self.args.gp_patience, text_embeddings = kwargs.get("text_embeddings", None))
            # 4) Evaluate
            out_dict = self.eval(testx.to(self.args.device))
            
   
        
        pred_probs = out_dict['probs']
        pred_vars = out_dict['std']
        dict = self.evaluate_new(pred_probs.cpu(),testy.cpu())
        dict.update(self.sanitize_args(self.args))
        dict.update({'discretization':'threshold'})
        return dict