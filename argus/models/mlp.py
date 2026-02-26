import numpy as np
import torch
from loguru import logger
from typing import Any, List
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, brier_score_loss
from CQA.datasets import GenericDataset
from argus.metrics.auc import MacroAUC, RocAUC

class Base():
    def __init__(self):
        self.n_concepts: Any = None
        self.model: List[Any] = []

    def evaluate(self, embeddings: torch.Tensor, dataset: GenericDataset) -> dict:
        f1s, min_pr_aucs, pr_aucs, roc_aucs, brier_skill_scores = [], [], [], [], []
        
        input_np = embeddings.detach().cpu().numpy()
        
        # Optimization: Pre-extract all labels once
        # Assuming dataset[i][1] returns a tensor of all concept labels for sample i
        all_labels = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0).cpu().numpy()

        for c_id in range(self.n_concepts):
            concept_labels = all_labels[:, c_id]
            model = self.model[c_id]

            # Handle concepts that were not trained (single-class case)
            if not getattr(model, "fitted_", False):
                logger.warning(f"Concept {c_id} not fitted. Using baseline predictions.")
                val = concept_labels[0]
                probs = np.full(len(input_np), float(val))
                predicted_concept = np.full(len(input_np), val)
                # Set a high/low logit for AUC calculation
                logits = np.full(len(input_np), 10.0 if val == 1 else -10.0)
            else:
                predicted_concept = model.predict(input_np)
                probs = model.predict_proba(input_np)[:, 1]
                # Logit calculation: ln(p / (1-p))
                logits = np.log(probs + 1e-8) - np.log(1 - probs + 1e-8)

            # Brier Skill Score (BSS)
            bs = brier_score_loss(concept_labels, probs)
            prior_prob = np.mean(concept_labels)
            bs_ref = brier_score_loss(concept_labels, np.full_like(probs, prior_prob))
            
            # If the reference score is 0 (dataset has only one class), BSS is 0
            bss = 1 - (bs / bs_ref) if bs_ref > 0 else 0.0
            brier_skill_scores.append(bss)

            # AUC Metrics
            res1 = MacroAUC(concept_labels, logits)
            res2 = RocAUC(concept_labels, logits)

            min_pr_aucs.append(res1.min_auc)
            pr_aucs.append(res1.prauc0)
            roc_aucs.append(res2.roc_auc)

            # F1 Score (Macro)
            f1s.append(
                classification_report(
                    concept_labels,
                    predicted_concept,
                    output_dict=True,
                    zero_division=0
                )['macro avg']['f1-score']
            )

        return {
            'bss': np.mean(brier_skill_scores),
            'roc_auc': np.mean(roc_aucs),
            'pr_auc': np.mean(pr_aucs),
            'min_pr_auc': np.mean(min_pr_aucs),
            'all_bss': brier_skill_scores,
            'all_roc_aucs': roc_aucs,
            'all_pr_aucs': pr_aucs,
            'all_min_pr_aucs': min_pr_aucs,
            'f1_cal': np.mean(f1s),
            'all_f1': f1s
        }

class MLPProbe(Base):
    def __init__(self, n_concepts: int, seed: int = 42, hidden_dim: int = 64):
        super().__init__()
        self.n_concepts = n_concepts
        self.model = [
            MLPClassifier(
                hidden_layer_sizes=(hidden_dim,),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                max_iter=500,
                early_stopping=False,
                n_iter_no_change=10,
                random_state=seed
            ) for _ in range(n_concepts)
        ]

    def train(self, embeddings: torch.Tensor, dataset: GenericDataset):
        train_input = embeddings.detach().cpu().numpy()
        all_labels = torch.stack([dataset[i][1] for i in range(len(dataset))], dim=0).cpu().numpy()

        for c_id in range(self.n_concepts):
            logger.debug(f"Training MLP for concept {c_id}")
            y = all_labels[:, c_id]

            if len(np.unique(y)) < 2:
                logger.error(f"Concept {c_id} has only one class. Skipping fit to avoid bias.")
                self.model[c_id].fitted_ = False
                continue

            self.model[c_id].fit(train_input, y)
            self.model[c_id].fitted_ = True