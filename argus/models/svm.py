
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
import torch
from loguru import logger
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt
from typing import Optional, Any
from CQA.datasets import GenericDataset
from argus.metrics.auc import MacroAUC, RocAUC
class Base():
    def __init__(self):
        self.n_concepts: Any = None
        self.model: Any = None
        
    def evaluate(self, embeddings:torch.Tensor, dataset:GenericDataset) -> dict:
        """
        Evaluate the model on given data.

        Parameters
        ----------
        embeddings : torch.Tensor
            (n_samples,embedding_dim) tensor containing the embeddings for all samples
        dataset : GenericDataset
            Dataset class 

        Returns
        -------
        dict
            Evaluation results (e.g., metrics like accuracy, AUC, etc.).
        """
        f1s = []
        min_pr_aucs = []
        pr_aucs = []
        roc_aucs = []
        brier_skill_scores = []
        for c_id in range(self.n_concepts):
            concept_labels = []
            # Collect train labels
            for i in range(len(dataset)):
                concept_labels.append(dataset[i][1][c_id])
            concept_labels = torch.stack(concept_labels, dim=0).detach().cpu().numpy()
            # Final shape will be (1, train_size)
            
            probs = self.model[c_id].predict_proba(X_test)
            
            bs = brier_score_loss(concept_labels, probs)
            prior_prob = np.mean(concept_labels)
            reference_probs = np.full_like(probs, prior_prob)
            bs_ref = brier_score_loss(concept_labels, reference_probs)
            if bs_ref == 0:
                bss = 0.0 if bs == 0 else -np.inf
            else:
                bss = 1 - (bs / bs_ref)
            brier_skill_scores.append(bss)
            
            
            input = embeddings.detach().cpu().numpy()
            predicted_concept = self.model[c_id].predict(input)
            logits = self.model[c_id].decision_function(input)
            res1 = MacroAUC(concept_labels, logits)
            res2 = RocAUC(concept_labels, logits)
            min_pr_aucs.append(res1.min_auc)
            pr_aucs.append(res1.prauc0)
            roc_aucs.append(res2.roc_auc)
            '''
            precision, recall, thresholds = precision_recall_curve(concept_labels, logits)
            pr_auc = auc(recall, precision)
            
            plt.figure()
            plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            plt.legend(loc="lower left")

            # Save the figure
            plt.savefig(f"./results/images/pr_auc_svm/linC{c_id}.png", dpi=200)  # you can change filename/format
            plt.close()  # close the figure if running in a loop
            '''
            #logger.debug(f"Testing SVM for concept {c_id}. PR-AUC={res.macro_auc}")
            #print(classification_report(concept_labels, predicted_concept))
            f1s.append(classification_report(concept_labels, predicted_concept, output_dict=True)['macro avg']['f1-score']) # type: ignore 
        #logger.info(np.mean(aucs))
        return {
            'bss': np.mean(bss_scores),
            'roc_auc': np.mean(roc_aucs),
            'pr_auc': np.mean(pr_aucs),
            'min_pr_auc': np.mean(min_pr_aucs),
            'all_roc_aucs': roc_aucs,
            'all_pr_aucs': pr_aucs,
            'all_min_pr_aucs': min_pr_aucs,
            'f1_cal': np.mean(f1s),
            'all_f1': f1s
        }


class SVM_Linear(Base):
    def __init__(self, n_concepts, seed = 42):
        self.n_concepts = n_concepts
        self.model = []
        for i in range(n_concepts):
            self.model.append(LinearSVC(random_state=seed))
        
    def train(self, embeddings, dataset):
        for c_id in range(self.n_concepts):
            logger.debug(f"Training SVM for concept {c_id}")
            train_concept_labels = []
            # Collect train labels
            for i in range(len(dataset)):
                train_concept_labels.append(dataset[i][1][c_id])
            train_concept_labels = torch.stack(train_concept_labels, dim=0).detach().cpu().numpy()
            # Final shape will be (1, train_size)
            
            train_input = embeddings.detach().cpu().numpy()
            try:
                self.model[c_id].fit(train_input, train_concept_labels)
            except ValueError as e:
                if str(e).startswith("This solver needs samples of at least 2"):
                    number = str(e).split(": ")[-1]
                    try:
                        float_number = float(number)
                    except:
                        float_number = float(number.split("(")[-1].split(")")[0])
                    missing_class = np.array([1-float_number])
                    
                    logger.error(f"Zero examples of class {missing_class}. Adding one randomly.")
                    #print(missing_class.shape)
                    temporary_labels = np.concatenate([train_concept_labels, missing_class], axis=0)
                    rand = np.random.rand(1,train_input.shape[1])
                    temporary_input = np.concatenate([train_input, rand], axis=0)
                    #print(temporary_labels.shape)
                    #print(temporary_input.shape)
                    self.model[c_id].fit(temporary_input, temporary_labels)
                    #print(train_input.shape)
                    #print(train_concept_labels.shape)
                else:
                    raise
            except Exception:
                raise
                


class SVM(Base):
    def __init__(self, n_concepts, seed = 42):
        self.n_concepts = n_concepts
        self.model = []
        for i in range(n_concepts):
            self.model.append(SVC(random_state=seed))
        
    def train(self, embeddings, dataset):
        for c_id in range(self.n_concepts):
            #logger.debug(f"Training SVM for concept {c_id}")
            train_concept_labels = []
            # Collect train labels
            for i in range(len(dataset)):
                train_concept_labels.append(dataset[i][1][c_id])
            train_concept_labels = torch.stack(train_concept_labels, dim=0).detach().cpu().numpy()
            # Final shape will be (1, train_size)
            
            train_input = embeddings.detach().cpu().numpy()
            try:
                self.model[c_id].fit(train_input, train_concept_labels)
            except ValueError as e:
                if str(e).startswith("This solver needs samples of at least 2"):
                    missing_class = np.array([1-float(str(e).split(": ")[-1])])
                    logger.error(f"Zero examples of class {missing_class}. Adding one randomly.")
                    #print(missing_class.shape)
                    temporary_labels = np.concatenate([train_concept_labels, missing_class], axis=0)
                    rand = np.random.rand(1,train_input.shape[1])
                    temporary_input = np.concatenate([train_input, rand], axis=0)
                    #print(temporary_labels.shape)
                    #print(temporary_input.shape)
                    self.model[c_id].fit(temporary_input, temporary_labels)
                    #print(train_input.shape)
                    #print(train_concept_labels.shape)
                elif str(e).endswith("got 1 class"):
                    unique = np.unique(train_concept_labels)
                    if unique.size == 1:
                        missing_class = np.array([1-unique[0]])
                    temporary_labels = np.concatenate([train_concept_labels, missing_class], axis=0)
                    rand = np.random.rand(1,train_input.shape[1])
                    temporary_input = np.concatenate([train_input, rand], axis=0)
                    self.model[c_id].fit(temporary_input, temporary_labels)
                else:
                    raise
            except Exception:
                raise
    
class SVM_labelClassification(Base):
    def __init__(self, n_classes, seed = 42):
        self.n_classes = n_classes
        self.model = []
        self.model = SVC(random_state=seed, class_weight="balanced")
        
    def train(self, embeddings, dataset):
        logger.debug(f"Training SVM for label")
        train_labels = []
        # Collect train labels
        for i in range(len(dataset)):
            train_labels.append(dataset[i][2])
        train_labels = torch.stack(train_labels, dim=0).detach().cpu().numpy()
        # Final shape will be (1, train_size)
        train_input = embeddings.detach().cpu().numpy()
        self.model.fit(train_input, train_labels)
            
    def predict(self, embeddings):
        """
        Predicts the class label for each sample.
        embeddings: torch.Tensor of shape (num_samples, embedding_dim)
        Returns: np.ndarray of shape (num_samples,) with predicted class index
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        # Collect decision function scores from each SVM
        scores = np.zeros((embeddings_np.shape[0], self.n_classes))
        predicted_classes = self.model.predict(embeddings_np)
        return predicted_classes
    
    def evaluate_classification(self, embeddings, dataset):
        from sklearn.metrics import classification_report
        predictions = self.predict(embeddings)
        # Collect train labels
        labels = []
        for i in range(len(dataset)):
            labels.append(dataset[i][2])
        labels = torch.stack(labels, dim=0).detach().cpu().numpy()
        print(classification_report(labels, predictions))
        
    def evaluate(self, embeddings:torch.Tensor, dataset:GenericDataset) -> dict:
        """
        Evaluate the model on given data.

        Parameters
        ----------
        embeddings : torch.Tensor
            (n_samples,embedding_dim) tensor containing the embeddings for all samples
        dataset : GenericDataset
            Dataset class 

        Returns
        -------
        dict
            Evaluation results (e.g., metrics like accuracy, AUC, etc.).
        """
        aucs = []
        f1s = []
        min_auc = []
        for c_id in range(self.n_classes):
            concept_labels = []
            # Collect train labels
            for i in range(len(dataset)):
                concept_labels.append(dataset[i][2])
            concept_labels = torch.stack(concept_labels, dim=0).detach().cpu().numpy()
            # Final shape will be (1, train_size)
            
            input = embeddings.detach().cpu().numpy()
            predicted_concept = self.model[c_id].predict(input)
            logits = self.model[c_id].decision_function(input)
            res = MacroAUC(concept_labels, logits)
            
            min_auc.append(min(res.prauc0,res.prauc1))
            '''
            precision, recall, thresholds = precision_recall_curve(concept_labels, logits)
            pr_auc = auc(recall, precision)
            
            plt.figure()
            plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            plt.legend(loc="lower left")

            # Save the figure
            plt.savefig(f"./results/images/pr_auc_svm/linC{c_id}.png", dpi=200)  # you can change filename/format
            plt.close()  # close the figure if running in a loop
            '''
            aucs.append(res.macro_auc)
            logger.debug(f"Testing SVM for concept {c_id}. PR-AUC={res.macro_auc}")
            print(classification_report(concept_labels, predicted_concept))
            f1s.append(classification_report(concept_labels, predicted_concept, output_dict=True)['macro avg']['f1-score']) # type: ignore 
        logger.info(np.mean(aucs))
        return {'auc':np.mean(aucs), 'f1':np.mean(f1s), 'minauc':np.mean(min_auc)}
    
def donothing():
    # Example: embeddings as input features (replace with your real embeddings)
    # Suppose you have 100 samples, each embedding is a 300-dim vector
    X = np.random.rand(100, 300)   # embeddings
    y = np.random.randint(0, 2, 100)  # binary labels (0 or 1)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define Linear SVM
    model = SVC(kernel='linear', probability=True, random_state=42)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # If you need probability scores instead of just labels:
    y_proba = model.predict_proba(X_test)[:, 1]
    print("Example predicted probabilities:", y_proba[:5])
