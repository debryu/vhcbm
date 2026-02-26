from sklearn.metrics import classification_report
from loguru import logger
from CQA.config import CONCEPT_SETS
from CQA.utils.utils import get_concept_names
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, auc, roc_auc_score
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch
from dataclasses import dataclass
'''
Current output:
out_dict = {
        "concepts_gt": annotations,
        "concepts_pred": concepts,
        "labels_gt": labels,
        "labels_pred": preds,
        "accuracy": acc_mean / len(loader.dataset)
      }
'''

def auc_roc(X,y, model_args):
  logger.debug("auc_roc function")
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=model_args.seed)
  classifier = make_pipeline(StandardScaler(), LinearSVC(random_state=model_args.seed))
  classifier.fit(X_train, y_train)
  display = PrecisionRecallDisplay.from_estimator(
    classifier, X_test, y_test,name='LINEAR SVC', plot_chance_level=True
  )
  _ = display.ax_.set_title(f'{model_args.dataset} AUC-ROC')
  y_preds = classifier.decision_function(X_test)
  # Compute precision-recall curve
  precision, recall, _ = precision_recall_curve(y_test, y_preds)

  # Compute PR AUC
  pr_auc = auc(recall, precision)
  logger.info(f"PR AUC: {pr_auc}")
  #plt.show()
  return pr_auc

def compute_AUCROC_concepts(output,args):
    logger.debug("Computing AUC-ROC")
    conc_pred = output['concepts_pred']
    conc_gt = output['concepts_gt']

    if not hasattr(args, 'num_c'):
      args.num_c = conc_pred.shape[1]
    
    auc_rocs = []
    for i in tqdm(range(args.num_c), desc="Computing AUC-ROC"):
      logger.info(f"Computing AUC-ROC for concept {i}")
      X = conc_pred[:,i].detach().cpu().numpy().reshape(-1,1)
      y = conc_gt[:,i].detach().cpu().numpy()
      auc_rocs.append(auc_roc(X,y, args))
    
    auc_dict = {'avg_concept_auc':np.mean(auc_rocs), 'concept_auc': auc_rocs}
    return auc_dict
  
@dataclass
class RocAUCResult:
    roc_auc: float

def RocAUC(concept_labels: np.ndarray, predicted_concept: np.ndarray) -> RocAUCResult:
    # Standard ROC-AUC (1 is positive)
    roc_auc = roc_auc_score(concept_labels, predicted_concept)
    return RocAUCResult(roc_auc=float(roc_auc))
     
@dataclass
class MacroAUCResult:
    macro_auc: float
    prauc0: float
    prauc1: float
    min_auc: float

def MacroAUC(concept_labels:np.ndarray,predicted_concept:np.ndarray) -> MacroAUCResult:
    """
    Compute macro-averaged AUC scores for predicted concepts.

    Parameters
    ----------
    concept_labels : array-like of shape (n_samples)
        Ground truth binary labels for the concept. 
    predicted_concept : array-like of shape (n_samples)
        Predicted scores or probabilities for the concept.

    Returns
    -------
    results : MacroAUCResult
        Dictionary containing evaluation metrics with the following keys:
        
        - "macro_auc" : float
            Armonic mean AUC score across all concepts.
        - "prauc0" : dict[str, float]
            AUC-PR considering the 0 of the concept as negative sample (and 1 as positive).
        - "prauc1" : int
            AUC-PR considering the 1 of the concept as negative sample (and 0 as positive).

    Examples
    --------
    >>> y_true = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]
    >>> y_pred = [[0.9, 0.2, 0.8], [0.1, 0.7, 0.4], [0.8, 0.6, 0.9]]
    >>> MacroAUC(y_true, y_pred)
    {'macro_auc': 0.92,
     'per_concept_auc': {0: 0.95, 1: 0.90, 2: 0.91},
     'n_concepts': 3,
     'n_samples': 3}
    """
    precision, recall, thresholds = precision_recall_curve(concept_labels, predicted_concept)
    other_class_pr, other_class_rec, other_class_thr = precision_recall_curve(1-concept_labels, -predicted_concept)
    inverted_pr_auc = auc(other_class_rec, other_class_pr)
    pr_auc = auc(recall, precision)   
    MacroAUC = (inverted_pr_auc + pr_auc)/2
    return MacroAUCResult(macro_auc=float(MacroAUC), prauc0=float(pr_auc), prauc1=float(inverted_pr_auc), min_auc=min(float(pr_auc), float(inverted_pr_auc)))
     
def get_conceptWise_metrics(output, model_args, main_args, threshold, name = '', dict_str='concepts_pred'):
    if main_args.wandb:
        import wandb
    ds = model_args.dataset.split("_")[0]
    concept_preds = output[dict_str]
    concept_gt = output['concepts_gt']
    # Should be already on cpu but just in case
    concept_pred = concept_preds.cpu()
    concept_gt = concept_gt.cpu()

    # Setting concepts to 1 if the value is above the threshold, 0 otherwise
    concept_pred = (torch.nn.functional.sigmoid(concept_pred) > threshold).float()
    logger.debug(f"Number of concetps: {concept_preds.shape[1]}")
    
     #print(concept_pred.T.shape)
    #print(concept_gt.T.shape)
    print(concept_pred.shape)
    accuracy = (concept_pred == concept_gt).sum(dim=0) / concept_gt.shape[0]
    #print(accuracy)
    concept_names = get_concept_names(CONCEPT_SETS[ds])
    concept_pred_list = []
    concept_gt_list = []
    for i in range(concept_gt.shape[1]):
        concept_pred_list.append(concept_pred[:,i].numpy())
        concept_gt_list.append(concept_gt[:,i].numpy())
    
    concept_accuracies = []
    concept_f1 = []
    classification_reports = []
    for i in range(len(concept_pred_list)):
        print(f"Concept {i}: {concept_names[i]}")
        tn = [f"No {concept_names[i]}",f"{concept_names[i]}"]
        cr = classification_report(concept_gt_list[i], concept_pred_list[i], target_names=tn, output_dict=True)
        classification_reports.append(cr)
        concept_f1.append(cr['macro avg']['f1-score'])  # type:ignore
        concept_accuracies.append(cr['accuracy'])       # type:ignore
        #print(classification_report(concept_gt_list[i], concept_pred_list[i], target_names=tn))
        #if main_args.wandb:
        #    print("logging",{f"concept_accuracy":cr['accuracy'], "manual_step":i})
        #   wandb.log({f"concept_accuracy":cr['accuracy'], "manual_step":i})
   
    return {f'{name}avg_concept_accuracy': sum(concept_accuracies)/len(concept_accuracies), 
            f'{name}concept_accuracy':concept_accuracies, 
            f'{name}concept_classification_reports':classification_reports,
            f'{name}avg_concept_f1': sum(concept_f1)/len(concept_f1),
            f'{name}concept_f1':concept_f1}

def get_metrics(output, requested:list[str]):
  metrics = []
  for metric in requested:
    if metric == 'classification_report':
      metrics.append(classification_report)
  return metrics