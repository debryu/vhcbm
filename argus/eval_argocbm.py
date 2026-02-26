from argus.models.argocbm import ArgoCBM
from argus.metrics.dci import DCI_wrapper
from CQA.datasets import GenericDataset
from CQA.utils.args_utils import save_args, load_args
import datetime
import argparse
from loguru import logger 
import sys
import importlib
import sys
import datetime
import setproctitle, socket, uuid
import json, os
import torch
from sklearn.metrics import classification_report
import numpy as np
from CQA.metrics.common import RocAUC,MacroAUC

def get_folder_name(args):
  ''' Naming convention for the saved model
  Available flags:
  '''
  return f"ARGOCBM_{args.dataset}_{args.date}_{args.time}_SEED={args.seed}"

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic flags based on initial flag value.")
    
    parser.add_argument('-logger', type=str, default="DEBUG", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("-load_dir", type=str, default=None, help="Folder where to save the model")
    # Parse known arguments to determine the value of --model
    args = parser.parse_args()
    
    ''' Set up logger'''
    logger.remove()
    def my_filter(record):
        return record["level"].no >= logger.level(args.logger).no
    logger.add(sys.stderr, filter=my_filter)
    ''' ------------------------------------- '''

    return args

def save_IM_as_img(save_path,name,title,importance_matrix,save_plot=True):
  dim1,dim2 = importance_matrix.shape
  heatmap(importance_matrix, plot_title=title, save_path=os.path.join(save_path,name))
  #visualise(save_path, name, importance_matrix, (dim1,dim2), title, save_plot=save_plot)
  
  return 

def heatmap(matrix, plot_title, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=matrix.shape)
    ax = sns.heatmap(matrix, annot=False, 
                fmt=".2f", 
                cmap="hot", 
                linewidths=0.5, 
                square=True, 
                
                cbar=True)
  
    plt.xticks(rotation=50, ha='right', va='top', )
    plt.title(plot_title, fontsize=80, pad=20)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def DCI(reprs, binary = False, train_test_ratio=0.7,max_samples:int = None, level = 'INFO'):
    n = len(reprs['concept_means'])
    indices = torch.randperm(n)
    train_size = int(n * train_test_ratio)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    representation_train = torch.zeros(train_size, reprs['concept_gts'].shape[1])
    representation_val = torch.zeros(reprs['concept_gts'].shape[0]-train_size, reprs['concept_gts'].shape[1])
        
    if binary:
        for i in range(representation_train.shape[1]):
            representation_train[:,i] = reprs['concept_means'][train_idx,2*i+1]
        
        for i in range(representation_val.shape[1]):
            representation_val[:,i] = reprs['concept_means'][val_idx,2*i+1]
    else:
        representation_train = reprs['concept_means'][train_idx]
        representation_val = reprs['concept_means'][val_idx]
            
    concept_gt_train = reprs['concept_gts'][train_idx]
    concept_gt_val = reprs['concept_gts'][val_idx]

    if max_samples != None:
      logger.warning(f"Cutting the max number of samples to {max_samples}!")
      representation_train = representation_train[:max_samples]
      concept_gt_train = concept_gt_train[:max_samples]
      representation_val = representation_val[:max_samples]
      concept_gt_val = concept_gt_val[:max_samples]

    logger.debug(f"Computing DCI with train_test_ratio={train_test_ratio}...")
    dci = DCI_wrapper(representation_train, concept_gt_train, representation_val, concept_gt_val, level)
    dci['train_test_ratio'] = train_test_ratio
    dci_dict = {
        'dci': dci,
        'disentanglement': dci['disentanglement'],
        'completeness': dci['completeness'],
        'importance_matrix': dci['importance_matrix']
    }
    
    return dci_dict


if __name__ == "__main__":
    args = parse_args()
    args_model = load_args(args)
    model = ArgoCBM(args_model)
    model.load()
    
    train_y = []
    data_train = GenericDataset(ds_name = args_model.dataset, split='train')
    for i in range(len(data_train)):
        train_y.append(data_train[i][2])
    train_y = torch.stack(train_y, dim=0)
    test_y = []
    data_test = GenericDataset(ds_name = args_model.dataset, split='test')
    for i in range(len(data_test)):
        test_y.append(data_test[i][2])
    test_y = torch.stack(test_y, dim=0)
    val_y = []
    data_val = GenericDataset(ds_name = args_model.dataset, split='val')
    for i in range(len(data_val)):
        val_y.append(data_val[i][2])
    val_y = torch.stack(val_y, dim=0)
    
    train = model.backbone.get_concept_preds('train')
    test = model.backbone.get_concept_preds('test')
    val = model.backbone.get_concept_preds('val')
    
    if val['concept_probs'].shape[1] != args_model.num_c:
        temp_binary = torch.zeros((val['concept_probs'].shape[0], args_model.num_c))
        for i in range(temp_binary.shape[1]):
            temp_binary[:,i] = val['concept_probs'][:,2*i+1]
        val['concept_probs'] = temp_binary
    if test['concept_probs'].shape[1] != args_model.num_c:
        temp_binary = torch.zeros((test['concept_probs'].shape[0], args_model.num_c))
        for i in range(temp_binary.shape[1]):
            temp_binary[:,i] = test['concept_probs'][:,2*i+1]
        test['concept_probs'] = temp_binary
    if train['concept_probs'].shape[1] != args_model.num_c:
        temp_binary = torch.zeros((train['concept_probs'].shape[0], args_model.num_c))
        for i in range(temp_binary.shape[1]):
            temp_binary[:,i] = train['concept_probs'][:,2*i+1]
        val['concept_probs'] = temp_binary
        
    #output = model.inference(torch.cat([test['concept_means'], test['concept_stds']**2], dim=1))
    output = model.inference(test['concept_means'])
    #import pdb; pdb.set_trace()
    
    concept_pred = []
    concept_gt = []
    concept_logits = []
    for i in range(test['concept_gts'].shape[1]):
        concept_pred.append(test['concept_preds'][:,i].numpy())
        concept_gt.append(test['concept_gts'][:,i].numpy()) 
        concept_logits.append(test['concept_means'][:,i])
        
    f1_raw = []
    acc_raw = []
    rec_raw = []
    prec_raw = []
    reports = []
    pr_aucs = []
    roc_aucs = []
    
    if args_model.dataset in ['shapes3d', 'dermamnist']:
        binary = False
    else:
        binary = True
    
    for i in range(test['concept_gts'].shape[1]):
        cr_concepts= classification_report(test['concept_gts'][:,i].numpy(), test['concept_preds'][:,i].numpy(), output_dict=True)
        f1_raw.append(cr_concepts['macro avg']['f1-score']) #type:ignore
        acc_raw.append(cr_concepts['accuracy'])  # type:ignore
        rec_raw.append(cr_concepts['macro avg']['recall'])   # type:ignore
        prec_raw.append(cr_concepts['macro avg']['precision'])   # type:ignore
        if binary:
            roc = RocAUC(test['concept_gts'][:,i].numpy(), test['concept_means'][:,2*i+1].numpy()) 
            pr = MacroAUC(test['concept_gts'][:,i].numpy(), test['concept_means'][:,2*i+1].numpy())
        else:
            roc = RocAUC(test['concept_gts'][:,i].numpy(), test['concept_means'][:,i].numpy())   
            pr = MacroAUC(test['concept_gts'][:,i].numpy(), test['concept_means'][:,i].numpy())
        pr_aucs.append(pr.prauc0)
        roc_aucs.append(roc.roc_auc)
        
    cr_labels = classification_report(test_y.cpu().numpy(), output['preds'].cpu().numpy(), output_dict=True)
    
    
    res = { 
            'f1_raw': np.mean(f1_raw),
            'all_f1_raw': f1_raw,
            'acc_raw': np.mean(acc_raw),
            'rec_raw': np.mean(rec_raw),
            'prec_raw':np.mean(prec_raw),
            'reports':reports,
            'pr_auc':np.mean(pr_aucs),
            'roc_auc':np.mean(roc_aucs),
            'all_roc_aucs': roc_aucs,
            'all_pr_aucs': pr_aucs,
            'label_f1': cr_labels['macro avg']['f1-score']  #type:ignore
            }
    
    dci_dict = DCI(test, binary = binary)
    res.update(dci_dict)
    save_IM_as_img(save_path = args.load_dir, 
                   name="dci.png", 
                   title="dci matrix", 
                   importance_matrix=dci_dict['importance_matrix'])
    
    #import pdb; pdb.set_trace()
    metrics = ['disentanglement','f1_raw','pr_auc','roc_auc', 'label_f1']
    serializable_dict = {}
    for key in metrics:
      try:
        serializable_dict[key] = float(res[key])
      except:
        pass
    with open(os.path.join(args.load_dir, "metrics.txt"), "w") as f:
      json.dump(serializable_dict, f, indent=2)
    print('\n ### Closing ###')