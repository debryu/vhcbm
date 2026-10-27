from argus.models.argocbm import ArgoCBM
from CQA.datasets import GenericDataset
from CQA.utils.args_utils import save_args
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
from argus.metrics.ece import plot_reliability_diagram, plot_im_reliability_diagram
def get_folder_name(args):
  ''' Naming convention for the saved model
  Available flags:
  '''
  return f"ARGOCBM_{args.dataset}_{args.date}_{args.time}_SEED={args.seed}"

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic flags based on initial flag value.")
    
    parser.add_argument('-logger', type=str, default="DEBUG", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("-dataset", type=str, default="shapes3d", help="Dataset to use")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-device", type=str, default="cuda", help="Device to use")
    parser.add_argument("-save_dir", type=str, default=None, help="Folder where to save the model")
    parser.add_argument("-wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("-seed", type=int, default=42, help="Set the random seed")
    #parser.add_argument("-model_path", type=str, default="shapes3d", help="Path to ARGO models")
    parser.add_argument("-resume", type=str, default=None, help="Path to a model to resume training")
    parser.add_argument("-model_path", type=str, default=None, help="Where the ARGO model is saved")
    # Parse known arguments to determine the value of --model
    args = parser.parse_args()
    
    ''' Set up logger'''
    logger.remove()
    def my_filter(record):
        return record["level"].no >= logger.level(args.logger).no
    logger.add(sys.stderr, filter=my_filter)
    ''' ------------------------------------- '''

    args.time = datetime.datetime.now().strftime("%H_%M")
    args.date = datetime.datetime.now().strftime("%Y_%m_%d")
    args.conf_host = socket.gethostname()
    args.conf_jobnum = str(uuid.uuid4())
    # set job name
    setproctitle.setproctitle('{}_{}'.format(args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    if args.dataset == 'cub':
        args.num_c = 112
        args.num_classes = 200
        args.concept_structure = {}
        for i in range(112):
            args.concept_structure[f'{i}'] = (i,i)
    elif args.dataset == 'celeba':
        args.num_c = 39
        args.num_classes = 2
        args.concept_structure = {   '1':(0,0),
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
    elif args.dataset == 'shapes3d':
        args.num_c = 42
        args.num_classes = 2
        args.concept_structure = {   'floor color':(0,9),
                        'wall color':(10,19),
                        'object color':(20,29),
                        'object size':(30,37),
                        'shape':(38,41),
                        }
    elif args.dataset == 'dermamnist':
        args.num_c = 7
        args.num_classes = 2
        args.concept_structure = {   'finding':(0,6),
                        }
    args.device = 'cuda'
    return args

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    args = parse_args()
    #args.model_path = "cub-220-randomc-RBF-686_2026_02_19_21_31_SEED=686.pt"
    #args.model_path = "celeba-220-randomc-RBF-868_2026_02_19_20_13_SEED=868.pt"
    #args.model_path = "cub-220-randomc-RBF-123_2026_02_21_19_34_SEED=123.pt"
    
    model = ArgoCBM(args)
    train_y = []
    data_train = GenericDataset(ds_name = args.dataset, split='train')
    for i in range(len(data_train)):
        train_y.append(data_train[i][2])
    train_y = torch.stack(train_y, dim=0)
    test_y = []
    data_test = GenericDataset(ds_name = args.dataset, split='test')
    for i in range(len(data_test)):
        test_y.append(data_test[i][2])
    test_y = torch.stack(test_y, dim=0)
    val_y = []
    data_val = GenericDataset(ds_name = args.dataset, split='val')
    for i in range(len(data_val)):
        val_y.append(data_val[i][2])
    val_y = torch.stack(val_y, dim=0)
    
    folder_name = args.model_path.replace(".pt","")
    args.save_dir = os.path.join(args.save_dir, folder_name)
    os.makedirs(args.save_dir, exist_ok=True)
    #train = model.backbone.get_concept_preds('train')
    test = model.backbone.get_concept_preds('test')
    #val = model.backbone.get_concept_preds('val')
    
    import pandas as pd
    import numpy as np
    
    #probs = train['concept_probs'].numpy()
    #preds = train['concept_preds'].numpy()
    #means = train['concept_means'].numpy()
    #stds  = train['concept_stds'].numpy()
    #gts   = train['concept_gts'].numpy()
    

    #import pdb; pdb.set_trace()
    
    '''N, C = preds.shape
    df_train = pd.DataFrame(
        data=np.concatenate([preds, means, stds, gts, train_y.unsqueeze(-1)], axis=1),
        columns=(
            #[f"probs_{i}" for i in range(probs.shape[1])] +
            [f"pred_{i}" for i in range(preds.shape[1])] +
            [f"mean_{i}" for i in range(means.shape[1])] +
            [f"std_{i}"  for i in range(stds.shape[1])] +
            [f"gt_concept_{i}"   for i in range(gts.shape[1])] +
            [f"gt_label" ]
        )
    )
    probs = train['concept_probs'].numpy()
    preds = test['concept_preds'].numpy()
    means = test['concept_means'].numpy()
    stds  = test['concept_stds'].numpy()
    gts   = test['concept_gts'].numpy()
    N, C = gts.shape
    df_test = pd.DataFrame(
        data=np.concatenate([preds, means, stds, gts, test_y.unsqueeze(-1)], axis=1),
        columns=(
            #[f"probs_{i}" for i in range(probs.shape[1])] +
            [f"pred_{i}" for i in range(preds.shape[1])] +
            [f"mean_{i}" for i in range(means.shape[1])] +
            [f"std_{i}"  for i in range(stds.shape[1])] +
            [f"gt_concept_{i}"   for i in range(gts.shape[1])] +
            [f"gt_label" ]
        )
    )
    probs = train['concept_probs'].numpy()
    preds = val['concept_preds'].numpy()
    means = val['concept_means'].numpy()
    stds  = val['concept_stds'].numpy()
    gts   = val['concept_gts'].numpy()
    N, C = preds.shape
    df_val = pd.DataFrame(
        data=np.concatenate([preds, means, stds, gts, val_y.unsqueeze(-1)], axis=1),
        columns=(
            #[f"probs_{i}" for i in range(probs.shape[1])] +
            [f"pred_{i}" for i in range(preds.shape[1])] +
            [f"mean_{i}" for i in range(means.shape[1])] +
            [f"std_{i}"  for i in range(stds.shape[1])] +
            [f"gt_concept_{i}"   for i in range(gts.shape[1])] +
            [f"gt_label" ]
        )
    )
    df_train.to_csv("./figs/cub-random-train.csv", index=False)
    df_test.to_csv("./figs/cub-random-test.csv", index=False)
    df_val.to_csv("./figs/cub-random-val.csv", index=False)
    '''

    num_MC_samples = 30
    num_samples = test['concept_means'].shape[0]
 
    X = torch.zeros((num_samples*num_MC_samples, test['concept_means'].shape[1]))
    for i in range(num_MC_samples):
        start = i * num_samples
        end = (i + 1) * num_samples
        
        # Generate noisy samples
        X[start:end, :] = test['concept_means'] + test['concept_stds'] * torch.randn_like(test['concept_means'])
    
    # 30 samples of the same image, then 30 samples of the next image, etc.

    #Y = train_y.repeat(num_MC_samples, 1).flatten()
     
    if X.shape[1] != args.num_c:
        X_binary = torch.zeros((num_samples*num_MC_samples, args.num_c))
        for i in range(X_binary.shape[1]):
            X_binary[:,i] = torch.nn.functional.softmax(X[:,2*i:2*i+2], dim=1)[:,1]
        X = X_binary
    else:
        for task_id, (task_name, idx_range) in enumerate(args.concept_structure.items()):
            num_classes = len(range(idx_range[0],idx_range[1]+1))
            X[:,idx_range[0]:idx_range[1]+1] = torch.softmax(X[:, idx_range[0]:idx_range[1]+1], dim=1)   

    # Reshape the tensor in [number of samples, number of MC samples, number of concepts]
    #     
    # Current shape: (num_MC_samples * num_samples, num_concepts)
    # 1. Reshape to (num_MC_samples, num_samples, num_concepts)
    # 2. Permute/Transpose to (num_samples, num_MC_samples, num_concepts)
    X = X.view(num_MC_samples, num_samples, -1).permute(1, 0, 2)
    #print(X.shape)
    #print(X[0,:,24])
    preds = torch.mean(X, dim=1)
    print(preds.shape)
    
    model = ArgoCBM(args)
    train_y = []
    data_train = GenericDataset(ds_name = args.dataset, split='train')
    for i in range(len(data_train)):
        train_y.append(data_train[i][2])
    train_y = torch.stack(train_y, dim=0)
    test_y = []
    data_test = GenericDataset(ds_name = args.dataset, split='test')
    for i in range(len(data_test)):
        test_y.append(data_test[i][2])
    test_y = torch.stack(test_y, dim=0)
    val_y = []
    data_val = GenericDataset(ds_name = args.dataset, split='val')
    for i in range(len(data_val)):
        val_y.append(data_val[i][2])
    val_y = torch.stack(val_y, dim=0)

    import pdb; pdb.set_trace()
    model.train(X, val['concept_means'], test['concept_means'], Y, val_y, test_y)
    