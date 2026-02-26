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
    train = model.backbone.get_concept_preds('train')
    test = model.backbone.get_concept_preds('test')
    val = model.backbone.get_concept_preds('val')
    
    import pandas as pd
    import numpy as np
    
    probs = train['concept_probs'].numpy()
    preds = train['concept_preds'].numpy()
    means = train['concept_means'].numpy()
    stds  = train['concept_stds'].numpy()
    gts   = train['concept_gts'].numpy()

    num_MC_samples = 30
    num_samples = train['concept_means'].shape[0]
 
    X = torch.zeros((num_samples*num_MC_samples, train['concept_means'].shape[1]))
    for i in range(num_MC_samples):
        start = i * num_samples
        end = (i + 1) * num_samples
        
        # Generate noisy samples
        X[start:end, :] = train['concept_means'] + train['concept_stds'] * torch.randn_like(train['concept_means'])
    
    Y = train_y.repeat(num_MC_samples, 1).flatten()
     
    model.train(X, val['concept_means'], test['concept_means'], Y, val_y, test_y)
    save_args(args)
    end_time = datetime.datetime.now()
    print('\n ### Total time taken: ', end_time - start_time)
    print('\n ### Closing ###')