from CQA.models.cbmlite import CBMLITE
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
    parser.add_argument("-batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
    parser.add_argument("-backbone", type=str, default="resnet18", help="Which ResNet pretrained model to use as backbone")
    parser.add_argument("-unfreeze", type=int, default=0, help="Number of conv layers to unfreeze from the pretrained model")
    parser.add_argument("-num_c", type=int, default=64, help="Number of concepts to learn when unsupervised")
    # Training
    parser.add_argument("-num_workers",type=int,default=4,help="Number of workers used for loading data")
    parser.add_argument("-predictor", type=str, default="saga", help="Which linear predictor to use", choices=['saga', 'svm'])
    parser.add_argument("-c_svm", type=float, default=1, help="C hyperparameter for SVM")
    parser.add_argument("-optimizer", type=str, default="adamw", help="Which optimizer to use", choices=['adam', 'adamw', 'sgd'])
    parser.add_argument("-subset_size", type=int, default=500, help="Subset size of random samples from the main dataset")
    parser.add_argument('-n_epochs','-epochs', type=int, default=50, help="Number of epochs to train the model.")
    parser.add_argument("-scheduler_type", type=str, default="plateau", help="Which scheduler to use", choices=['plateau', 'step'])
    parser.add_argument("-scheduler_kwargs", type=dict, default={}, help="Scheduler kwargs")
    parser.add_argument("-optimizer_kwargs", type=dict, default={}, help="Optimizer kwargs")
    parser.add_argument("-balanced", action="store_true", help="Add cross entropy loss balancing for imbalanced datasets")
    parser.add_argument("-balancing_weight", type=float, default=0.4, help="Weight for balancing the loss")
    parser.add_argument("-patience", type=int, default=16, help="Patience for early stopping")
    parser.add_argument("-dropout_prob", type=float, default=0.01, help="Dropout probability")
    parser.add_argument("-val_interval", type=int, default=1, help="Validation interval, every n epochs do validation")
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
    
    # NOTE: cluster-specific path, replace with your own location
    args.model = os.path.join('/leonardo_scratch/fast/IscrC_ARGO/results/CBMAT', args.model_path)
    model = CBMLITE(args)
    train_y = []
    #data_train = GenericDataset(ds_name = args.dataset, split='train')
    #for i in range(len(data_train)):
    #    train_y.append(data_train[i][2])
    #train_y = torch.stack(train_y, dim=0)
    test_y = []
    data_test = GenericDataset(ds_name = args.dataset, split='test', transform=model.get_transform('test'))
    imgs_test = []
    concepts_gts = []
    for i in range(len(data_test)):
        test_y.append(data_test[i][2])
        imgs_test.append(data_test[i][0])
        concepts_gts.append(data_test[i][1])
    #imgs_test = torch.cat(imgs_test, dim=0)
    #print(imgs_test.shape)
    test_y = torch.stack(test_y, dim=0)
    concepts_gts = torch.stack(concepts_gts, dim=0)
    #val_y = []
    #data_val = GenericDataset(ds_name = args.dataset, split='val')
    #for i in range(len(data_val)):
    #    val_y.append(data_val[i][2])
    #val_y = torch.stack(val_y, dim=0)
    
    folder_name = args.model_path.replace(".pt","")
    args.save_dir = os.path.join(args.save_dir, folder_name)
    os.makedirs(args.save_dir, exist_ok=True)
    #train = model.backbone.get_concept_preds('train')
    loader = torch.utils.data.DataLoader(imgs_test, batch_size = model.args.batch_size, shuffle = False)
    test = []
    for batch in loader:
        print(batch.shape)
        batch = batch.to(args.device)
        test.append(model.model(batch)['concept_probs'])
    test = torch.cat(test, dim=0)
    print(test.shape)
    
    
    import pandas as pd
    import numpy as np
    
    
    for i in range(test.shape[1]):
        plot_im_reliability_diagram(y_probs=test.detach().cpu()[:,i], y_true = concepts_gts[:,i], save_path=f"./figs/ece/{args.dataset}{i}_CBM_im_calibration.png", title="CBM@ concept calibration")

    from argus.metrics.ecce import _cumulative, compute_metrics, _equiprob, _equisamp
    data = []
    for i in range(test.shape[1]):
        kuiper, ks, lenscale = _cumulative(probs=test.detach().cpu()[:,i], labels= concepts_gts[:,i], majorticks=5, minorticks=20, filename=f"./figs/ece/{args.dataset}{i}_CBMAT_ECCE.pdf")
        ece1,ece2 = _equiprob(probs=test.detach().cpu()[:,i],labels= concepts_gts[:,i], nbins=20, filename=f"./figs/ece/{args.dataset}{i}_CBMAT_ECE.pdf")
        data.append({
            'kuiper': kuiper,
            'ks': ks,
            'lenscale':lenscale,
            'ece1':ece1,
            'ece2':ece2,
        })
        plot_im_reliability_diagram(y_probs=test.detach().cpu()[:,i], y_true = concepts_gts[:,i], save_path=f"./figs/ece/{args.dataset}{i}_CBM_im_calibration.png", title="CBM@ concept calibration")
    
    import pandas as pd
    df = pd.DataFrame(data)
    seed = args.model_path.split('SEED=')[1].split('_')[0]
    df.to_csv(f"{args.save_dir}_ARGOCBM.csv", index=False)

    res = {}
    for m in ['kuiper', 'ks', 'lenscale','ece1', 'ece2']:
        average = []
        for d in data:
            average.append(d[m])
        res[m] = np.mean(average)

    print(res)
        