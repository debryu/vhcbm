from CQA.models.vlgcbm import VLGCBM
from CQA.datasets import GenericDataset
from CQA.config import CONCEPT_SETS, SAVED_MODELS_FOLDER, ACTIVATIONS_PATH
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
    parser.add_argument("-save_dir", type=str, default=None, help="Folder where to save the model")
    parser.add_argument("-wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("-seed", type=int, default=42, help="Set the random seed")
    #parser.add_argument("-model_path", type=str, default="shapes3d", help="Path to ARGO models")
    parser.add_argument("-resume", type=str, default=None, help="Path to a model to resume training")
    parser.add_argument("-model_path", type=str, default=None, help="Where the ARGO model is saved")
    parser.add_argument("-batch_size", type=int, default=128, help="Batch size used when saving model/CLIP activations")
    parser.add_argument("-filter_set", type=str, default=None, help="path to concept set name")
    parser.add_argument("-val_split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("-backbone",type=str,default="clip_RN50",help="Which pretrained model to use as backbone",)
    parser.add_argument("-feature_layer",type=str,default="layer4",help="Which layer to collect activations from. Should be the name of second to last layer in the model",)
    parser.add_argument("-use_clip_penultimate",action="store_true",help="Whether to use the penultimate layer of the CLIP backbone",)
    parser.add_argument("-skip_concept_filter",action="store_true",help="Whether to skip filtering concepts",)
    parser.add_argument("-annotation_dir",type=str,default="./data/VLG_annotations",help="where annotations are saved",)
    parser.add_argument("-device", type=str, default="cuda", help="Which device to use")
    parser.add_argument("-num_workers",type=int,default=1,help="Number of workers used for loading data",)
    parser.add_argument("-allones_concept",action="store_true",help="Change concept dataset to ones corresponding to class",)
    # arguments for CBL
    parser.add_argument("-crop_to_concept_prob",type=float,default=0.0,help="Probability of cropping to concept granuality",)
    parser.add_argument("-cbl_confidence_threshold",type=float,default=0.15,help="Confidence threshold for bouding boxes to use",)
    parser.add_argument("-cbl_hidden_layers",type=int,default=3,help="how many hidden layers to use in the projection layer",)
    parser.add_argument("-cbl_batch_size",type=int,default=32,help="Batch size used when fitting projection layer",)
    parser.add_argument("-cbl_epochs",type=int,default=20,help="how many steps to train the projection layer for",)
    parser.add_argument("-cbl_weight_decay",type=float,default=1e-5,help="weight decay for training the projection layer",)
    parser.add_argument("-cbl_lr",type=float,default=5e-4,help="learning rate for training the projection layer",)
    parser.add_argument("-cbl_loss_type", choices=["bce", "twoway"],default="bce",help="Loss type for training CBL",)
    parser.add_argument("-cbl_twoway_tp",type=float, default=4.0,help="TPE hyperparameter for TwoWay CBL loss",)
    parser.add_argument("-cbl_pos_weight",type=float,default=1.0,help="loss weight for positive examples",)
    parser.add_argument("-cbl_auto_weight",action="store_true",help="whether to automatically weight positive examples",)
    parser.add_argument("-cbl_finetune",action="store_true",help="Enable finetuning backbone in CBL training",)
    parser.add_argument("-cbl_bb_lr_rate",type=float,default=1,help="Rescale the learning rate of backbone during finetuning",)
    parser.add_argument("-cbl_optimizer",choices=["adam", "sgd"],default="sgd",help="Optimizer used in CBL training.",)
    parser.add_argument("-cbl_scheduler",choices=[None, "cosine"],default=None,help="Scheduler used in CBL training.",)
    # arguments for SAGA
    parser.add_argument("-saga_batch_size",type=int,default=512,help="Batch size used when fitting final layer",)
    parser.add_argument("-saga_step_size", type=float, default=0.1, help="Step size for SAGA")
    parser.add_argument("-saga_lam",type=float,default=0.0007,help="Sparsity regularization parameter, higher->more sparse",)
    parser.add_argument("-saga_n_iters",type=int,default=2000,help="How many iterations to run the final layer solver for",)
    parser.add_argument("-dense", action="store_true", help="train with dense or sparse method")
    parser.add_argument("-dense_lr",type=float,default=0.001,help="Learning rate for the dense final layer training",)
    parser.add_argument("-data_parallel", action="store_true")
    parser.add_argument("-visualize_concepts", action="store_true", help="Visualize concepts")
    #parser.add_argument("-feature_layer", type=str, default='layer4', 
                        #help="Which layer to collect activations from. Should be the name of second to last layer in the model")
    #parser.add_argument("-backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
    #parser.add_argument("-activation_dir", type=str, default='shared', help="save location for backbone and CLIP activations")
    
    # Parse known arguments to determine the value of --model
    args = parser.parse_args()
    args.concept_set = CONCEPT_SETS[args.dataset]
    
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
    SAVE_DIR = args.save_dir
    args.load_dir = args.model_path
    #args.model_path = "cub-220-randomc-RBF-686_2026_02_19_21_31_SEED=686.pt"
    #args.model_path = "celeba-220-randomc-RBF-868_2026_02_19_20_13_SEED=868.pt"
    #args.model_path = "cub-220-randomc-RBF-123_2026_02_21_19_34_SEED=123.pt"
    
    # NOTE: cluster-specific path, replace with your own location
    args.model = os.path.join('/leonardo_scratch/fast/IscrC_ARGO/competitors/ICMLmodels', args.model_path)
    args.load_dir = os.path.join('/leonardo_scratch/fast/IscrC_ARGO/competitors/ICMLmodels', args.model_path)
    model = VLGCBM(args)
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
    args.save_dir = os.path.join(SAVE_DIR, folder_name)
    #os.makedirs(args.save_dir, exist_ok=True)
    #train = model.backbone.get_concept_preds('train')
    #loader = torch.utils.data.DataLoader(imgs_test, batch_size = model.args.batch_size, shuffle = False)
    loader = model.get_loader(split='test')
    test = []
    for batch, concepts_one_hot, targets in loader:
        # batch is a list of indices — convert to tensor
        # on the cub dataset, comment the following line and use "out = model.model(batch.to(args.device))""
        # else
        #features = torch.tensor(batch, dtype=torch.long).to(args.device) and "out = model.model(features)""
        with torch.no_grad():                       
            out = model.model(batch.to(args.device))
            test.append(out['concepts'].cpu())     
    test = torch.cat(test, dim=0)
    
    import pandas as pd
    import numpy as np
    
    
    #for i in range(test.shape[1]):
    #    plot_im_reliability_diagram(y_probs=test.detach().cpu()[:,i], y_true = concepts_gts[:,i], save_path=f"./figs/ece/{args.dataset}{i}_CBM_im_calibration.png", title="CBM@ concept calibration")

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
        #plot_im_reliability_diagram(y_probs=test.detach().cpu()[:,i], y_true = concepts_gts[:,i], save_path=f"./figs/ece/{args.dataset}{i}_CBM_im_calibration.png", title="CBM@ concept calibration")
    
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(f"{args.save_dir}_VLGCBM.csv", index=False)

    res = {}
    for m in ['kuiper', 'ks', 'lenscale','ece1', 'ece2']:
        average = []
        for d in data:
            average.append(d[m])
        res[m] = np.mean(average)

    print(res)
        

#python scripts/labo_calibration.py -save_dir /leonardo_scratch/fast/IscrC_ARGO/results/calibration -dataset celeba -model_path labo_celeba_2025_11_12_11_47