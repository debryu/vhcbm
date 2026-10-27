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
    from argus.models.linear import load_trainer
    print(args.model_path)
    model = load_trainer(args.model_path)

    test_y = []
    concepts_gts = []
    data_test = GenericDataset(ds_name = args.dataset, split='test')
    for i in range(len(data_test)):
        test_y.append(data_test[i][2])
        concepts_gts.append(data_test[i][1])
    concepts_gts = torch.stack(concepts_gts, dim=0)

    folder_name = args.model_path.replace(".pt","")
    args.save_dir = os.path.join(args.save_dir, folder_name)
    #os.makedirs(args.save_dir, exist_ok=True)

    DEVICE = str(os.environ.get("DEVICE",'cuda'))
    KERNEL = os.environ.get("KERNEL", 'linear')
    FAST_STORAGE = os.environ.get("FAST", "/home/nicola.debole/projects/Argus/argus")
    SAVE_FOLDER = f"{FAST_STORAGE}/results/baselines"
    MODEL_NAME = str(os.environ.get("MODEL_NAME",'ViT-L/14'))
    activations_folder = os.path.join(FAST_STORAGE,"activations")
    # Load Embeddings
    MODEL_NAME_SANITIZED = MODEL_NAME.replace('/','%')

    with torch.no_grad():
        train_embeddings = torch.load(os.path.join(activations_folder,f'{args.dataset}_train_{MODEL_NAME_SANITIZED}.pth'), map_location=DEVICE)
        # Normalization
        train_embeddings = (train_embeddings - torch.mean(train_embeddings, dim=0, keepdim=True))/torch.std(train_embeddings, dim=0, keepdim=True)
        
        test_embeddings = torch.load(os.path.join(activations_folder,f'{args.dataset}_test_{MODEL_NAME_SANITIZED}.pth'), map_location=DEVICE)
        # Normalization
        test_embeddings = (test_embeddings - torch.mean(test_embeddings, dim=0, keepdim=True))/torch.std(test_embeddings, dim=0, keepdim=True)
        
        val_embeddings = torch.load(os.path.join(activations_folder,f'{args.dataset}_val_{MODEL_NAME_SANITIZED}.pth'), map_location=DEVICE)
        # Normalization
        val_embeddings = (val_embeddings - torch.mean(val_embeddings, dim=0, keepdim=True))/torch.std(val_embeddings, dim=0, keepdim=True)
        
        text_embeddings = torch.load(os.path.join(activations_folder,f'{args.dataset}_concepts_{MODEL_NAME_SANITIZED}.pth'))
        # Normalization
        text_embeddings = (text_embeddings - torch.mean(text_embeddings, dim=0, keepdim=True))/torch.std(text_embeddings, dim=0, keepdim=True)

    test,_ = model.eval(test_embeddings, concepts_gts)
    
    #import pdb; pdb.set_trace()
    import pandas as pd
    import numpy as np
    
    #for i in range(test.shape[1]):
    #    plot_reliability_diagram(probs=test.detach().cpu()[:,i], labels= concepts_gts[:,i], save_path=f"./figs/ece/{args.dataset}{i}_LINEAR_PROBE_calibration.png", title="global concept calibration")
        
    from argus.metrics.ecce import _cumulative, compute_metrics, _equiprob, _equisamp
    data = []
    preds = test.detach().cpu()
    for i in range(preds.shape[1]):
        kuiper, ks, lenscale = _cumulative(probs=preds.cpu()[:,i],labels= concepts_gts[:,i], majorticks=5, minorticks=20, filename=f"./figs/ece/{args.dataset}{i}_LP_ECCE.pdf")
        ece1,ece2 = _equiprob(probs=preds.cpu()[:,i],labels= concepts_gts[:,i], nbins=20, filename=f"./figs/ece/{args.dataset}{i}_LP_ECE.pdf")
        data.append({
            'kuiper': kuiper,
            'ks': ks,
            'lenscale':lenscale,
            'ece1':ece1,
            'ece2':ece2,
        })
        #res2 = _equisamp(probs=preds.cpu()[:,2],labels= test['concept_gts'][:,2], nbins=100)
        #import pdb; pdb.set_trace()
        #plot_reliability_diagram(probs=preds.cpu()[:,i], labels= concepts_gts[:,i], save_path=f"./figs/ece/{args.dataset}{i}_LINEAR_PROBE_calibration.png", title="global concept calibration")
        #plot_im_reliability_diagram(y_probs=preds.cpu()[:,i], y_true= concepts_gts[:,i], save_path=f"./figs/ece/{args.dataset}{i}_ARGO_im_calibration.png", title="global concept calibration")

    import pandas as pd
    df = pd.DataFrame(data)
    seed = args.model_path.split('SEED=')[1].split('.')[0]
    df.to_csv(f"{args.save_dir}_LPAT.csv", index=False)
    res = {}
    for m in ['kuiper', 'ks', 'lenscale','ece1', 'ece2']:
        average = []
        for d in data:
            average.append(d[m])
        res[m] = np.mean(average)