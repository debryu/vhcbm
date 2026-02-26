from argus.models.SVGP import SVGP
import torch
import importlib
from utils.utils import set_seed
from loguru import logger
import sys
import json
import os
# Remove default handler
logger.remove()

# Add a new handler that only shows INFO and higher
logger.add(
    sink=sys.stdout,       # or "file.log"
    level="DEBUG"           # Only log INFO, WARNING, ERROR, CRITICAL
)



if os.path.exists("./results/grid_search.json"):
    # Load JSON from file
    with open('./results/grid_search.json', 'r') as f:
        data = json.load(f)

    # Sorted by keys
    sorted_dict = dict(sorted(data.items(), key = lambda x: x[1]['f1']))  
    print(sorted_dict)
    with open(os.path.join("./results",f'sorted.json'), 'w') as f:
        json.dump(sorted_dict, f, indent=4)


name = 'SVGP'
argparser_module = importlib.import_module(f"utils.args")
try:
    generate_model_args = getattr(argparser_module, f"generate_{name.lower()}_args")
except AttributeError:
    raise ValueError(f"Argparser method for {name} not defined. Do it in CQA/utils/arg.py")
    
    
'''
DATASET
'''
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from CQA.datasets import GenericDataset
import random 

ds_name = 'celeba'

if ds_name == 'celeba':
    n_concepts = 39
else: 
    n_concepts = 40
    
if ds_name == 'cub':
    classes = range(200) 
else:
    classes = [0,1]         # CELEBA and shapes3d classes

# Load Embeddings
CLIP_NAME = 'ViT-L/14'
CLIP_NAME_SANITIZED = CLIP_NAME.replace('/','%')
with torch.no_grad():
    train_embeddings = torch.load(f'./activations/{ds_name}_train_{CLIP_NAME_SANITIZED}.pth')
    # Normalization
    train_embeddings /= torch.norm(train_embeddings, dim=1, keepdim=True)
    
    test_embeddings = torch.load(f'./activations/{ds_name}_test_{CLIP_NAME_SANITIZED}.pth')
    # Normalization
    test_embeddings /= torch.norm(test_embeddings, dim=1, keepdim=True)
    
    val_embeddings = torch.load(f'./activations/{ds_name}_val_{CLIP_NAME_SANITIZED}.pth')
    # Normalization
    val_embeddings /= torch.norm(val_embeddings, dim=1, keepdim=True)
    
    text_embeddings = torch.load(f'./activations/{ds_name}_concepts_{CLIP_NAME_SANITIZED}.pth')
    # Normalization
    text_embeddings /= torch.norm(text_embeddings, dim=1, keepdim=True)


from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

preprocess =  Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        ToTensor(),
    ])

results = {}
 
for ls in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    for outs in [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:
        for lr in [0.01,0.005,0.001,0.0005,0.0001]:
            dicta = {"gp_training_iter":1000, 
                    'inducing_points': train_embeddings[5000:6000], 
                    "lengthscale":ls,
                    "outputscale":outs,
                    "lr": lr,
                    "n_inducing_points": 100,
                    "seed": random.randint(0,100000),
                    "num_concepts":n_concepts}
            args = generate_model_args(**dicta)
            model = SVGP(args)

            data_train = GenericDataset(ds_name=ds_name, split = 'train')
            train_indices = list(range(len(data_train)))
            


            N = 100
            num_tasks = n_concepts # 10 independent binary labels
            train_subset = train_indices[0:N]
            train_x = train_embeddings[train_subset,:]

            train_y = []
            for i in train_subset:
                concepts = data_train[i][1].unsqueeze(0).to(args.device)
                train_y.append(concepts)
            train_y = torch.cat(train_y, dim=0)  # [N, T]



            model.train(train_x,train_y)

            data_test = GenericDataset(ds_name=ds_name, split = 'test')
            test_indices = list(range(len(data_test)))
            test_subset = test_indices[0:]
            test_x = test_embeddings[test_subset,:]

            test_y = []
            for i in test_subset:
                concepts = data_test[i][1].unsqueeze(0).to(args.device)
                test_y.append(concepts)
            test_y = torch.cat(test_y, dim=0)  # [N, T]

            # These are the predicted probabilities [N, T]
            out_dict = model.eval(test_x,test_y)
            pred_probs = out_dict['probs']
            pred_vars = out_dict['std']

            # Evaluate for the first time with 100 samples
            res = model.evaluate(pred_probs.cpu(),test_y.cpu())
        
            results[f"lr>{lr}> ls>{ls}> out>{outs}"] = res

with open(os.path.join("./results",f'grid_search.json'), 'w') as f:
        json.dump(results, f, indent=4)