import torch
from CQA.datasets import GenericDataset
import os 
from loguru import logger
import random
import torch.nn.functional as F

FAST_STORAGE = os.environ["FAST"]
MODEL_NAME = str(os.environ.get("MODEL_NAME",'ViT-L/14'))
DEVICE = str(os.environ.get("DEVICE",'cuda'))
from argus.models.MCSVGPDO import MCSVGP
import os
ds_name = os.environ.get("DS_NAME", "shapes3d")
acq_fn = os.environ.get("ACQ_FN", "random")
activations_folder = os.path.join(FAST_STORAGE,"activations")
SAVE_FOLDER = os.path.join(FAST_STORAGE,"models")



def get_argo_act(model_name, ds_name, SAVE_F = SAVE_FOLDER):
    if ds_name == 'celeba':
        n_concepts = 39
        concept_structure = {   '1':(0,0),
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
    elif ds_name == 'dermamnist':
       
        n_concepts = 7
        concept_structure = {   'finding':(0,6),
                        }
    elif ds_name == 'cub':
        
        n_concepts = 112
        concept_structure = {}
        for i in range(112):
            concept_structure[f'{i}'] = (i,i)
    elif ds_name == 'shapes3d':
    
        n_concepts = 42
        concept_structure = {
            'floor color': (0, 9),
            'wall color': (10, 19),
            'object color': (20, 29),
            'object size': (30, 37),
            'shape': (38, 41),
        }
    else:
        
        subset_size = 500
    steps = range(n_concepts,700, n_concepts)
    
    # Load Embeddings
    MODEL_NAME_SANITIZED = MODEL_NAME.replace('/','%')

    with torch.no_grad():
        train_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_train_{MODEL_NAME_SANITIZED}.pth'), map_location=DEVICE)
        # Normalization
        train_embeddings = (train_embeddings - torch.mean(train_embeddings, dim=0, keepdim=True))/torch.std(train_embeddings, dim=0, keepdim=True)
        
        test_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_test_{MODEL_NAME_SANITIZED}.pth'), map_location=DEVICE)
        # Normalization
        test_embeddings = (test_embeddings - torch.mean(test_embeddings, dim=0, keepdim=True))/torch.std(test_embeddings, dim=0, keepdim=True)
        
        val_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_val_{MODEL_NAME_SANITIZED}.pth'), map_location=DEVICE)
        # Normalization
        val_embeddings = (val_embeddings - torch.mean(val_embeddings, dim=0, keepdim=True))/torch.std(val_embeddings, dim=0, keepdim=True)
        
        text_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_concepts_{MODEL_NAME_SANITIZED}.pth'))
        # Normalization
        text_embeddings = (text_embeddings - torch.mean(text_embeddings, dim=0, keepdim=True))/torch.std(text_embeddings, dim=0, keepdim=True)
        train_embeddings = F.normalize(train_embeddings, p=2, dim=-1)
        test_embeddings = F.normalize(test_embeddings, p=2, dim=-1)
        val_embeddings = F.normalize(val_embeddings, p=2, dim=-1)
    model = MCSVGP.load(os.path.join(SAVE_F,model_name), device = 'cuda')
    
    output = model.evalmc(train_embeddings)
    data_train = []
    gt_data = GenericDataset(ds_name=ds_name, split='train')
    for i in range(len(output['probs'])):
        data_train.append((output['probs'][i].detach().cpu(),output['preds'][i].detach().cpu(), output['soft_std'][i].detach().cpu(), output['means'][i].detach().cpu(), output['std'][i].detach().cpu(), gt_data[i][1] ))
    
    output = model.evalmc(val_embeddings)
    data_val = []
    gt_data = GenericDataset(ds_name=ds_name, split='val')
    for i in range(len(output['probs'])):
        data_val.append((output['probs'][i].detach().cpu(),output['preds'][i].detach().cpu(), output['soft_std'][i].detach().cpu(), output['means'][i].detach().cpu(), output['std'][i].detach().cpu(), gt_data[i][1] ))
    
    output = model.evalmc(test_embeddings)
    data_test = []
    gt_data = GenericDataset(ds_name=ds_name, split='test')
    for i in range(len(output['probs'])):
        data_test.append((output['probs'][i].detach().cpu(),output['preds'][i].detach().cpu(), output['soft_std'][i].detach().cpu(), output['means'][i], output['std'][i].detach().cpu(), gt_data[i][1] ))
    
    return data_train, data_val, data_test


