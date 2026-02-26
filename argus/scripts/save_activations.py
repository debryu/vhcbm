from CQA.datasets import GenericDataset
import CQA.utils.clip as clip
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from functools import partial 
import pickle
import os
import torch.nn.functional as F
import json
from sklearn.metrics import classification_report
from loguru import logger
import numpy as np

device = 'cuda'
ACTIVATIONS_FOLDER = './activations'

MODEL_NAME = 'ViT-L/14' #['ViT-L/14', 'dinov3']
MODEL_NAME_SANITIZED = MODEL_NAME.replace('/','%')

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC # type:ignore
standard_preprocess =  Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

from argus.utils.save_activations import save_activations

# Load your dataset here
split = ['val','test','train']
ds_name = ['shapes3d','cub','dermamnist','celeba']
for ds in ds_name:
    for s in split:
        dataset = GenericDataset(ds, split = s, transform=standard_preprocess)
        print(dataset)
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        save_activations(MODEL_NAME,s,ds,ACTIVATIONS_FOLDER)
        

