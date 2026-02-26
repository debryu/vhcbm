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
import math
import traceback
from torchvision.transforms import v2
from CQA.utils.resnetcbm_utils import PretrainedResNetModel

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

class MODEL_WRAPPER(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def get_transform(self):
        try:
            from torchvision.transforms import InterpolationMode
            BICUBIC = InterpolationMode.BICUBIC
        except ImportError:
            BICUBIC = Image.BICUBIC # type:ignore
        preprocess =  Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                ToTensor(),
            ])
        return preprocess

    def encode_text(self,text):
        raise NotImplementedError()
    
class RESNET_CUB_WRAPPER(MODEL_WRAPPER):
    name = 'resnet18_cub'
    def __init__(self, args, device='cuda'):
        super().__init__()
        self.device = device
        self.model = PretrainedResNetModel(args)

        self._embeddings = None

        def _hook(module, inputs, output):
            # inputs is a tuple; embeddings are inputs[0]
            self._embeddings = inputs[0].detach()

        self._hook_handle = self.model.target_model.output.register_forward_hook(_hook)
    def forward(self,x):
        with torch.no_grad():
            logits = self.model(x)
            embeddings = self._embeddings
            #print(embeddings)
            
        return embeddings
    
    

class CLIP_WRAPPER(MODEL_WRAPPER):
    name = 'ViT-L/14'
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.device = device
        MODEL_NAME_SANITIZED = self.name.replace('/','%')
        self.model, self.preprocess = clip.load(self.name, device=device)
        
    def forward(self,x):
        return self.model.encode_image(x.to(self.device)).float()
    
    def encode_text(self,text):
        return self.model.encode_text(text).float()
    
    def get_transform(self):
        return self.preprocess
    
class DINO_WRAPPER(MODEL_WRAPPER):
    name = 'dinov3'
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.device = device
        from transformers import AutoImageProcessor, AutoModel
        from transformers.image_utils import load_image
        pretrained_model_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
        self.model = AutoModel.from_pretrained(
            pretrained_model_name, 
            device_map="auto", 
        )

    def forward(self,x):
        with torch.inference_mode():
            output = self.model(x.to(self.device))
        return output.pooler_output
    
    def encode_text(self,text):
        return self.model.encode_text(text).float()
    
    def get_transform(self, size=256):
        # Use the advertised transform
        to_tensor = v2.ToImage()
        resize = v2.Resize((size, size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return v2.Compose([to_tensor, resize, to_float, normalize])
    
class BIOVIL_WRAPPER(MODEL_WRAPPER):
    name = 'BIOVIL_T'
    def __init__(self, device = 'cuda'):
        super().__init__()
        self.device = device
        MODEL_NAME_SANITIZED = self.name.replace('/','%')
        from health_multimodal.image import get_image_inference
        from health_multimodal.image.utils import ImageModelType
        self.image_inference = get_image_inference(ImageModelType.BIOVIL_T)
        self.image_inference.to('cuda')
        
    def forward(self,x):
        out = self.image_inference.model(x.to(self.device))
        return out.img_embedding
    
    def get_transform(self):
        return self.image_inference.transform

def save_activations(model_name, split, dataset, activations_folder, **kwargs):
    device = 'cuda'
    MODEL_NAME_SANITIZED = model_name.replace('/','%')

    if model_name == 'ViT-L/14':
        target_model = CLIP_WRAPPER()
    elif model_name == 'BIOVIL_T':
        target_model = BIOVIL_WRAPPER()
    elif model_name == 'dinov3':
        target_model = DINO_WRAPPER()
    elif model_name == 'resnet18_cub':
        import argparse
        parser = argparse.ArgumentParser(description="Dynamic flags based on initial flag value.")
        parser.add_argument("-device", type=str, default="cuda", help="Which device to use")
        parser.add_argument("-batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
        parser.add_argument("-backbone", type=str, default="resnet18_cub", help="Which ResNet pretrained model to use as backbone")
        parser.add_argument("-unfreeze", type=int, default=0, help="Number of conv layers to unfreeze from the pretrained model")
        parser.add_argument("-num_c", type=int, default=64, help="Number of concepts to learn when unsupervised")
        # Training
        parser.add_argument("-num_workers",type=int,default=4,help="Number of workers used for loading data")
        parser.add_argument("-predictor", type=str, default="saga", help="Which linear predictor to use", choices=['saga', 'svm'])
        parser.add_argument("-c_svm", type=float, default=1, help="C hyperparameter for SVM")
        parser.add_argument("-optimizer", type=str, default="adamw", help="Which optimizer to use", choices=['adam', 'adamw', 'sgd'])
        parser.add_argument("-lr", type=float, default=0.001, help="Learning rate")
        parser.add_argument('-n_epochs','-epochs', type=int, default=50, help="Number of epochs to train the model.")
        parser.add_argument("-scheduler_type", type=str, default="plateau", help="Which scheduler to use", choices=['plateau', 'step'])
        parser.add_argument("-scheduler_kwargs", type=dict, default={}, help="Scheduler kwargs")
        parser.add_argument("-optimizer_kwargs", type=dict, default={}, help="Optimizer kwargs")
        parser.add_argument("-balanced", action="store_true", help="Add cross entropy loss balancing for imbalanced datasets")
        parser.add_argument("-balancing_weight", type=float, default=0.4, help="Weight for balancing the loss")
        parser.add_argument("-patience", type=int, default=16, help="Patience for early stopping")
        parser.add_argument("-dropout_prob", type=float, default=0.01, help="Dropout probability")
        parser.add_argument("-val_interval", type=int, default=1, help="Validation interval, every n epochs do validation")
        args = parser.parse_args()
        args.fc_layers = []
        target_model = RESNET_CUB_WRAPPER(args=args)
        
    preprocess = kwargs.get('transform',target_model.get_transform())
    batch_size = kwargs.get('batch_size',512)
    logger.debug(f"Saving activations for {model_name} {split} {dataset}. Using preprocess:{preprocess}")
    
    data = GenericDataset(dataset, split = split, transform=preprocess, **kwargs)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers = 2)
    activations = []
    for batch in tqdm(loader, desc=f"Processing {dataset} {split}"):
        imgs,concepts,labels = batch
        activations.append(target_model(imgs.to(device)).detach().cpu())
    activations = torch.cat(activations, dim=0)
    
    # Save activations
    os.makedirs(activations_folder, exist_ok = True)
    torch.save(activations, os.path.join(activations_folder,f'{dataset}_{split}_{MODEL_NAME_SANITIZED}.pth'))
    with open(os.path.join(activations_folder,f'{MODEL_NAME_SANITIZED}_{split}_transform.txt'), 'w') as f:
        f.write(str(preprocess))
        
def save_concept_activations(model_name, split, dataset, activations_folder, **kwargs):
    device = 'cuda'
    MODEL_NAME_SANITIZED = model_name.replace('/','%')
    if MODEL_NAME_SANITIZED == 'ViT-L%14':
        target_model = CLIP_WRAPPER()
    elif model_name == 'BIOVIL_T':
        target_model = BIOVIL_WRAPPER()
    else:
        return
            
    logger.debug(f"Saving CONCEPTS activations for {model_name} {split} {dataset}.")
    
    # Load concepts
    with open(os.path.join(os.environ.get("CONCEPTS_PATH", './data/concepts'),'concepts.txt')) as f:
        concept_texts = f.read().split("\n")
    print(concept_texts)
    
    text = clip.tokenize(["{}".format(concept) for concept in concept_texts]).to(device)
    text_features = []
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(target_model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    
    # Save activations
    os.makedirs(activations_folder, exist_ok = True)
    torch.save(text_features, os.path.join(activations_folder,f'{dataset}_concepts_{MODEL_NAME_SANITIZED}.pth'))
    logger.debug("Concept embeddings saved")
    
def check_activations(model_name, dataset, activations_folder, **kwargs):
    MODEL_NAME_SANITIZED = model_name.replace('/','%')
    for split in ['train', 'test', 'val']:
        if not os.path.exists(os.path.join(activations_folder,f'{dataset}_{split}_{MODEL_NAME_SANITIZED}.pth')):
            save_activations(model_name, split, dataset, activations_folder, **kwargs)
    if model_name in ['resnet18_cub']:
        return
    elif not os.path.exists(os.path.join(activations_folder,f'{dataset}_concepts_{MODEL_NAME_SANITIZED}.pth')):
        logger.warning(f"{os.path.join(activations_folder,f'{dataset}_concepts_{MODEL_NAME_SANITIZED}.pth')} not found, saving CONCEPTS activations")
        save_concept_activations(model_name=model_name, split=split, dataset=dataset, activations_folder=activations_folder, **kwargs) 
    #try:
    #    save_ensembled_activations(model_name, split, dataset, activations_folder, **kwargs)
    #except Exception as e:
    #    logger.error(f"{e}")    
    #    traceback.print_exc()   # prints full stacktrace
    
    return

def save_ensembled_activations(model_name, split, dataset, activations_folder, **kwargs):
    device = 'cuda'
    MODEL_NAME_SANITIZED = model_name.replace('/','%')

    if model_name == 'ViT-L/14':
        target_model = CLIP_WRAPPER()

    if model_name == 'BIOVIL_T':
        target_model = BIOVIL_WRAPPER()
        
    logger.debug(f"Saving ENSEMBLED CONCEPTS activations for {model_name} {split} {dataset}.")
    
    # Load concepts
    with open(os.path.join(os.environ.get("CONCEPTS_PATH", '/home/default_path'),'concepts.txt')) as f:
        concept_texts = f.read().split("\n")
        
    # load prompts
        
    with open(os.path.join(os.environ.get("PROMPTS_PATH", '/home/default_path'),'prompts.txt')) as f:
        prompts = f.read().split("\n")
        
    
    ensembled_text_feature = []
    for c in concept_texts:
        text = clip.tokenize([p.replace("{}",c) for p in prompts]).to(device)
        text_features = []
        batch_size = 64
        with torch.no_grad():
            for i in tqdm(range(math.ceil(len(text)/batch_size))):
                text_features.append(target_model.encode_text(text[batch_size*i:batch_size*(i+1)]))
        text_features = torch.cat(text_features, dim=0)
        ensembled_text_feature.append(torch.mean(text_features, dim=0))

    ensembled_text_feature = torch.stack(ensembled_text_feature, dim=0)
    
    # Save activations
    os.makedirs(activations_folder, exist_ok = True)
    torch.save(ensembled_text_feature, os.path.join(activations_folder,f'{dataset}_ensembled_concepts_{MODEL_NAME_SANITIZED}.pth'))
    