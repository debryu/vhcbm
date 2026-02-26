import torch
import os
from loguru import logger
from CQA.datasets import GenericDataset

# TODO: add path into config 
def get_augmentations(ds_name, indexes:list, name, device = 'cuda'):
    # Load Embeddings
    CLIP_NAME = 'ViT-L/14'
    CLIP_NAME_SANITIZED = CLIP_NAME.replace('/','%')
    path = f'./activations/{ds_name}_train_{CLIP_NAME_SANITIZED}_{name}.pth'
    if not os.path.exists(path):
        logger.error(f"Can't find {path}")
    with torch.no_grad():
        train_embeddings = torch.load(path, weights_only=True)
        # Normalization
        train_embeddings /= torch.norm(train_embeddings, dim=1, keepdim=True)
        
    data = GenericDataset(ds_name=ds_name, split = 'train')
    
    concepts = []
    for i in indexes:
        concepts.append(data[i][1])
    concepts = torch.stack(concepts, dim = 0)
    
    train_embeddings = train_embeddings.to(device)
    concepts = concepts.to(device)

    return train_embeddings[indexes], concepts
