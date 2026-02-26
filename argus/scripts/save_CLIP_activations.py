import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import v2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
CLIP_NAME = 'ViT-L/14'
ACTIVATIONS_DIR = "./activations"
CLIP_NAME_SANITIZED = CLIP_NAME.replace('/','%')
AUGMENTATION = 'noise'
BATCH_SIZE = 1000
device = 'cuda'


from CQA.utils.clip.clip import _convert_image_to_rgb, BICUBIC
transform_list = [
                    Resize(224, interpolation=BICUBIC),
                    CenterCrop(224),
                    _convert_image_to_rgb,
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                 ]

#save activations and get save_paths
for ds in ['shapes3d']:
    for split in ['train']:
        if AUGMENTATION is not None:
            save_name = os.path.join(ACTIVATIONS_DIR,f'{ds}_{split}_{CLIP_NAME_SANITIZED}_{AUGMENTATION}.pth')
            if AUGMENTATION.startswith('blur'):
                transform_list.insert(4,v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.)))
            if AUGMENTATION.startswith('noise'):
                transform_list.insert(4,v2.GaussianNoise(0.0, 0.1, clip=True))
                
        else:
            save_name = os.path.join(ACTIVATIONS_DIR,f'{ds}_{split}_{CLIP_NAME_SANITIZED}.pth')
        
        # Check if they are already saved
        if os.path.exists(save_name):
            continue
        
        if CLIP_NAME == 'ViT-L/14':
            import CQA.utils.clip as clip
            target_model, _ = clip.load(CLIP_NAME, device=device)
        
        # Define preprocessing steps
        preprocess = Compose(transform_list)
        
        # Create dataset
        from CQA.datasets import GenericDataset
        dataset = GenericDataset(ds_name=ds, split=split, transform=preprocess)
        
        import matplotlib.pyplot as plt
        image = dataset[3][0]
        
        if split == 'train':
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu()  # remove gradients, move to CPU if needed
                
                if image.shape[0] == 1:  # grayscale
                    image = image.squeeze(0)
                    plt.imshow(image, cmap='gray')
                else:
                    image = image.permute(1, 2, 0)  # C x H x W -> H x W x C
                    # Unnormalize 
                    image = image * torch.tensor((0.26862954, 0.26130258, 0.27577711)) + torch.tensor((0.48145466, 0.4578275, 0.40821073))
                    plt.imshow(image)
                
            else:
                # If it's a PIL image already
                plt.imshow(image)
            plt.axis('off')
            plt.savefig(f'IMGsanityCHK_{ds}_{AUGMENTATION}.png', bbox_inches='tight')
            plt.close()

        all_features = []
        with torch.no_grad():
            for images, _, _ in tqdm(DataLoader(dataset, BATCH_SIZE)):
                images = images.to(device)
                features = target_model.encode_image(images.to(device))
                all_features.append(features.cpu())

        torch.save(torch.cat(all_features), save_name)
        
