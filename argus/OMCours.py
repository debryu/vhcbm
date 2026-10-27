from argus.models.SVGP import SVGP
#from argus.models.MCSVGP import MCSVGP
from argus.models.MCSVGPDO import MCSVGP
import torch
import importlib
from argus.utils.utils import set_seed
from loguru import logger
import sys
import json
import os
import copy
from argus.datasets import get_augmentations
from tqdm import tqdm
from argus.utils.save_activations import check_activations
import datetime
import math
import torch.nn.functional as F

start = datetime.datetime.now()
    
    
# TODO:
# 1) use test data in the first eval instead of val

time = datetime.datetime.now().strftime("%H_%M")
date = datetime.datetime.now().strftime("%Y_%m_%d")
time_date = f"{date}_{time}"

# Remove default handler
logger.remove()
logger_level = os.environ.get("LOGGER_LEVEL", "DEBUG")
logger.add(
    sink=sys.stdout,
    level=logger_level          # Only log INFO, WARNING, ERROR, CRITICAL
)

env = os.environ


FAST_STORAGE = os.environ["FAST"]
SCRATCH_STORAGE = os.environ["SCRATCH"]
extra_parameters = {"alpha":1, "beta":1, "local":True}
ds_name = os.environ.get("DS_NAME", "shapes3d")
acq_fn = os.environ.get("ACQ_FN", "random")
activations_folder = os.path.join(FAST_STORAGE,"activations")
SAVE_FOLDER = os.path.join(SCRATCH_STORAGE,"models")
KERNEL = os.environ.get("KERNEL", "RBF")     # [poly, RBF, lin, cos, sum (cos and RBF), learnable(Linear Combination of cos and RBF), nn ]
SEED = int(os.environ.get("SEED", 42))
MODEL_NAME = os.environ.get('VLM', 'ViT-L/14')
# Kernel hyperparameter overrides (sensitivity sweep, Table 3); defaults match the original values
LENGTHSCALE = float(os.environ.get("LENGTHSCALE", 1.41))
OUTPUTSCALE = float(os.environ.get("OUTPUTSCALE", 1.0))
LEARN_KERNEL = os.environ.get("LEARN_KERNEL", "1") == "1"
# CURRENTLY OVERRIDDEN TO BE EQUAL TO N_CONCEPTS
K=7           # This is the number of samples added for each active learning iterations
# CURRENTLY OVERRIDDEN TO BE EQUAL TO N_CONCEPTS
if acq_fn.endswith("c"):
    MASKING = True
else:
    MASKING = False
MASKING = True
#variation_list = ['blur', 'noise']
variation_list = ['']
# Tag sweep runs in the save name so result JSONs don't collide; absent env vars keep legacy names.
# NOTE: kept separate from variation_list, whose entries are treated as augmentations to load.
savename_variation_list = variation_list
if any(k in os.environ for k in ("LENGTHSCALE", "OUTPUTSCALE", "LEARN_KERNEL")):
    # get_savename does not encode the backbone (known caveat), so a
    # non-CLIP sweep would collide with the CLIP JSONs of the same config — tag it explicitly.
    _bb_tag = "" if MODEL_NAME == 'ViT-L/14' else "_" + MODEL_NAME.replace('/', '%')
    savename_variation_list = variation_list + [f"LS{LENGTHSCALE}_OS{OUTPUTSCALE}_LK{int(LEARN_KERNEL)}{_bb_tag}"]

# Acquisition Functions
RANDOM = True
ENTROPY = True
VARIANCE = True
SANITY_CHECK = False
SMART_INIT = False
LEARN_MEAN = True

# Since CUB has way more concepts, it has custom parameters set at line 116
N_PER_CONCEPT = 2           # Number of concepts added as initialization
FORGET = True
TRAINING_ITER = 8000
GP_PATIENCE = 3000000000
MAX_INIT_N_SAMPLES = 100
NUM_LATENTS = 10

device = 'cuda'
def get_savename(model_name, ds_name,acq_fn,K,kernel,variation_list):
    return f'{name}_{ds_name}_{acq_fn}_{kernel}_{K}_{"-".join(variation_list)}'

def save(obj, save_name:str, i:str = '0', SEED=42):
    path = os.path.join(f"{SCRATCH_STORAGE}/results/GPs/GPs/",f'{i}-{save_name}-SEED={SEED}.json')
    logger.info(f'Saved as {path}')
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)
        
def sanitize_args(args):
    args_dict = vars(args)
    sanitized = {}
    for k, v in args_dict.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            sanitized[k] = v
        elif isinstance(v, (list, dict)): # type: ignore
            try:
                json.dumps(v)  # still check serializability
                sanitized[k] = v
            except Exception:
                sanitized[k] = str(type(v))
        else:
            sanitized[k] = str(type(v))  # or custom placeholder
    return sanitized

name = 'MCSVGP'
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

kwargs = {}
if ds_name == 'celeba':
    n_concepts = 39
    BALANCED = True
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
elif ds_name == 'shapes3d': 
    n_concepts = 42
    BALANCED = False
    concept_structure = {   'floor color':(0,9),
                        'wall color':(10,19),
                        'object color':(20,29),
                        'object size':(30,37),
                        'shape':(38,41),
                        }
elif ds_name == 'nih':
    kwargs = {'root':'/home/nicola.debole/cv_datasets/NIHChestX', 'fold':2}
    n_concepts = 14
elif ds_name == 'nih4':
    kwargs = {'root':'/home/nicola.debole/cv_datasets/NIHChestX4', 'fold':2}
    n_concepts = 4
elif ds_name == 'dermamnist':
    n_concepts = 7
    BALANCED = False
    # NOTE: cluster-specific path, replace with your own location
    kwargs = {'root':'/leonardo/home/userexternal/ndebole0/data/dermamnist'}
    concepts_path = os.path.join(FAST_STORAGE,"concepts/dermamnist")
    concept_structure = {   'finding':(0,6),
                        }
else:
    n_concepts = 112

# Set K to n_concepts in order to always add at least one for each concept    
K=60
if acq_fn == 'activeours':
    K=int(K/0.633)
N_INDUCING_POINTS = 40    # Arbitrarily choosen

# Since CUB has way more concepts, do not initialize using CLIP heuristic and set a low K
if ds_name == 'cub':
    MAX_INIT_N_SAMPLES = 15
    
if ds_name == 'cub':
    classes = range(200) 
    concept_structure = {}
    for i in range(112):
        concept_structure[f'{i}'] = (i,i)
    BALANCED = True
else:
    classes = [0,1]         # CELEBA and shapes3d classes

    
#MODEL_NAME = 'BIOVIL_T'
# Load Embeddings
CLIP_NAME = 'ViT-L/14'
CLIP_NAME_SANITIZED = CLIP_NAME.replace('/','%')
MODEL_NAME_SANITIZED = MODEL_NAME.replace('/','%')


check_activations(MODEL_NAME, ds_name, activations_folder, **kwargs)

with torch.no_grad():
    train_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_train_{MODEL_NAME_SANITIZED}.pth'), map_location=device)
    # Normalization
    train_embeddings = (train_embeddings - torch.mean(train_embeddings, dim=0, keepdim=True))/torch.std(train_embeddings, dim=0, keepdim=True)
    train_embeddings = F.normalize(train_embeddings, p=2, dim=-1)
    test_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_test_{MODEL_NAME_SANITIZED}.pth'), map_location=device)
    # Normalization
    test_embeddings = (test_embeddings - torch.mean(test_embeddings, dim=0, keepdim=True))/torch.std(test_embeddings, dim=0, keepdim=True)
    test_embeddings = F.normalize(test_embeddings, p=2, dim=-1)
    val_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_val_{MODEL_NAME_SANITIZED}.pth'), map_location=device)
    
    # Normalization
    val_embeddings = (val_embeddings - torch.mean(val_embeddings, dim=0, keepdim=True))/torch.std(val_embeddings, dim=0, keepdim=True)
    val_embeddings = F.normalize(val_embeddings, p=2, dim=-1)
    #text_embeddings = torch.load(f'/home/nicola.debole/projects/Caching/activations/{ds_name}_concepts_{MODEL_NAME_SANITIZED}.pth')
    # Normalization
    #text_embeddings = (text_embeddings - torch.mean(text_embeddings, dim=0, keepdim=True))/torch.std(text_embeddings, dim=0, keepdim=True)

    clip_train_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_train_{CLIP_NAME_SANITIZED}.pth'))
    # Normalization
    clip_train_embeddings = (clip_train_embeddings - torch.mean(clip_train_embeddings, dim=0, keepdim=True))/torch.std(clip_train_embeddings, dim=0, keepdim=True)
    
    clip_test_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_test_{CLIP_NAME_SANITIZED}.pth'))
    # Normalization
    clip_test_embeddings = (clip_test_embeddings - torch.mean(clip_test_embeddings, dim=0, keepdim=True))/torch.std(clip_test_embeddings, dim=0, keepdim=True)
    
    clip_val_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_val_{CLIP_NAME_SANITIZED}.pth'))
    # Normalization
    clip_val_embeddings = (clip_val_embeddings - torch.mean(clip_val_embeddings, dim=0, keepdim=True))/torch.std(clip_val_embeddings, dim=0, keepdim=True)
    
    clip_text_embeddings = torch.load(os.path.join(activations_folder,f'{ds_name}_concepts_{CLIP_NAME_SANITIZED}.pth'))
    # Normalization
    clip_text_embeddings = (clip_text_embeddings - torch.mean(clip_text_embeddings, dim=0, keepdim=True))/torch.std(clip_text_embeddings, dim=0, keepdim=True)
    
    
    

train_size = train_embeddings.shape[0]

from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

preprocess =  Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        ToTensor(),
        #Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

dicta = {"gp_training_iter": TRAINING_ITER,
         "gp_patience": GP_PATIENCE, 
         "lr": 1e-2,
         "kernel": KERNEL,      # [poly, RBF, lin]
         "num_latents":NUM_LATENTS,
         "inducing_points": train_embeddings, 
         "lengthscale":LENGTHSCALE,         # How far function varies over the inputs
         "outputscale":OUTPUTSCALE,         # Signal function standard deviation
         "inputnoise":0.1,          # Observation noise
         "n_inducing_points": N_INDUCING_POINTS,
         "learn_mean": LEARN_MEAN,
         "learnnoise": True,
         "learnoutputscale": LEARN_KERNEL,
         "learnlengthscale": LEARN_KERNEL,
         "num_concepts":n_concepts,
         "dataset": ds_name,
         "augmentations": variation_list,
         "acq_fn":acq_fn,
         "device":device,
         "clip_init": N_PER_CONCEPT,
         "seed":SEED,
         "backbone": MODEL_NAME,
         "masking": MASKING,
         "balanced":BALANCED,
         "restart_training":FORGET,
         "smart initialization": SMART_INIT}

res_ind = 0
path = get_savename(name,ds_name,acq_fn,K,KERNEL,savename_variation_list)

# Add unique identifier ad the beginning
print('exists?')
while os.path.exists(os.path.join("./results",f'{res_ind}-{path}.json')):
    print(os.path.join("./results",f'{res_ind}-{path}.json'))
    res_ind+=1
  
args = generate_model_args(**dicta)
logger.debug(args)
logger.debug('Generating model')

model = MCSVGP(args, concept_structure=concept_structure)


model.set_text_embeddings(text_embeddings = clip_text_embeddings)
logger.debug('fetching dataset')
data_train = GenericDataset(ds_name=ds_name, split = 'train', **kwargs)
train_indices = list(range(len(data_train)))
logger.debug('Setting seed')
set_seed(args.seed)
random.shuffle(train_indices)
logger.debug('shuffled')
num_tasks = n_concepts 

# Add possible embedding indexes to the pool
model.set_pool(train_embeddings)
model.set_mask(train_embeddings, len(concept_structure.keys()))

# Prepare Val
data_val = GenericDataset(ds_name=ds_name, split = 'val', **kwargs)
val_indices = list(range(len(data_val)))

val_subset = val_indices
val_x = val_embeddings[val_subset,:]

val_y = []
for i in val_subset:
    if type(data_val[i][1]) != torch.Tensor:
        concepts = torch.tensor(data_val[i][1]).unsqueeze(0).to(args.device)
    else:
        concepts = data_val[i][1].unsqueeze(0).to(args.device)
    val_y.append(concepts)
val_y = torch.cat(val_y, dim=0)  # [N, T]
    
# Prepare Test
data_test = GenericDataset(ds_name=ds_name, split = 'test', **kwargs)
test_indices = list(range(len(data_test)))

test_subset = test_indices
test_x = test_embeddings[test_subset,:]

test_y = []
for i in test_subset:
    if type(data_test[i][1]) != torch.Tensor:
        concepts = torch.tensor(data_test[i][1]).unsqueeze(0).to(args.device)
    else:
        concepts = data_test[i][1].unsqueeze(0).to(args.device)
    test_y.append(concepts)
test_y = torch.cat(test_y, dim=0)  # [N, T]


# The number is either 10 examples per concept (5 for positive, 5 for negative)
# Or 1% of the training dataset (approximated by defect) evenly split for each concept
train_subset = []
# Load Clip scores
# For each concept, sort and choose samples from an uniform range
with torch.no_grad():
    #clip_feats = (train_embeddings.norm(dim=-1, keepdim=True).float() @ text_embeddings.norm(dim=-1, keepdim=True).float().T)
    text_normed = (clip_text_embeddings/(clip_text_embeddings.norm(p=2, dim=-1, keepdim = True).float())).cpu()
    clip_normed = (clip_train_embeddings/(clip_train_embeddings.norm(p=2, dim=-1, keepdim = True).float())).cpu()
    clip_feats = clip_normed @ text_normed.T


for task in range(clip_feats.shape[1]):
    values, indexes = torch.sort(clip_feats[:,task], dim=0)
    #print(values)
    #print(indexes)
    
    choosen = indexes[:N_PER_CONCEPT//2]
    #print(choosen)
    #print(len(choosen))
    
    for id in choosen:
        if len(train_subset) < MAX_INIT_N_SAMPLES:
            train_subset.append(indexes[id].item())
        
    if len(train_subset) >= MAX_INIT_N_SAMPLES:
        break
    
    choosen = indexes[-N_PER_CONCEPT//2:]
    #print(choosen)
    #print(len(choosen))
    for id in choosen:
        if len(train_subset) < MAX_INIT_N_SAMPLES:
            train_subset.append(indexes[id].item())
    
    if len(train_subset) >= MAX_INIT_N_SAMPLES:
        break
        
    
train_subset = random.sample(range(len(model.pool)), N_INDUCING_POINTS)

#dist_matrix = torch.cdist(train_embeddings[train_subset], train_embeddings[train_subset], p=2)
#print(dist_matrix)
#n = dist_matrix.size(0)
#indices = torch.triu_indices(n,n, offset = 1)
#distances = dist_matrix[indices[0], indices[1]]
#print(torch.median(distances).item())

#print(train_subset)
original_len = len(train_subset)
train_subset = list(set(train_subset))
removed = original_len - len(train_subset)
logger.debug(f"Removed {removed} duplicates. Final size = {len(train_subset)}")

# Add more samples if there are not enough due to duplicates
'''
if len(train_subset) < K:
    logger.warning(f"{len(train_subset)} is lower than {K} requested samples. Adding some more randomly.")
    some_random_samples = random.sample(range(len(model.pool)), len(model.pool))
    ii = 0
    while len(train_subset) < K:
        if some_random_samples[ii] not in train_subset:
            train_subset.append(some_random_samples[ii])
        ii+=1
'''

model.set_training_pool(list(train_subset))
train_x = train_embeddings[train_subset,:]

logger.debug("Collecting ground-truth labels")
train_y = []
for i in tqdm(range(len(train_embeddings))):
    if type(data_train[i][1]) != torch.Tensor:
        concepts = torch.tensor(data_train[i][1]).unsqueeze(0).to(args.device)
    else:
        concepts = data_train[i][1].unsqueeze(0).to(args.device)
    train_y.append(concepts)
train_y = torch.cat(train_y, dim=0)  # [N, T]

all_train_y = []
for i in tqdm(range(len(data_train))):
    if type(data_train[i][1]) != torch.Tensor:
        concepts = torch.tensor(data_train[i][1]).unsqueeze(0).to(args.device)
    else:
        concepts = data_train[i][1].unsqueeze(0).to(args.device)
    all_train_y.append(concepts)
all_train_y = torch.cat(all_train_y, dim=0)  # [N, T]

for aug in variation_list:
    if aug == '':
        continue
    logger.info(f'Adding {aug} variation')
    augmented_embs, augmented_labels = get_augmentations(ds_name, train_subset, aug, device=args.device)
    train_x = torch.cat((train_x,augmented_embs), dim=0)
    train_y = torch.cat((train_y,augmented_labels), dim=0)

init_training_freqs = model.get_train_freqs(data_train)
logger.debug(init_training_freqs)

'''
# Add manually missing concepts
left_to_add = []
for c_index,c_freq in enumerate(init_training_freqs):
    if c_freq == 0.0:
        # If there are no positive samples for that concept, add one manually.
        for i in range(len(data_train)):
            _,c,l = data_train[i]
            if c[c_index] == 1:
                if i not in left_to_add:
                    left_to_add.append(i)
                    logger.warning(f"Missing positive example for concept {c_index}. Adding sample {i} from training dataset.")
                    # Limit only to one
                    break
    elif c_freq == 1.0:
        # If there are no positive samples for that concept, add one manually.
        for i in range(len(data_train)):
            _,c,l = data_train[i]
            if c[c_index] == 0:
                if i not in left_to_add:
                    left_to_add.append(i)
                    logger.warning(f"Missing negative example for concept {c_index}. Adding sample {i} from training dataset.")
                    # Limit only to one
                    break
    else:
        pass
model.set_training_pool(left_to_add)
'''

model.train_labels = all_train_y
model.test_labels = test_y
model.val_labels = val_y

init_training_freqs = model.get_train_freqs(data_train)
logger.debug(init_training_freqs)

model.train(train_embeddings.to(args.device),train_y.to(args.device))
params = model.get_params()

# These are the predicted probabilities [N, T]
out_dict = model.eval(test_x.to(args.device))
pred_probs = out_dict['probs']

pred_vars = out_dict['std']

#for task_id, (task_name, idx_range) in enumerate(concept_structure.items()):
    #num_c = len(range(idx_range[0],idx_range[1]+1))
    #for i in range(len(pred_probs)):
    #    logger.warning(f"_________")
    #    print("mean", pred_probs[i,idx_range[0]:idx_range[1]+1])
    #    print("std", pred_vars[i,idx_range[0]:idx_range[1]+1])
    #    alphas = torch.exp(pred_vars[i,idx_range[0]:idx_range[1]+1])
    #    alpha_0 = torch.sum(alphas, dim=-1, keepdim=True)
    #    term1 = torch.digamma(alpha_0 + 1)
    #    term2 = torch.sum((alphas / alpha_0) * torch.digamma(alphas + 1), dim=-1, keepdim=True)
    #    aleatoric_uncertainty = (term1 - term2)/ torch.log(torch.tensor(num_c))
    #    expected_probs = alphas / alpha_0
    #    total_uncertainty = -torch.sum(expected_probs * torch.log(expected_probs + 1e-10), dim=-1)/ torch.log(torch.tensor(num_c))
    #    print("alea", aleatoric_uncertainty)
    #    print("total unc", total_uncertainty)
    #    if num_c == 1:
    #        label = val_y[i,idx_range[0]] > 0.5
    #        logger.warning(f"was {label} but predicted {(pred_probs[i,idx_range[0]] > 0.5).cpu()}")
    #    else:
    #        label = torch.argmax(val_y[i,idx_range[0]:idx_range[1]+1])
    #        logger.warning(f"was {label} but predicted {torch.argmax(pred_probs[i,idx_range[0]:idx_range[1]+1]).cpu()}")
    #input("..")

#input("..")
print(pred_probs)
print(pred_probs.shape)
print(val_y.shape)
#SVGP_res_base = model.evaluate(pred_probs.cpu(),val_y.cpu())
#SVGP_res_base.update(sanitize_args(args))
SVGP_res_base_test = model.evaluate_new(pred_probs.cpu(),test_y.cpu())
SVGP_res_base_test.update(sanitize_args(args))
SVGP_res_base_test.update({'freqs':model.get_train_freqs(data_train), 'parameters':params})


# Evaluate for the first time with 100 samples
#SVGP_res.append(model.evaluate(pred_probs.cpu(),test_y.cpu()))


#SVGP_res = [SVGP_res_base]
SVGP_res_noLR = [SVGP_res_base_test]
model.show_training_stats(data_train)

'''
dicta = {"gp_training_iter": TRAINING_ITER,
         "gp_patience": GP_PATIENCE, 
         "lr": 0.01,
         "kernel": 'RBF',      # [poly, RBF, lin]
         "inducing_points": train_embeddings, 
         "lengthscale":0.8,         # How far function varies over the inputs
         "outputscale":1.0,         # Signal function standard deviation
         "inputnoise":0.1,          # Observation noise
         "n_inducing_points": N_INDUCING_POINTS,
         "learnnoise": True,
         "num_concepts":n_concepts,
         "dataset": ds_name,
         "augmentations": variation_list,
         "acq_fn":acq_fn,
         "device":"cuda",
         "clip_init": N_PER_CONCEPT,
         "seed":SEED,
         "backbone": MODEL_NAME,
         "restart_training":FORGET,
         "smart initialization": SMART_INIT}
args = generate_model_args(**dicta)
inference_model = SVGP(args)
evaled = 0
'''

stopping = 0
max_f1 = 0
logger.debug(f"The train set has a len of {train_size}. Stopping at 10% the training set or 1000 samples.")
while(stopping <= 5):
    '''
    if len(model.training_pool)//100 > evaled:
        # Create the training pool
        ## Retrieve labels (GT data)
        T_labels = []
        for i in model.training_pool:
            concepts = data_train[i][1]
            T_labels.append(concepts)
            
        T_labels = torch.stack(T_labels, dim=0).to(model.args.device)
        ## Retrive embeddings (input)
        T_embeddings = train_embeddings[model.training_pool].to(model.args.device)
        #inference_model.train(T_embeddings,T_labels, patience=inference_model.args.gp_patience)
        # Save annotations
        #T_output = inference_model.eval(test_embeddings)
        #T_res = model.evaluate(T_output['probs'].cpu(), test_y.cpu(), mode='threshold')
        #print(T_res['auc'])
        evaled = len(model.training_pool)//100
    '''
    model.save(os.path.join(SAVE_FOLDER, f"{model.get_name()}_{time_date}_SEED={SEED}.pt"))
    res = model.train_loop(embeddings=train_embeddings, data=data_train, testx=test_x, testy=test_y, acq_fn=acq_fn, K=K, variation_list=variation_list, **extra_parameters)
    model.show_training_stats(data_train)
    params = model.get_params()
    #SVGP_res.append(res)
    #save(SVGP_res, path, time_date)
    
    # Save annotations
    output = model.eval(test_embeddings)
    print(output)
    print(test_y)

    binary_preds = (output['probs'] > 0.5).int()
    res_test = model.evaluate_new(output['probs'].cpu(), test_y.cpu())
    res_test.update({'freqs':model.get_train_freqs(data_train), 'parameters':params})
    logger.debug(model.get_train_freqs(data_train))
    #res_test['discretization'] = 'threshold'
    res_test.update(model.sanitize_args(model.args))
    SVGP_res_noLR.append(res_test)
    save(SVGP_res_noLR, f"{path}_LOGS", time_date, SEED)

    # Save the annotated training and validation set
    output_train = model.eval(train_embeddings)
    output_val = model.eval(val_embeddings)
    #print(len(val_embeddings), len(output_val['std']))

    #annotated_train = []
    #for i in range(len(data_train)):
    #    img,concepts,label = data_train[i]
    #    annotated_train.append((None,output_train['probs'][i],output_train['std'][i],label))
    #torch.save(annotated_train,f"annotated_datasets/TRAIN-{model.get_name()}_{time_date}.pt")
    
    #annotated_val = []
    #for i in range(len(data_val)):
    #    img,concepts,label = data_val[i]
    #    annotated_val.append((None,output_val['probs'][i],output_val['std'][i],label))
    #torch.save(annotated_val,f"annotated_datasets/VAL-{model.get_name()}_{time_date}.pt")
    stopping += 1
print(args)
end = datetime.datetime.now()
print('\n ### Total time taken: ', end - start)
print('\n ### Closing ###')