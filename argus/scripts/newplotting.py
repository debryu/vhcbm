import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from loguru import logger
import traceback
from datetime import datetime

FAST_STORAGE = os.environ.get("FAST", "./")
SCRATCH_STORAGE = os.environ.get("SCRATCH", "./")
RESULTS_FOLDER = os.path.join(SCRATCH_STORAGE,"results/GPs/GPs" )
BASELINES_FOLDER = os.path.join(SCRATCH_STORAGE,"results/baselines" )
SAVE_FOLDER = "./figs"
VLM = 'dinov3'
VLM = 'ViT-L/14'
VLM_TO_TEXT = {
    'dinov3': 'dinov3',
    'ViT-L/14': 'clip'
}
if True:
    SAVE_FOLDER = "./figsdino"
datasets = ['dermamnist'] 
datasets = ['celeba', 'shapes3d', 'cub', 'dermamnist']
n_concepts = 7
K_increment = 15
blur_augmentation = 0       # 0 without, 1 with
noise_augmentation = 0      # 0 without, 1 with
later_then = 1767614092     # Unix timestamp
later_then = 1770378619     # Unix timestamp
later_then = 1770873850
later_then = 1771307475

DATASET_TO_NCONCEPTS = {
    "celeba": 39,
    "shapes3d": 42,
    "dermamnist": 7,
    "cub": 112,
}

results = os.listdir(RESULTS_FOLDER)
all_results = []
for r in results:
    path = os.path.join(RESULTS_FOLDER, r)
    if path.endswith(".txt"):
        continue
    try:
        date = r.split("-")[0]
        # Parse components
        year, month, day, hour, minute = map(int, date.split("_"))

        # Create datetime, assume hour=0, second=0
        dt = datetime(year, month, day, hour, minute, 0)

        # Convert to Unix timestamp
        unix_time = dt.timestamp()
        run = r.split("-")[1].replace(".json","")
        model = run.split("_")[0]
        dataset = run.split("_")[1]
        acq_fn = run.split("_")[2]
        try:
            K = int(run.split("_")[4])
        except:
            K = int(run.split("_")[5])
            
        with open(path, 'r') as f:
            dict = json.load(f)
        
        logger.info(f"{r},{len(dict)}")
        for d in dict:
            if 'augmentations' in d.keys():
                if 'blur' in d['augmentations']:
                    blur = 1
                else:
                    blur = 0
                if 'noise' in d['augmentations']:
                    noise = 1
                else:
                    noise = 0
            else:
                blur = 0
                noise = 0
            if 'concept_used' not in d.keys():
                continue
            d.update({  "run":run,
                        "acq_fn": acq_fn,
                        "dataset":dataset,
                        "K":K,
                        "blur":blur,
                        "noise":noise,
                        "date":date,
                        "unix": unix_time,
                        "model": model,
                        })
            all_results.append(d)
    except Exception as e:
        logger.error(f"Discarding {r} because of \n{e}")
        traceback.print_exc()
        pass
        #logger.debug(f"Discarding {r}")

# Convert dict to DataFrame
df = pd.DataFrame(all_results)
df['order'] = df.index

df.to_csv("./figsdino/results.csv")

for ds in datasets:
    df_filtered = df[(df['dataset'] == ds)]
    
    random = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    RR = df_filtered[(df_filtered['acq_fn'] == 'randomc') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]

    #print(RR['run'], RR['date'])
    
    RRcos = df_filtered[(df_filtered['acq_fn'] == 'randomc') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    
    RRuni = df_filtered[(df_filtered['acq_fn'] == 'randomuni') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    RA = df_filtered[(df_filtered['acq_fn'] == 'activeours') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    RAcos = df_filtered[(df_filtered['acq_fn'] == 'activeours') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    
    active = df_filtered[(df_filtered['acq_fn'] == 'active1') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]

    activec = df_filtered[(df_filtered['acq_fn'] == 'activec') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    random_RBF_MC_DIR = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    random_RBF_MC_DIR2 = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 200) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    random_RBF_MC_SOFTMAX = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 100) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    random_RBF_MC_SOFTMAX2 = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 500) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    active1_RBF_MC_DIR = df_filtered[(df_filtered['acq_fn'] == 'active') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 321) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    active_RBF_MC_DIR2 = df_filtered[(df_filtered['acq_fn'] == 'active') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 322) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    active_RBF_MC_DIR3 = df_filtered[(df_filtered['acq_fn'] == 'active') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 400) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]

    random_RBF_MC = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    random_cosb_MC = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 42) &
                            (df_filtered['kernel'] == 'cosb') 
                            ]
    
    random_cosb_MC_lowerLR = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 8) &
                            (df_filtered['kernel'] == 'cosb') 
                            ]
    
    random_RBF_MC_10 = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 10) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    random_RBF_MC_11 = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 11) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    random_RBF_MC_13 = df_filtered[(df_filtered['acq_fn'] == 'boh') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 13) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    random_cos = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 88) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    random_learn = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 88) &
                            (df_filtered['kernel'] == 'learnable') 
                            ]
    random_cosb = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= 1769299898) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cosb') 
                            ]
   
    random_cosb_clip_init = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 100) &
                            (df_filtered['kernel'] == 'cosb') 
                            ]
    ucbf_cos = df_filtered[(df_filtered['acq_fn'] == 'ucbf') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 88) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    ucbf1_cos = df_filtered[(df_filtered['acq_fn'] == 'ucbf1') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 88) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    Irandom_cos = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 89) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    Iucbf_cos = df_filtered[(df_filtered['acq_fn'] == 'ucbf') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 89) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    Iucbf1_cos = df_filtered[(df_filtered['acq_fn'] == 'ucbf1') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 89) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    
    I5random_cos = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 90) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    I5ucbf_cos = df_filtered[(df_filtered['acq_fn'] == 'ucbf') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 90) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    I5ucbf1_cos = df_filtered[(df_filtered['acq_fn'] == 'ucbf1') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 90) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    FS5random_cos = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 92) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    FS5ucbf1_cos = df_filtered[(df_filtered['acq_fn'] == 'ucbf1') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 92) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    new_random = df_filtered[(df_filtered['acq_fn'] == 'random') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] > 107) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    gp_cos = df_filtered[(df_filtered['acq_fn'] == 'gp') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] > 100) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    gb_cos = df_filtered[(df_filtered['acq_fn'] == 'gb') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] > 100) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
###
    random_nn = df_filtered[(df_filtered['acq_fn'] == 'random') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'nn') 
                            ]
    random_RBF = df_filtered[(df_filtered['acq_fn'] == 'random') &
                             (df_filtered['model'] == 'SVGP') & 
                             (df_filtered['backbone'] == VLM) &
                             (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    random_sum = df_filtered[(df_filtered['acq_fn'] == 'random') &
                             (df_filtered['model'] == 'SVGP') & 
                             (df_filtered['backbone'] == VLM) &
                             (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'sum') 
                            ]

    combinedeven_cos = df_filtered[(df_filtered['acq_fn'] == 'combinedeven') &
                                   (df_filtered['model'] == 'SVGP') & 
                                   (df_filtered['backbone'] == VLM) &
                                   (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    combinedeven_nn = df_filtered[(df_filtered['acq_fn'] == 'combinedeven') &
                                  (df_filtered['model'] == 'SVGP') & 
                                  (df_filtered['backbone'] == VLM) &
                                  (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'nn') 
                            ]
    combinedeven_RBF = df_filtered[(df_filtered['acq_fn'] == 'combinedeven') &
                                   (df_filtered['model'] == 'SVGP') & 
                                   (df_filtered['backbone'] == VLM) &
                                   (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    combinedeven_sum = df_filtered[(df_filtered['acq_fn'] == 'combinedeven') &
                                   (df_filtered['model'] == 'SVGP') & 
                                   (df_filtered['backbone'] == VLM) &
                                   (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'sum') 
                            ]
    old_entropy_cos = df_filtered[(df_filtered['acq_fn'] == 'entropy') &
                                  (df_filtered['model'] == 'SVGP') & 
                                  (df_filtered['backbone'] == VLM) &
                                  (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    old_entropy_nn = df_filtered[(df_filtered['acq_fn'] == 'entropy') &
                                 (df_filtered['model'] == 'SVGP') & 
                                 (df_filtered['backbone'] == VLM) &
                                 (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'nn') 
                            ]
    old_entropy_RBF = df_filtered[(df_filtered['acq_fn'] == 'entropy') &
                                  (df_filtered['model'] == 'SVGP') & 
                                  (df_filtered['backbone'] == VLM) &
                                  (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    old_entropy_sum = df_filtered[(df_filtered['acq_fn'] == 'entropy') &
                                  (df_filtered['model'] == 'SVGP') & 
                                  (df_filtered['backbone'] == VLM) &
                                  (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'sum') 
                            ]
    entropy_cos = df_filtered[(df_filtered['acq_fn'] == 'entropyeven') &
                              (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    entropy_nn = df_filtered[(df_filtered['acq_fn'] == 'entropyeven') &
                             (df_filtered['model'] == 'SVGP') & 
                             (df_filtered['backbone'] == VLM) &
                             (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'nn') 
                            ]
    entropy_RBF = df_filtered[(df_filtered['acq_fn'] == 'entropyeven') &
                              (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    entropy_sum = df_filtered[(df_filtered['acq_fn'] == 'entropyeven') &
                              (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'sum') 
                            ]
    ucb_cos = df_filtered[(df_filtered['acq_fn'] == 'ucb') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    
    ucbf2_cos = df_filtered[(df_filtered['acq_fn'] == 'ucbf2') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    
    uucb_cos = df_filtered[(df_filtered['acq_fn'] == 'uucb') &
                           (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    ucb_sum = df_filtered[(df_filtered['acq_fn'] == 'ucb') &
                          (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'sum') 
                            ]
    rand_mcsvgp_cos = df_filtered[(df_filtered['acq_fn'] == 'random') &
                          (df_filtered['model'] == 'MCSVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    entr_mcsvgp_cos = df_filtered[(df_filtered['acq_fn'] == 'entropy') &
                          (df_filtered['model'] == 'MCSVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    ucb_mcsvgp_cos = df_filtered[(df_filtered['acq_fn'] == 'ucb') &
                          (df_filtered['model'] == 'MCSVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    rand_mcsvgp_RBF = df_filtered[(df_filtered['acq_fn'] == 'random') &
                          (df_filtered['model'] == 'MCSVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    rand_Ysvgp_cos = df_filtered[(df_filtered['acq_fn'] == 'random') &
                          (df_filtered['model'] == 'YSVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos_y') 
                            ]
     
    rand_Ysvgp_had = df_filtered[(df_filtered['acq_fn'] == 'random') &
                          (df_filtered['model'] == 'YSVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'hadamard') 
                            ]
     
    uucb_rbf = df_filtered[(df_filtered['acq_fn'] == 'uucb') &
                           (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    
    ucbf_last = df_filtered[(df_filtered['acq_fn'] == 'ucbf') &
                           (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'last') 
                            ]
    
    random_last = df_filtered[(df_filtered['acq_fn'] == 'random') &
                           (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'last') 
                            ]
    ucbf_last_randomINIT = df_filtered[(df_filtered['acq_fn'] == 'ucbf') &
                           (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 12) & 
                            (df_filtered['kernel'] == 'last') 
                            ]
    ucbf_last_400indpoints = df_filtered[(df_filtered['acq_fn'] == 'ucbf') &
                           (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 13) & 
                            (df_filtered['kernel'] == 'last') 
                            ]
    ucbf_last_100startingpoolsize = df_filtered[(df_filtered['acq_fn'] == 'ucbf') &
                           (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['seed'] == 14) & 
                            (df_filtered['kernel'] == 'last') 
                            ]
    
    
    bald_cos = df_filtered[(df_filtered['acq_fn'] == 'bald') &
                           (df_filtered['model'] == 'SVGP') & 
                              (df_filtered['backbone'] == VLM) &
                              (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    
    variance_cos = df_filtered[(df_filtered['acq_fn'] == 'variance') &
                               (df_filtered['model'] == 'SVGP') & 
                               (df_filtered['backbone'] == VLM) &
                               (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    variance_RBF = df_filtered[(df_filtered['acq_fn'] == 'variance') &
                               (df_filtered['model'] == 'SVGP') & 
                               (df_filtered['backbone'] == VLM) &
                               (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]
    poly = df_filtered[
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'poly') 
                            ]
    copy_cos = df_filtered[
                            (df_filtered['acq_fn'] == 'copy') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['backbone'] == VLM) &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    
    dinov3ecombinedevenCos = df_filtered[
                            (df_filtered['backbone'] == 'dinov3') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['kernel'] == 'cos') &
                            (df_filtered['acq_fn'] == 'combinedeven') &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation)
                            ]
    dinov3entropyevenCos = df_filtered[
                            (df_filtered['backbone'] == 'dinov3') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['kernel'] == 'cos') &
                            (df_filtered['acq_fn'] == 'entropyeven') &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation)
                            ]
    dinov3ecombinedevenSum = df_filtered[
                            (df_filtered['backbone'] == 'dinov3') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['kernel'] == 'sum') &
                            (df_filtered['acq_fn'] == 'combinedeven') &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation)
                            ]
    dinov3ours = df_filtered[
                            (df_filtered['backbone'] == 'dinov3') &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['kernel'] == 'sum') &
                            (df_filtered['acq_fn'] == 'entropy') &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation)
                            ]
    zero_mean = df_filtered[(df_filtered['backbone'] == VLM) &
                            (df_filtered['model'] == 'SVGP') & 
                            (df_filtered['unix'] >= 1761637760) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) 
                            ]
    svm_lin = df_filtered[
                          (df_filtered['model'] == 'SVM') & 
                          (df_filtered['kernel'] == 'linear') 
                            ]
    
    linear_probe = df_filtered[
                          (df_filtered['backbone'] == VLM) & 
                          (df_filtered['model'] == 'LP') & 
                          (df_filtered['kernel'] == 'linear') 
                            ]
    
    svm_nonlin = df_filtered[
                          (df_filtered['model'] == 'SVM') & 
                          (df_filtered['kernel'] == 'nonlinear') 
                            ]
    # try loading the baselines
    try:
        with open(os.path.join(BASELINES_FOLDER, f"{ds}.json"), 'r') as f:
            baselines = json.load(f)
    except:
        baselines = None
        logger.error(f"Error while loading the baseline values.\nConsider running baselines.py to generate them, or check if they are being loaded correctly!")
    
    '''
    # Modern and clean aesthetic
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.figure(figsize=(8, 6))    
    #sns.lineplot(data=entropy_cos, x='training_pool_size', y='min auc', label='entr stratified-cos', color='green', linestyle='-', err_style='bars', errorbar=('sd', 1))
    sns.lineplot(data=random_cos, x='training_pool_size', y='min auc', label='ARGO (random)', color='#ad080f', linestyle='-')
    sns.lineplot(data=old_entropy_cos, x='training_pool_size', y='min auc', label='ARGO (entropy)', color='#568d66', linestyle='--')
    sns.lineplot(data=ucb_cos, x='training_pool_size', y='min auc', label='ARGO (ucb cos)', color='blue', linestyle='-')
    sns.lineplot(data=ucbf_cos, x='training_pool_size', y='min auc', label='ARGO (ucbf cos)', color='blue', linestyle='-')
    sns.lineplot(data=uucb_cos, x='training_pool_size', y='min auc', label='ARGO (uucb cos)', color='black', linestyle='-')
    sns.lineplot(data=ucb_sum, x='training_pool_size', y='min auc', label='ARGO (ucb sum)', color='yellow', linestyle='-')
    sns.lineplot(data=ucbf_last, x='training_pool_size', y='min auc', label='ARGO (ucbf last)', color='yellow', linestyle='-')
    sns.lineplot(data=ucbf_last_randomINIT, x='training_pool_size', y='min auc', label='rand init', color='black', linestyle='--')
    sns.lineplot(data=ucbf_last_400indpoints, x='training_pool_size', y='min auc', label='400 ind points', color='blue', linestyle='--')
    sns.lineplot(data=ucbf_last_100startingpoolsize, x='training_pool_size', y='min auc', label='100starting', color='green', linestyle='--')
    
    sns.lineplot(data=bald_cos, x='training_pool_size', y='min auc', label='ARGO (bald cos)', color='purple', linestyle='-')
    #sns.lineplot(data=rand_mcsvgp_cos, x='training_pool_size', y='min auc', label='mcsvgp_cos', color='pink', linestyle='-')
    #sns.lineplot(data=rand_mcsvgp_RBF, x='training_pool_size', y='min auc', label='mcsvgp_rbf', color='pink', linestyle='--')
    #sns.lineplot(data=entr_mcsvgp_cos, x='training_pool_size', y='min auc', label='mcsvgp_entropy_cos', color='pink', linestyle='-.')
    #sns.lineplot(data=random_RBF, x='training_pool_size', y='min auc', label='random-RBF', color='red', linestyle='-.')
    #sns.lineplot(data=ucb_mcsvgp_cos, x='training_pool_size', y='min auc', label='mcsvgp_ucb-cos', color='pink', linestyle=':')
    sns.lineplot(data=rand_Ysvgp_cos, x='training_pool_size', y='min auc', label='rand_Y_cosy', color='black', linestyle=':')
    sns.lineplot(data=rand_Ysvgp_had, x='training_pool_size', y='min auc', label='rand_Y_had', color='black', linestyle='-')
    sns.lineplot(data=svm_lin, x='training_pool_size', y='min auc', label='LIN SVM', color='orange', linestyle='-')
    sns.lineplot(data=svm_nonlin, x='training_pool_size', y='min auc', label='NON-LIN SVM', color='orange', linestyle='--')
    
    #sns.lineplot(data=entropy_sum, x='training_pool_size', y='min auc', label='entr stratified-sum', color='green', linestyle='--', err_style='bars', errorbar=('sd', 1))
    #sns.lineplot(data=random_sum, x='training_pool_size', y='min auc', label='random-sum', color='purple', linestyle='--', err_style='bars', errorbar=('sd', 1))
    #sns.lineplot(data=old_entropy_sum, x='training_pool_size', y='min auc', label='entropy-sum', color='cyan', linestyle='-', err_style='bars', errorbar=('sd', 1))
    logger.info(f"GP activations for {ds}-CLIP can be identified with timestamp:")
    unique_timestamps = []
    for d in random_cos['date']:
        unique_timestamps.append(d)
    for d in old_entropy_cos['date']:
        unique_timestamps.append(d)
    unique_timestamps = set(unique_timestamps)
    print(unique_timestamps)
    #print(dinov3ours)
    
    #sns.lineplot(data=dinov3ecombinedevenCos, x='training_pool_size', y='min auc', label='DINO-COMB-COS', color='yellow', linestyle='-', err_style='bars', errorbar=('sd', 1))
    #sns.lineplot(data=dinov3ecombinedevenSum, x='training_pool_size', y='min auc', label='DINO-COMB-SUM', color='yellow', linestyle='--', err_style='bars', errorbar=('sd', 1))
    #sns.lineplot(data=dinov3entropyevenCos, x='training_pool_size', y='min auc', label='DINO-STRAT-COS', color='blue', linestyle='--', err_style='bars', errorbar=('sd', 1))
    #sns.lineplot(data=dinov3ours, x='training_pool_size', y='min auc', label='DINO-ENTR-SUM', color='blue', linestyle='-', err_style='bars', errorbar=('sd', 1))
    #sns.lineplot(data=zero_mean, x='training_pool_size', y='min auc', label='DINO-ENTR-SUM', color='yellow', linestyle='-', err_style='bars', errorbar=('sd', 1))
    
    #sns.lineplot(data=combinedeven_sum, x='training_pool_size', y='min auc', label='combined-sum', color='black', linestyle='-.')
    #sns.lineplot(data=combinedeven_cos, x='training_pool_size', y='min auc', label='combined-cos', color='black', linestyle='-.')
    #sns.lineplot(data=copy_cos, x='training_pool_size', y='min auc', label='copy-cos', color='red', linestyle='-')
    #sns.lineplot(data=entropy_RBF, x='training_pool_size', y='min auc', label='entr-RBF', color='cyan', linestyle='-.')
    
    #sns.lineplot(data=old_entropy_RBF, x='training_pool_size', y='min auc', label='old-entr-RBF', color='blue', linestyle='-.')
    #sns.lineplot(data=entropy_nn, x='training_pool_size', y='min auc', label='entr-nn', color='cyan', linestyle=':')
    #sns.lineplot(data=random_nn, x='training_pool_size', y='min auc', label='random-nn', color='red', linestyle=':')
    #sns.lineplot(data=old_entropy_nn, x='training_pool_size', y='min auc', label='old-entr-nn', color='blue', linestyle=':')
    
    #sns.lineplot(data=poly, x='training_pool_size', y='auc', label='old-entr-poly', color='yellow', linestyle='-')
    #sns.lineplot(data=variance_cos, x='training_pool_size', y='auc', label='var-cos', color='orange', linestyle='-')
    #sns.lineplot(data=variance_RBF, x='training_pool_size', y='auc', label='var-RBF', color='orange', linestyle='--')
    #sns.lineplot(data=entropy_poly, x='training_pool_size', y='auc', label='entr poly', color='orange', linestyle='-')
    #sns.lineplot(data=calibrated, x='n_samples', y='auc', label='calibrated-rbf', color='red')
    #sns.lineplot(data=calibrated_lin, x='n_samples', y='auc', label='calibrated-lin', color='blue')
    #sns.lineplot(data=calibrated_poly, x='n_samples', y='auc', label='calibrated-poly',  color='green')
    
    # Compute mean ± SD manually to plot *horizontal* error bars
    import numpy as np
    import pandas as pd

    def add_horizontal_errorbars(data, color, linestyle):
        grouped = data.groupby('training_pool_size')['min auc']
        x = grouped.mean().index
        y = grouped.mean().values
        yerr = grouped.std().values

        plt.errorbar(
            x, y, 
            yerr=yerr, 
            fmt='none', 
            ecolor=color, 
            elinewidth=2,      # thicker horizontal bars
            capsize=0,         # no vertical caps
            alpha=0.4
        )

    # Apply to both datasets
    #add_horizontal_errorbars(random_cos, color='red', linestyle='-')
    #add_horizontal_errorbars(old_entropy_cos, color='green', linestyle='--')

    try:
        with open(os.path.join(BASELINES_FOLDER, f"{ds}.json"), 'r') as f:
            baselines = json.load(f)
            CLIP = baselines['CLIP']['minauc']
            linear = baselines['LINEAR']['minauc']
            non_linear = baselines['NON-LINEAR']['minauc']

            plt.axhline(y=CLIP, color="#9381ff", linestyle="-", label='CLIP scores', linewidth=2)
            plt.axhline(y=linear, color="#007ea7", linestyle="-.", label='Full supervision Linear Probe', linewidth=2)
            plt.axhline(y=non_linear, color="#0a4771", linestyle="-", label='Full supervision Non-Linear Probe', linewidth=2)
    except Exception as e:
        logger.error(f"Error loading baselines for {ds}: {e}")


    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Times New Roman']  # or 'Times', 'Palatino', etc.
    #plt.rcParams['mathtext.fontset'] = 'cm'           # use Computer Modern for math
    #plt.rcParams['font.size'] = 12                    # base font size
    #plt.rcParams['axes.labelsize'] = 14               # axis label size
    #plt.rcParams['axes.titlesize'] = 16               # title size
    #plt.rcParams['legend.fontsize'] = 11
    #plt.rcParams['xtick.labelsize'] = 11
    #plt.rcParams['ytick.labelsize'] = 11

    plt.legend()
    #plt.ylabel("Macro PR-AUC (avg)")
    plt.xlabel("Number of annotated samples")
    #plt.title(f"Annotator Macro PR-AUC {ds}")
    #plt.legend(frameon=False)
    #sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_FOLDER}/results_AUC_{ds}.pdf", bbox_inches='tight',format='pdf' )
    plt.close()
    '''
    # Plot all metrics
    for metric in ['f1_cal','f1_raw','rocauc', 'prauc']:
        plt.rcParams.update({
            'figure.dpi': 300,  # high resolution
            'savefig.dpi': 300,
            'axes.titlesize': 18,  # title font size
            'axes.labelsize': 16,  # x/y label font size
            'xtick.labelsize': 36,  # tick label sizes
            'ytick.labelsize': 36,
            'legend.fontsize': 14,
            'font.size': 14
        })
        #sns.set(font_scale=3,
        #    style="ticks",
        #    rc={
        #        "text.usetex": True,
        #        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} \usepackage{bm}',
        #        "font.family": "serif",
        #    })
        
        sns.set_theme(style="whitegrid", font_scale=1.3)
        plt.figure(figsize=(8, 6)) 
        sns.set_palette("colorblind")
        
        #sns.lineplot(data=random, x='concept_used', y=metric, label='ARGO random', color='green', linestyle='--')
        #print(random[metric])
        #sns.lineplot(data=active, x='concept_used', y=metric, label='ARGO ract', color='red', linestyle='-')
        #print(active[metric])
        #print(activec[metric])
        #sns.lineplot(data=activec, x='concept_used', y=metric, label='ARGO active', color='blue', linestyle='-')
        sns.lineplot(data=RR, x='concept_used', y=metric, label='ARGO random', marker= "o", linestyle='-')
        print(RR)
        #sns.lineplot(data=RRuni, x='concept_used', y=metric, label='ARGO random', color='grey', linestyle='--')
        sns.lineplot(data=RA, x='concept_used', y=metric, label='ARGO active', marker= "s", linestyle='-')
        #sns.lineplot(data=RRcos, x='concept_used', y=metric, label='ARGO random', color='grey', linestyle='-')
        #sns.lineplot(data=RAcos, x='concept_used', y=metric, label='ARGO active', color='black', linestyle='-')
        sns.lineplot(data=linear_probe, x='concept_used', y=metric, label='linear probe', marker="d", linestyle='-')
        #sns.lineplot(data=random_cos, x='training_pool_size', y=metric, label='cos', color='red', linestyle='-')
        #sns.lineplot(data=random_cosb, x='training_pool_size', y=metric, label='cos + bias', color='green', linestyle='-')
        #sns.lineplot(data=random_RBF, x='training_pool_size', y=metric, label='RBF', color='orange', linestyle='-')
        #sns.lineplot(data=random_last, x='training_pool_size', y=metric, label='RBF x COSB', color='yellow', linestyle='-')
        #sns.lineplot(data=random_nn, x='training_pool_size', y=metric, label='nn', color='grey', linestyle='-')
        #sns.lineplot(data=random_learn, x='training_pool_size', y=metric, label='RBF + COSB (learnable)', color='purple', linestyle='--')
        #sns.lineplot(data=random_RBF_MC, x='training_pool_size', y=metric, label='mc rbf', color='blue', linestyle='-')
        #sns.lineplot(data=random_cosb_MC, x='training_pool_size', y=metric, label='mc cosb', color='red', linestyle='-')
        #sns.lineplot(data=random_cosb_MC_lowerLR, x='training_pool_size', y=metric, label='mc cosb', color='red', linestyle='--')
        #sns.lineplot(data=random_RBF_MC_10, x='training_pool_size', y=metric, label='ARGO RBF', color='green', linestyle='--')
        #sns.lineplot(data=random_RBF_MC_11, x='training_pool_size', y=metric, label='ARGO RBF', color='black', linestyle='--')
        #sns.lineplot(data=random_RBF_MC_13, x='training_pool_size', y=metric, label='ARGO RBF', color='black', linestyle='-')
        #sns.lineplot(data=active1_RBF_MC_DIR, x='training_pool_size', y=metric, label='active 1', color='green', linestyle='-')
        #sns.lineplot(data=active_RBF_MC_DIR2, x='training_pool_size', y=metric, label='active 2', color='green', linestyle='--')
        #sns.lineplot(data=active_RBF_MC_DIR3, x='training_pool_size', y=metric, label='active 3', color='green', linestyle=':')
        #sns.lineplot(data=random_RBF_MC_DIR, x='training_pool_size', y=metric, label='random dir 1', color='red', linestyle='-')
        #sns.lineplot(data=random_RBF_MC_DIR2, x='training_pool_size', y=metric, label='random dir 2', color='red', linestyle='--')
        #sns.lineplot(data=random_RBF_MC_SOFTMAX, x='training_pool_size', y=metric, label='random soft 1', color='black', linestyle='-')
        #sns.lineplot(data=random_RBF_MC_SOFTMAX2, x='training_pool_size', y=metric, label='random soft 2', color='black', linestyle='--')
        #sns.lineplot(data=ucbf_cos, x='training_pool_size', y=metric, label='ARGO (ucbf cos)', color='blue', linestyle='-')
        #sns.lineplot(data=ucbf1_cos, x='training_pool_size', y=metric, label='ARGO (ucbf1 cos)', color='cyan', linestyle='-')
        #sns.lineplot(data=Irandom_cos, x='training_pool_size', y=metric, label='ARGO it(random cos)', color='#ad080f', linestyle='--')
        #sns.lineplot(data=Iucbf_cos, x='training_pool_size', y=metric, label='ARGO it(ucbf cos)', color='blue', linestyle='--')
        #sns.lineplot(data=Iucbf1_cos, x='training_pool_size', y=metric, label='ARGO it(ucbf1 cos)', color='cyan', linestyle='--')
        #sns.lineplot(data=I5random_cos, x='training_pool_size', y=metric, label='ARGO it5(random cos)', color='#ad080f', linestyle=':')
        #sns.lineplot(data=I5ucbf_cos, x='training_pool_size', y=metric, label='ARGO it5(ucbf cos)', color='blue', linestyle=':')
        #sns.lineplot(data=I5ucbf1_cos, x='training_pool_size', y=metric, label='ARGO it5(ucbf1 cos)', color='cyan', linestyle=':')
        #sns.lineplot(data=FS5random_cos, x='training_pool_size', y=metric, label='ARGO fs5(random cos)', color='gray', linestyle='-')
        #sns.lineplot(data=FS5ucbf1_cos, x='training_pool_size', y=metric, label='ARGO fs5(ucbf1 cos)', color='black', linestyle='-')
        #sns.lineplot(data=new_random, x='training_pool_size', y=metric, label='ARGO', color='red', linestyle='-')
        #sns.lineplot(data=gp_cos, x='training_pool_size', y=metric, label='ARGO gpcos', color='green', linestyle='-')
        #sns.lineplot(data=gb_cos, x='training_pool_size', y=metric, label='ARGO gbcos', color='green', linestyle='--')
        #sns.lineplot(data=random_cosb, x='training_pool_size', y=metric, label='cos + bias', color='green', linestyle='--')
        #sns.lineplot(data=random_cosb_clip_init, x='training_pool_size', y=metric, label='cos + bias', color='green', linestyle='-')
        #sns.lineplot(data=ucbf2_cos, x='training_pool_size', y=metric, label='ARGO (ucbf2 cos)', color='cyan', linestyle='--')
        #sns.lineplot(data=ucbf_last, x='training_pool_size', y=metric, label='ARGO (ucbf last)', color='blue', linestyle='--')
        #sns.lineplot(data=random_last, x='training_pool_size', y=metric, label='ARGO (random last)', color='red', linestyle='--')
        #sns.lineplot(data=svm_lin, x='training_pool_size', y='min auc', label='LIN SVM', color='orange', linestyle='-')
        #sns.lineplot(data=svm_nonlin, x='training_pool_size', y='min auc', label='NON-LIN SVM', color='orange', linestyle='--')
        logger.info(f"GP activations for {ds}-CLIP can be identified with timestamp:")
        unique_timestamps = []
        for d in random_cos['date']:
            unique_timestamps.append(d)
        for d in old_entropy_cos['date']:
            unique_timestamps.append(d)
        unique_timestamps = set(unique_timestamps)
        import numpy as np
        import pandas as pd

        '''
        if baselines is not None:
            try: 
                possible_keys = baselines['CLIP'].keys()
                if metric in possible_keys:
                    CLIP = baselines['CLIP'][metric]
                else:
                    CLIP = baselines['CLIP'][metric.replace("auc","_auc")]
                possible_keys = baselines['LINEAR'].keys()
                if metric in possible_keys:
                    linear = baselines['LINEAR'][metric]
                else:
                    linear = baselines['LINEAR'][metric.replace("auc","_auc")]
                possible_keys = baselines['NON-LINEAR'].keys()
                if metric in possible_keys:
                    non_linear = baselines['NON-LINEAR'][metric]
                else:
                    non_linear = baselines['NON-LINEAR'][metric.replace("auc","_auc")]
                plt.axhline(y=CLIP, color="black", linestyle=":", label='CLIP scores', linewidth=2)
                plt.axhline(y=linear, color="black", linestyle="-.", label='Linear probe on CLIP',linewidth=2)
                plt.axhline(y=non_linear, color="black", linestyle="-", label='Non-linear probe on CLIP',linewidth=2)
            except:
                logger.error("Failed to plot baselines, the metric may be missing")
        '''
        plt.legend()
        plt.ylabel(metric)
        plt.xlabel("Number of annotated concepts")
        plt.tight_layout()
        plt.savefig(f"{SAVE_FOLDER}/results_{VLM_TO_TEXT[VLM]}_{metric}_{ds}.pdf", bbox_inches='tight',format='pdf' )
        plt.close()