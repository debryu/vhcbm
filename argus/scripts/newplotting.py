import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from loguru import logger
import traceback
from datetime import datetime

FAST_STORAGE = os.environ.get("FAST", "./")
RESULTS_FOLDER = os.path.join(FAST_STORAGE,"results/GPs/GPs" )
BASELINES_FOLDER = os.path.join(FAST_STORAGE,"results/baselines" )
SAVE_FOLDER = "./figs"
datasets = ['celeba', 'shapes3d', 'cub', 'dermamnist']
n_concepts = 7
K_increment = 15
blur_augmentation = 0       # 0 without, 1 with
noise_augmentation = 0      # 0 without, 1 with
later_then = 1700000000     # Unix timestamp


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

df.to_csv("./figs/results.csv")


for ds in datasets:
    df_filtered = df[df['dataset'] == ds]
    
    RR = df_filtered[(df_filtered['acq_fn'] == 'randomc') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == 'ViT-L/14') &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'RBF') 
                            ]

    RRcos = df_filtered[(df_filtered['acq_fn'] == 'randomc') &
                            (df_filtered['model'] == 'MCSVGP') & 
                            (df_filtered['backbone'] == 'ViT-L/14') &
                            (df_filtered['unix'] >= later_then) &
                            (df_filtered['blur'] == blur_augmentation) & 
                            (df_filtered['noise'] == noise_augmentation) &
                            #(df_filtered['seed'] == 123) &
                            (df_filtered['kernel'] == 'cos') 
                            ]
    
    linear_probe = df_filtered[
                          (df_filtered['model'] == 'SVM') & 
                          (df_filtered['kernel'] == 'linear') 
                            ]
    
    # try loading the baselines
    try:
        with open(os.path.join(BASELINES_FOLDER, f"{ds}.json"), 'r') as f:
            baselines = json.load(f)
    except:
        baselines = None
        logger.error(f"Error while loading the baseline values.\nConsider running baselines.py to generate them, or check if they are being loaded correctly!")
    
    # Plot all metrics
    for metric in ['f1_raw','rocauc', 'prauc']:
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
       
        sns.set_theme(style="whitegrid", font_scale=1.3)
        plt.figure(figsize=(8, 6)) 
        sns.set_palette("colorblind")
        sns.lineplot(data=RR, x='concept_used', y=metric, label='ARGO random', marker= "o", linestyle='-')
        sns.lineplot(data=linear_probe, x='concept_used', y=metric, label='linear probe', marker="d", linestyle='-')
    
        logger.info(f"GP activations for {ds}-CLIP can be identified with timestamp:")
        unique_timestamps = []
        unique_timestamps = set(unique_timestamps)
        import numpy as np
        import pandas as pd
        plt.legend()
        plt.ylabel(metric)
        plt.xlabel("Number of annotated concepts")
        plt.tight_layout()
        plt.savefig(f"{SAVE_FOLDER}/results_{metric}_{ds}.pdf", bbox_inches='tight',format='pdf' )
        plt.close()