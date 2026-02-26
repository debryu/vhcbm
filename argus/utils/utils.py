import random
import numpy as np
import torch
from loguru import logger
import os
import pandas as pd
import json

def set_seed(seed):
    logger.info(f"SEED: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def collect_results(folder_res = "./results"):
    results = os.listdir(folder_res)
    all_results = []
    for r in results:
        path = os.path.join(folder_res, r)
        try:
            run = r.split("-")[0]
            dataset = r.split("_")[1]
            if dataset.endswith('.json'):
                dataset = dataset.split('.json')[0]
            acq_fn = r.split("_")[2]
            #print(acq_fn, r)
            if acq_fn.endswith('.json'):
                raise ValueError()
            if r.split("_")[2] == 'SANITYCHECK':
                    acq_fn = 'random'
            try:
                n_samples = int(r.split("_")[3])
                #print(n_samples)
            except:
                n_samples = 100
                
            with open(path, 'r') as f:
                dict = json.load(f)
            # ADD additional metadata
            i = 0
            #logger.info(f"{r},{len(dict)}")
            for d in dict:
                i+=1
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
                    
                if 'TEST' in path:
                    d.update({  "run":run,
                                "acq_fn": acq_fn,
                                "n_samples": n_samples*i,
                                "dataset":dataset,
                                "K":n_samples,
                                "blur":blur,
                                "noise":noise,
                                })
                else:
                    d.update({  "run":run,
                                "acq_fn": acq_fn,
                                "n_samples": n_samples*i,
                                "dataset":dataset,
                                "K":n_samples,
                                "blur":blur,
                                "noise":noise,
                                })
                all_results.append(d)
        except:
            pass
            #logger.debug(f"Discarding {r}")

    # Convert dict to DataFrame
    df = pd.DataFrame(all_results)
    df['order'] = df.index
    return df