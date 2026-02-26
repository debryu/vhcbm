import os
import json
import pandas as pd
from loguru import logger
FAST_STORAGE = os.environ["FAST"]
folders = os.listdir(os.path.join(FAST_STORAGE, "results/ARGOCBM")) 

res = []
n_concept_per_ds = {
    'celeba':39,
    'cub':112,
    'dermamnist':1,
    'shapes3d':5,
}

for folder in folders:
    metrics_path = os.path.join(FAST_STORAGE,"results/ARGOCBM",folder,"metrics.txt")
    if os.path.exists(metrics_path):
        print(folder)
        with open(metrics_path, 'r') as file:
            data = json.load(file)
            dataset = folder.split("-")[0]
            n_concepts = n_concept_per_ds[dataset]
            K = int(folder.split("-")[1])
            seed = int(folder.split("SEED=")[-1])
            data.update({"model":"ours",
                        "concept_used": K*n_concepts,
                        "seed":seed,
                        "dataset":dataset
                        })
            res.append(data)
    else:
        logger.error(f"Can't find {metrics_path}")
dataframe = pd.DataFrame(res)
dataframe.head()
dataframe.to_csv("./figs/argocbm.csv", index=False)