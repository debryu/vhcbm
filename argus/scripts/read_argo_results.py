import os
import json
import pandas as pd
from loguru import logger
FAST_STORAGE = os.environ["FAST"]
BASE = os.path.join(FAST_STORAGE, "results/ARGOCBM/clip")
folders = os.listdir(BASE) 

res = []
n_concept_per_ds = {
    'celeba':39,
    'cub':112,
    'dermamnist':1,
    'shapes3d':5,
}

for folder in folders:
    metrics_path = os.path.join(BASE,folder,"metrics.txt")
    if os.path.exists(metrics_path):
        print(folder)
        with open(metrics_path, 'r') as file:
            data = json.load(file)
            dataset = folder.split("-")[0]
            n_concepts = n_concept_per_ds[dataset]
            K = int(folder.split("-")[1])
            acq_fn = folder.split("-")[2]
            seed = int(folder.split("SEED=")[-1])
            data.update({"model":"ours",
                        "concept_used": K*n_concepts,
                        "seed":seed,
                        "dataset":dataset,
                        "acq_fn": acq_fn,
                        })
            if K != 340 and K != 510:
                logger.warning(f"Unexpected {folder}")
            res.append(data)
    else:
        logger.debug(f"Can't find {metrics_path}")
dataframe = pd.DataFrame(res)
dataframe.head()

metrics = ['disentanglement', 'f1_raw', 'roc_auc','label_f1']
stats = (
    dataframe
    .groupby(['dataset', 'acq_fn'])[metrics]
    .agg(['mean', 'std'])
    .reset_index()
)

print(stats)

#dataframe.to_csv("./figs/argocbm-dino.csv", index=False)