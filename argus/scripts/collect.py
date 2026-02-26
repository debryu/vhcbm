import os
import json
import pandas as pd
FAST_STORAGE = os.environ.get("FAST", "/mnt/external_storage")

cmb_path = os.path.join(FAST_STORAGE, "results/CBMAT")

n_concept_per_ds = {
    'celeba':39,
    'cub':112,
    'dermamnist':1,
    'shapes3d':5,
}

results = []
for path in os.listdir(cmb_path):
    if path.endswith("txt"):
        continue
    dataset = path.split("_")[1]
    
    seed = int(path.split("SEED=")[1].split("_")[0])
    K = int(path.split("SUBSETSIZE=")[1])
    metrics_path = f"{cmb_path}/{path}/metrics.txt"
    with open(metrics_path, 'r') as file:
        data = json.load(file)
        n_concepts = n_concept_per_ds[dataset]
        data.update({"model":"cbmat",
                     "concept_used": K*n_concepts,
                     "seed":seed,
                     "dataset":dataset
                     })
        results.append(data)

dataframe = pd.DataFrame(results)
dataframe.to_csv("./figs/cbmat.csv", index=False)