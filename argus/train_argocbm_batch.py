import os, itertools, subprocess, datetime
from loguru import logger
import json 

BACKBONE = 'ViT-L/14'
ACQ_FN = 'activeours'
SEEDS = ['111','222','333','123','132','231','213','321','312']
SEEDS = ['132','213','312']
BASE = os.environ['FAST']
GPs_FOLDER = os.path.join(BASE, 'results/GPs/GPs')
models = os.listdir(os.path.join(BASE,"models_staging"))

selection = os.listdir(GPs_FOLDER)
selection = [a for a in selection if 'MCSVGP' in a]
to_run = []
for s in selection:
    with open(os.path.join(GPs_FOLDER,s), 'r') as f:
        dict = json.load(f)
    args = dict[0]
    backbone = args['backbone']
    if args['acq_fn'] == ACQ_FN and args['backbone'] == BACKBONE:
        logger.warning(s)
    else:
        continue
    seed = s.split("SEED=")[1].replace('.json','')
    if seed not in SEEDS:
        continue
    date = s.split("-")[0]
    dataset = s.split("-")[1].split("_")[1]
    #if dataset != 'cub':
    #    continue
    acq_fn = s.split("-")[1].split("_")[2]
    to_run.append((seed,date,dataset,acq_fn,backbone))
# Experiment combinations



VLM_TO_TEXT = {
    'dinov3': 'dinov3',
    'ViT-L/14': 'clip'
}

print(to_run)
print(len(to_run))
selected_models = []
backbones = []
for (seed,date,dataset, acq_fn,backbone) in to_run:
    for model in models:
        if seed not in model or date not in model or dataset not in model or acq_fn not in model:
            continue
        else:
            K = model.split('-')[1]
            if K!='340' and K!='510':
                continue
            else:
                print(model)
                selected_models.append(model)
                backbones.append(VLM_TO_TEXT[backbone])

print(selected_models)
print(len(selected_models))

# Get combination for this SLURM array task
for idx in range(len(selected_models)):
    model = selected_models[idx]
    seed = model.split("SEED=")[-1].replace(".pt","")
    # Set env vars for the experiment
    env = os.environ.copy()
    dataset = model.split("-")[0]
    save_dir = os.path.join(os.environ['FAST'],f"results/ARGOCBM/{backbones[idx]}")
    # Timestamp for output filename
    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    outfile = f"{os.environ['FAST']}/results/CBMs/{time_date}_ARGOCBM.txt"
    # Run the experiment
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    logger.error(model)
    with open(outfile, "w") as f:
        print(["python", "train_argocbm.py", 
                                "-dataset", dataset,
                                "-save_dir", save_dir,
                                "-model_path", model,
                                "-seed", seed
                                ])
        
        result = subprocess.run(["python", "train_argocbm.py", 
                                "-dataset", dataset,
                                "-save_dir", save_dir,
                                "-model_path", model,
                                "-seed", seed
                                ], env=env, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        f.write(result.stdout)
        if result.stderr:
            f.write("\nERRORS:\n")
            f.write(result.stderr)