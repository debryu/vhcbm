import os, itertools, subprocess, datetime

FAST_STORAGE = os.environ["FAST"]
to_evaluate = []
# Experiment combinations
BASE = f"{FAST_STORAGE}/results/ARGOCBM/clip"
models = os.listdir(BASE)
for folder in models:
    if "340" in folder:
        continue
    if "280" in folder:
        continue
    if "220" in folder:
        continue
    metrics_path = os.path.join(FAST_STORAGE,"results/ARGOCBM",folder,"metrics.txt")
    to_evaluate.append(os.path.join(BASE,folder))
    #if os.path.exists(metrics_path):
    #    continue
    #else:
    #    if os.path.exists(os.path.join(FAST_STORAGE,"results/ARGOCBM",folder,"args.txt")): 
    #        to_evaluate.append(os.path.join(FAST_STORAGE,"results/ARGOCBM",folder))

# Get combination for this SLURM array task
idx = 0#int(os.environ['SLURM_ARRAY_TASK_ID'])
if idx < len(to_evaluate):
    load_dir = to_evaluate[idx]

    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    outfile = f"{os.environ['FAST']}/results/CBMs/{time_date}_evalARGOCBMeval.txt"
    env = os.environ.copy()
    print(load_dir)
    # Run the experiment
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        result = subprocess.run(["python", "eval_argocbm.py", 
                                "-load_dir", load_dir,
                                ], env=env, capture_output=True, text=True)
        f.write(result.stdout)
        if result.stderr:
            f.write("\nERRORS:\n")
            f.write(result.stderr)
