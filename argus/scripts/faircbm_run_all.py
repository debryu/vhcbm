import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys, json, time
# NOTE: cluster-specific path, replace with your own location
sys.path.insert(0, '/leonardo_scratch/fast/IscrC_ARGO/argobackup')
sys.path.insert(0, '/leonardo/home/userexternal/ndebole0/projects/CQA')

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from faircbm_compute_metrics import run_one

DATASETS = ['shapes3d', 'dermamnist', 'celeba', 'cub']
BACKBONES = ['clip', 'dinov3']
SEEDS = [132, 213, 312]

# NOTE: cluster-specific path, replace with your own location
OUT_PATH = "/leonardo_scratch/fast/IscrC_ARGO/results/FAIRCBM_metrics_work/faircbm_metrics_N340.jsonl"


def already_done():
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add((r['dataset'], r['backbone'], r['seed']))
    return done


if __name__ == "__main__":
    done = already_done()
    jobs = [(ds, bb, s) for ds in DATASETS for bb in BACKBONES for s in SEEDS if (ds, bb, s) not in done]
    print(f"{len(done)} already done, {len(jobs)} remaining", flush=True)
    t0 = time.time()
    ctx = multiprocessing.get_context('spawn')
    out_f = open(OUT_PATH, "a")
    with ProcessPoolExecutor(max_workers=10, mp_context=ctx) as ex:
        futs = {ex.submit(run_one, ds, bb, s): (ds, bb, s) for ds, bb, s in jobs}
        for fut in as_completed(futs):
            ds, bb, s = futs[fut]
            try:
                res = fut.result()
                out_f.write(json.dumps(res) + "\n")
                out_f.flush()
                print(f"DONE {ds}/{bb}/{s}  ecer={res['ecer']:.4f} disent={res['disentanglement']:.4f}  ({time.time()-t0:.1f}s elapsed total)", flush=True)
            except Exception as e:
                print(f"FAILED {ds}/{bb}/{s}: {e}", flush=True)
    out_f.close()
    print("ALL_JOBS_SUBMITTED_DONE")
