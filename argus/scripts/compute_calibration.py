import os, itertools, subprocess, datetime

ARGO_MODELS=[
    'cub-340-randomc-RBF-123_2026_02_21_19_34_SEED=123',
    'cub-340-randomc-RBF-231_2026_02_22_11_48_SEED=231',
    'cub-340-randomc-RBF-321_2026_02_21_19_34_SEED=321',

    'celeba-340-randomc-RBF-123_2026_02_21_19_34_SEED=123',
    'celeba-340-randomc-RBF-231_2026_02_22_01_37_SEED=231',
    'celeba-340-randomc-RBF-321_2026_02_21_19_34_SEED=321',

    'dermamnist-340-randomc-RBF-123_2026_02_21_19_33_SEED=123',
    'dermamnist-340-randomc-RBF-231_2026_02_21_20_57_SEED=231',
    'dermamnist-340-randomc-RBF-321_2026_02_21_20_32_SEED=321',

    'shapes3d-340-randomc-RBF-123_2026_02_21_19_33_SEED=123',
    'shapes3d-340-randomc-RBF-321_2026_02_21_19_44_SEED=321',
    'shapes3d-340-randomc-RBF-231_2026_02_21_20_44_SEED=231',
]

ARGO_ACTIVE_MODELS=[
    #'clip/cub-510-activeours-RBF-132_2026_04_05_11_56_SEED=132',
    #'clip/cub-510-activeours-RBF-213_2026_04_05_11_56_SEED=213',
    #'clip/cub-510-activeours-RBF-312_2026_04_05_11_56_SEED=312',

    #'clip/celeba-510-activeours-RBF-132_2026_04_05_11_55_SEED=132',
    #'clip/celeba-510-activeours-RBF-213_2026_04_05_17_51_SEED=213',
    'clip/celeba-510-activeours-RBF-321_2026_04_01_07_25_SEED=321',

    #'clip/dermamnist-510-activeours-RBF-132_2026_04_14_12_02_SEED=132',
    #'clip/dermamnist-510-activeours-RBF-213_2026_04_14_12_12_SEED=213',
    #'clip/dermamnist-510-activeours-RBF-312_2026_04_14_12_12_SEED=312',

    #'clip/shapes3d-510-activeours-RBF-132_2026_04_05_11_55_SEED=132',
    #'clip/shapes3d-510-activeours-RBF-213_2026_04_05_12_47_SEED=213',
    #'clip/shapes3d-510-activeours-RBF-312_2026_04_05_13_17_SEED=312',
]

ARGO_ACTIVE_MODELS_DINO=[
    'dino/cub-510-activeours-RBF-111_2026_03_29_16_28_SEED=111',
    'dino/cub-510-activeours-RBF-222_2026_03_29_16_29_SEED=222',
    'dino/cub-510-activeours-RBF-333_2026_03_29_16_54_SEED=333',

    'dino/celeba-510-activeours-RBF-111_2026_03_27_11_07_SEED=111',
    'dino/celeba-510-activeours-RBF-222_2026_03_27_19_59_SEED=222',
    'dino/celeba-510-activeours-RBF-333_2026_03_28_04_40_SEED=333',

    'dino/dermamnist-510-activeours-RBF-222_2026_04_14_13_23_SEED=222',
    'dino/dermamnist-510-activeours-RBF-333_2026_04_14_13_23_SEED=333',
    'dino/dermamnist-510-activeours-RBF-444_2026_04_14_13_23_SEED=444',

    'dino/shapes3d-510-activeours-RBF-111_2026_03_27_11_07_SEED=111',
    'dino/shapes3d-510-activeours-RBF-222_2026_03_27_13_15_SEED=222',
    'dino/shapes3d-510-activeours-RBF-333_2026_03_27_14_38_SEED=333',
    
]


CBMAT_MODELS = [
    'cbmlite_celeba_2026_02_22_18_17_SEED=123_SUBSETSIZE=340',
    'cbmlite_celeba_2026_02_22_18_35_SEED=321_SUBSETSIZE=340',
    'cbmlite_celeba_2026_02_22_18_52_SEED=231_SUBSETSIZE=340',

    'cbmlite_cub_2026_02_22_18_37_SEED=123_SUBSETSIZE=340',
    'cbmlite_cub_2026_02_22_19_24_SEED=231_SUBSETSIZE=340',
    'cbmlite_cub_2026_02_22_19_02_SEED=321_SUBSETSIZE=340',

    'cbmlite_dermamnist_2026_02_22_18_21_SEED=123_SUBSETSIZE=340',
    'cbmlite_dermamnist_2026_02_22_18_34_SEED=231_SUBSETSIZE=340',
    'cbmlite_dermamnist_2026_02_22_18_29_SEED=321_SUBSETSIZE=340',

    'cbmlite_shapes3d_2026_02_22_18_20_SEED=123_SUBSETSIZE=340',
    'cbmlite_shapes3d_2026_02_22_18_41_SEED=231_SUBSETSIZE=340',
    'cbmlite_shapes3d_2026_02_22_18_32_SEED=321_SUBSETSIZE=340', 
]

LP_MODELS = [
    '2026_02_20_17_47-SVM_celeba_random_linear_39__SEED=123.json',
    '2026_02_21_17_44-SVM_celeba_random_linear_39__SEED=231.json',
    '2026_02_20_16_20-SVM_celeba_random_linear_39__SEED=321.json',

    '2026_02_20_17_47-SVM_cub_random_linear_112__SEED=123.json',
    '2026_02_21_17_44-SVM_cub_random_linear_112__SEED=231.json',
    '2026_02_20_16_20-SVM_cub_random_linear_112__SEED=321.json',

    '2026_02_20_17_47-SVM_dermamnist_random_linear_7__SEED=123.json',
    '2026_02_21_17_44-SVM_dermamnist_random_linear_7__SEED=231.json',
    '2026_02_20_16_20-SVM_dermamnist_random_linear_7__SEED=321.json',

    '2026_02_20_17_47-SVM_shapes3d_random_linear_42__SEED=123.json',
    '2026_02_21_17_44-SVM_shapes3d_random_linear_42__SEED=231.json',
    '2026_02_20_16_20-SVM_shapes3d_random_linear_42__SEED=321.json',
]

LABO_MODELS = [
    #'labo_celeba_2025_11_12_11_47',
    #'labo_celeba_2025_11_12_11_58',
    #'labo_celeba_2025_11_12_12_11',
    #'labo_celeba_2025_11_12_12_23',
    #'labo_celeba_2025_11_12_12_35',
    #'labo_cub_2026_05_06_01_01_SEED=42',
    #'labo_cub_2026_05_06_01_02_SEED=43',
    #'labo_cub_2026_05_06_01_03_SEED=44',
    #'labo_cub_2026_05_06_01_04_SEED=45',
    #'labo_cub_2026_05_06_01_05_SEED=46',
    'labo_dermamnist_2026_05_06_12_00_SEED=42',
    'labo_dermamnist_2026_05_06_12_00_SEED=43',
    'labo_dermamnist_2026_05_06_12_01_SEED=44',
    'labo_dermamnist_2026_05_06_12_01_SEED=45',
    'labo_dermamnist_2026_05_06_12_02_SEED=46',
    #'labo_shapes3d_2026_05_06_00_51_SEED=42',
    #'labo_shapes3d_2026_05_06_00_55_SEED=43',
    #'labo_shapes3d_2026_05_06_00_56_SEED=44',
    #'labo_shapes3d_2026_05_06_00_57_SEED=45',
    #'labo_shapes3d_2026_05_06_00_58_SEED=46',
]



LFCBM_MODELS = [
    'lfcbm_cub_2026_05_06_10_32_SEED=42',
    'lfcbm_cub_2026_05_06_10_35_SEED=43',
    'lfcbm_cub_2026_05_06_10_37_SEED=44',
    'lfcbm_cub_2026_05_06_10_40_SEED=45',
    'lfcbm_cub_2026_05_06_10_42_SEED=46',

    #'lfcbm_celeba_2025_11_12_11_29',
    #'lfcbm_celeba_2025_11_12_11_31',
    #'lfcbm_celeba_2025_11_12_11_32',
    #'lfcbm_celeba_2025_11_12_11_33',
    #'lfcbm_cub_2025_03_08_20_41',
    #'lfcbm_cub_2025_03_08_20_44',
    #'lfcbm_cub_2025_03_08_20_47',
    #'lfcbm_cub_2025_03_08_20_49',
    #'lfcbm_cub_2025_03_08_20_52',
    #'lfcbm_dermamnist_2025_11_14_15_07',
    #'lfcbm_dermamnist_2025_11_14_15_09',
    #'lfcbm_dermamnist_2025_11_14_15_10',
    #'lfcbm_dermamnist_2025_11_14_15_12',
    #'lfcbm_dermamnist_2025_11_14_15_13',
    #'lfcbm_shapes3d_2025_11_12_11_36',
    #'lfcbm_shapes3d_2025_11_12_11_38',
    #'lfcbm_shapes3d_2025_11_12_11_40',
    #'lfcbm_shapes3d_2025_11_12_11_41',
    #'lfcbm_shapes3d_2025_11_12_11_43',
]

VLGCBM_MODELS = [
    'vlgcbm_cub_2025_03_06_14_36',
    'vlgcbm_cub_2025_03_06_14_57',
    'vlgcbm_cub_2025_03_06_15_12',
    'vlgcbm_cub_2025_03_06_15_29',
    'vlgcbm_cub_2025_03_06_15_42',
    
    #'vlgcbm_celeba_2025_11_13_10_27',
    #'vlgcbm_celeba_2025_11_13_10_52',
    #'vlgcbm_celeba_2025_11_13_11_16',
    #'vlgcbm_celeba_2025_11_13_11_42',
    #'vlgcbm_celeba_2025_11_13_12_07',
    #'vlgcbm_cub_2025_03_10_00_57',
    #'vlgcbm_cub_2025_03_10_01_33',
    #'vlgcbm_cub_2025_03_10_02_09',
    #'vlgcbm_cub_2025_03_10_02_44',
    #'vlgcbm_cub_2025_03_10_03_20',
    #'vlgcbm_dermamnist_2025_11_13_16_22',
    #'vlgcbm_dermamnist_2025_11_13_16_26',
    #'vlgcbm_dermamnist_2025_11_13_16_30',
    #'vlgcbm_dermamnist_2025_11_13_16_34',
    #'vlgcbm_dermamnist_2025_11_13_16_39',
    #'vlgcbm_shapes3d_2025_11_13_12_31',
    #'vlgcbm_shapes3d_2025_11_13_13_06',
    #'vlgcbm_shapes3d_2025_11_13_13_42',
    #'vlgcbm_shapes3d_2025_11_13_14_23',
    #'vlgcbm_shapes3d_2025_11_13_14_59',
]

RESNETCBM_MODELS = [
    'resnetcbm_celeba_2025_11_14_18_41',
    'resnetcbm_celeba_2025_11_14_20_07',
    'resnetcbm_celeba_2025_11_14_21_34',
    'resnetcbm_celeba_2025_11_14_22_59',
    'resnetcbm_celeba_2025_11_15_00_24',
    'resnetcbm_cub_2026_05_06_11_34_SEED=42',
    'resnetcbm_cub_2026_05_06_11_47_SEED=43',
    'resnetcbm_cub_2026_05_06_12_00_SEED=44',
    'resnetcbm_cub_2026_05_06_12_12_SEED=45',
    'resnetcbm_cub_2026_05_06_12_25_SEED=46',
    'resnetcbm_dermamnist_2025_11_15_09_59',
    'resnetcbm_dermamnist_2025_11_15_10_18',
    'resnetcbm_dermamnist_2025_11_15_10_35',
    'resnetcbm_dermamnist_2025_11_15_10_53',
    'resnetcbm_dermamnist_2025_11_15_11_11',
    'resnetcbm_shapes3d_2025_11_15_01_50',
    'resnetcbm_shapes3d_2025_11_15_03_29',
    'resnetcbm_shapes3d_2025_11_15_05_05',
    'resnetcbm_shapes3d_2025_11_15_06_42',
    'resnetcbm_shapes3d_2025_11_15_08_19',
]

# NOTE: cluster-specific path, replace with your own location
save_dir = "/leonardo_scratch/fast/IscrC_ARGO/results/calibration"

'''for model_path in RESNETCBM_MODELS:
    dataset = model_path.split('_')[1]
    ARGO = ['python',
            'scripts/resnetcbm_calibration.py',
            '-save_dir', save_dir,
            '-dataset', dataset,
            '-model_path', f"{model_path}",
            ]
    print(ARGO)
    res = subprocess.run(ARGO, capture_output=True, text=True)
    print(res)'''

'''for model_path in LABO_MODELS:
    dataset = model_path.split('_')[1]
    ARGO = ['python',
            'scripts/labo_calibration.py',
            '-save_dir', save_dir,
            '-dataset', dataset,
            '-model_path', f"{model_path}",
            ]
    print(ARGO)
    res = subprocess.run(ARGO, capture_output=True, text=True)
    print(res)'''

'''for model_path in LFCBM_MODELS:
    dataset = model_path.split('_')[1]
    if dataset == 'cub':
        ARGO = ['python',
                'scripts/lfcbm_calibration.py',
                '-save_dir', save_dir,
                '-dataset', dataset,
                '-model_path', f"{model_path}",
                '-backbone', 'resnet18_cub'
                ]
    else:
        ARGO = ['python',
                'scripts/lfcbm_calibration.py',
                '-save_dir', save_dir,
                '-dataset', dataset,
                '-model_path', f"{model_path}",
                ]
    print(ARGO)
    res = subprocess.run(ARGO, capture_output=True, text=True)
    print(res)'''

for model_path in VLGCBM_MODELS:
    dataset = model_path.split('_')[1]
    if dataset == 'cub':
        ARGO = ['python',
                'scripts/vlgcbm_calibration.py',
                '-save_dir', save_dir,
                '-dataset', dataset,
                '-model_path', f"{model_path}",
                '-backbone', 'resnet18_cub',
                '-feature_layer', 'features.final_pool'
                ]
    else:
        ARGO = ['python',
                'scripts/vlgcbm_calibration.py',
                '-save_dir', save_dir,
                '-dataset', dataset,
                '-model_path', f"{model_path}",
                ]
    print(ARGO)
    res = subprocess.run(ARGO, capture_output=True, text=True)
    print(res)

'''for model_path in ARGO_ACTIVE_MODELS:
    dataset = model_path.replace('clip/','').replace('dino/','').split('-')[0]
    ARGO = ['python',
            'scripts/calibration.py',
            '-save_dir', save_dir,
            '-dataset', dataset,
            '-model_path', f"{model_path}.pt",
            ]
    res = subprocess.run(ARGO, capture_output=True, text=True)
    print(res)'''

'''for model_path in ARGO_ACTIVE_MODELS_DINO:
    dataset = model_path.replace('clip/','').replace('dino/','').split('-')[0]
    ARGO = ['python',
            'scripts/calibration.py',
            '-save_dir', save_dir,
            '-dataset', dataset,
            '-model_path', f"{model_path}.pt",
            ]
    res = subprocess.run(ARGO, capture_output=True, text=True)
    print(res)'''


'''for model_path in ARGO_MODELS:
    dataset = model_path.split('-')[0]
    ARGO = ['python',
            'scripts/calibration.py',
            '-save_dir', save_dir,
            '-dataset', dataset,
            '-model_path', f"{model_path}.pt",
            ]
    res = subprocess.run(ARGO, capture_output=True, text=True)
    print(res)'''

'''
for model_path in LP_MODELS:
    dataset = model_path.split('-')[1].split('_')[1]
    LP = ['python',
            'scripts/linear_probe_calibration.py',
            '-save_dir', save_dir,
            '-dataset', dataset,
            '-model_path', f"/leonardo_scratch/fast/IscrC_ARGO/results/baselines/{model_path}.pth",
            ]
    res = subprocess.run(LP, capture_output=True, text=True)
    print(res)

for model_path in CBMAT_MODELS:
    dataset = model_path.split('_')[1]
    CBMAT = ['python',
            'scripts/cbm_calibration.py',
            '-save_dir', save_dir,
            '-dataset', dataset,
            '-model_path', f"{model_path}",
            ]
    res = subprocess.run(CBMAT, capture_output=True, text=True)
    print(res)'''