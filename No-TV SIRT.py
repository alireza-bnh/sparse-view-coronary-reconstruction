import os
import astra
import numpy as np
import pyvista as pv
from skimage.restoration import denoise_tv_chambolle

# Set random seed so Poisson noise is 100% reproducible across runs
np.random.seed(42)

RES = 300
NUM_VIEWS = 9
SOD = 750.0
SDD = 1200.0
magnification = SDD / SOD
det_pixel_size = 1.0 * magnification
ODD = SDD - SOD

DATA_DIR = r"C:\Users\alire\Desktop\Data"

PHANTOMS = [
    ("Model 1", os.path.join(DATA_DIR, "Normal_3.vtp")),
    ("Model 2", os.path.join(DATA_DIR, "Normal_18.vtp")),
    ("Model 3", os.path.join(DATA_DIR, "solid_400001.vtp"))
]

def calculate_metrics(gt, rec, threshold):
    rec_bin = (rec > threshold).astype(bool)
    gt_bin = (gt > 0.5).astype(bool)
    intersection = np.sum(rec_bin & gt_bin)
    union = np.sum(rec_bin | gt_bin)
    rec_vol = np.sum(rec_bin)
    gt_vol = np.sum(gt_bin)
    
    dice = 2.0 * intersection / (rec_vol + gt_vol) if (rec_vol + gt_vol) > 0 else 0.0
    jaccard = intersection / union if union > 0 else 0.0
    vr = rec_vol / gt_vol if gt_vol > 0 else 0.0
    return dice, jaccard, vr

def calculate_rmse(vol1, vol2):
    return np.sqrt(np.mean((vol1 - vol2) ** 2))

def get_best_metrics(vol_gt, vol_rec):
    thresholds = np.linspace(0.01, 0.50, 20)
    best_dice = -1.0
    best_t = 0.0
    best_j = 0.0
    best_vr = 0.0
    for t in thresholds:
        d, j, vr = calculate_metrics(vol_gt, vol_rec, t)
        if d > best_dice:
            best_dice = d
            best_t = t
            best_j = j
            best_vr = vr
    rmse = calculate_rmse(vol_gt, vol_rec)
    return best_t, best_dice, best_j, best_vr, rmse

print("=" * 88)
print(f"{'Model':<10} | {'Method':<22} | {'Thresh':<6} | {'DSC':<6} | {'IoU':<6} | {'VR':<6} | {'RMSE':<6}")
print("=" * 88)

for model_name, filepath in PHANTOMS:
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue

    mesh = pv.read(filepath)
    try:
        grid = pv.ImageData(dimensions=(RES, RES, RES))
    except:
        grid = pv.UniformGrid(dimensions=(RES, RES, RES))
    grid.spacing = (mesh.length / RES, mesh.length / RES, mesh.length / RES)
    grid.origin = mesh.center - np.array(grid.spacing) * RES / 2
    selection = grid.select_enclosed_points(mesh, tolerance=0.0)
    vol_gt = selection.point_data['SelectedPoints'].reshape((RES, RES, RES), order='F').astype(np.float32)

    vol_geom = astra.create_vol_geom(RES, RES, RES)
    angles = np.linspace(0, np.pi, NUM_VIEWS, False)
    proj_geom = astra.create_proj_geom('cone', det_pixel_size, det_pixel_size, RES, RES, angles, SOD, ODD)

    clean_proj_id, clean_sino = astra.create_sino3d_gpu(vol_gt, proj_geom, vol_geom)
    I0 = 5000.0
    noisy_trans = np.random.poisson(I0 * np.exp(-clean_sino))
    noisy_trans[noisy_trans == 0] = 1
    noisy_sino = -np.log(noisy_trans / I0)
    proj_id = astra.data3d.create('-sino', proj_geom, noisy_sino)

    # 1. Binary Pure (Unregularized Discrete Backprojection)
    rec_id_bp = astra.data3d.create('-vol', vol_geom)
    cfg_bp = astra.astra_dict('BP3D_CUDA')
    cfg_bp['ProjectionDataId'] = proj_id
    cfg_bp['ReconstructionDataId'] = rec_id_bp
    alg_bp = astra.algorithm.create(cfg_bp)
    astra.algorithm.run(alg_bp)
    vol_bin_pure = astra.data3d.get(rec_id_bp)
    if vol_bin_pure.max() > vol_bin_pure.min():
        vol_bin_pure = (vol_bin_pure - vol_bin_pure.min()) / (vol_bin_pure.max() - vol_bin_pure.min())
    
    # 2. Binary + TV (Applying TV proximal step to Binary reconstruction)
    vol_bin_tv = denoise_tv_chambolle(vol_bin_pure, weight=0.05)
    if vol_bin_tv.max() > vol_bin_tv.min():
        vol_bin_tv = (vol_bin_tv - vol_bin_tv.min()) / (vol_bin_tv.max() - vol_bin_tv.min())

    # 3. Weighted Pure SIRT (150 iterations, NO TV)
    rec_id_sirt_pure = astra.data3d.create('-vol', vol_geom)
    cfg_sirt_pure = astra.astra_dict('SIRT3D_CUDA')
    cfg_sirt_pure['ProjectionDataId'] = proj_id
    cfg_sirt_pure['ReconstructionDataId'] = rec_id_sirt_pure
    cfg_sirt_pure['option'] = {'MinConstraint': 0}
    alg_sirt_pure = astra.algorithm.create(cfg_sirt_pure)
    astra.algorithm.run(alg_sirt_pure, 150)
    vol_w_pure = astra.data3d.get(rec_id_sirt_pure)
    if vol_w_pure.max() > vol_w_pure.min():
        vol_w_pure = (vol_w_pure - vol_w_pure.min()) / (vol_w_pure.max() - vol_w_pure.min())

    # 4. Weighted SIRT + TV (150 iterations, TV every 10 iters, lambda=0.05)
    rec_id_sirt_tv = astra.data3d.create('-vol', vol_geom)
    cfg_sirt_tv = astra.astra_dict('SIRT3D_CUDA')
    cfg_sirt_tv['ProjectionDataId'] = proj_id
    cfg_sirt_tv['ReconstructionDataId'] = rec_id_sirt_tv
    cfg_sirt_tv['option'] = {'MinConstraint': 0}
    alg_sirt_tv = astra.algorithm.create(cfg_sirt_tv)
    for _ in range(15):
        astra.algorithm.run(alg_sirt_tv, 10)
        curr = astra.data3d.get(rec_id_sirt_tv)
        reg = denoise_tv_chambolle(curr, weight=0.05)
        astra.data3d.store(rec_id_sirt_tv, reg)
    vol_w_tv = astra.data3d.get(rec_id_sirt_tv)
    if vol_w_tv.max() > vol_w_tv.min():
        vol_w_tv = (vol_w_tv - vol_w_tv.min()) / (vol_w_tv.max() - vol_w_tv.min())

    # Print all 4 rows for this phantom
    results = [
        ("Binary (Pure)", vol_bin_pure),
        ("Binary (+TV)", vol_bin_tv),
        ("Weighted (Pure SIRT)", vol_w_pure),
        ("Weighted (SIRT+TV)", vol_w_tv)
    ]
    for label, vol in results:
        t, d, j, vr, rmse = get_best_metrics(vol_gt, vol)
        print(f"{model_name:<10} | {label:<22} | {t:<6.2f} | {d:<6.2f} | {j:<6.2f} | {vr:<6.2f} | {rmse:<6.2f}")
    print("-" * 88)

    astra.algorithm.delete([alg_bp, alg_sirt_pure, alg_sirt_tv])
    astra.data3d.delete([clean_proj_id, proj_id, rec_id_bp, rec_id_sirt_pure, rec_id_sirt_tv])

astra.clear()