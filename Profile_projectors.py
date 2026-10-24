import os
import time
import astra
import numpy as np
import pyvista as pv
from skimage.restoration import denoise_tv_chambolle

RES = 300
NUM_VIEWS = 9
SOD = 750.0
SDD = 1200.0
magnification = SDD / SOD
det_pixel_size = 1.0 * magnification
ODD = SDD - SOD

DATA_DIR = r"C:\Users\alire\Desktop\Data"
filepath = os.path.join(DATA_DIR, "solid_400001.vtp")

print("Loading phantom for timing benchmark...")
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

# Warm up GPU
sino_id, sino_data = astra.create_sino3d_gpu(vol_gt, proj_geom, vol_geom)
vol_id = astra.data3d.create('-vol', vol_geom, vol_gt)

N_RUNS = 50

# 1. Measure Standalone Forward Projection (FP3D_CUDA)
cfg_fp = astra.astra_dict('FP3D_CUDA')
cfg_fp['VolumeDataId'] = vol_id
cfg_fp['ProjectionDataId'] = sino_id
alg_fp = astra.algorithm.create(cfg_fp)

t0 = time.perf_counter()
for _ in range(N_RUNS):
    astra.algorithm.run(alg_fp)
t_fp_avg = (time.perf_counter() - t0) / N_RUNS

# 2. Measure Standalone Back-Projection (BP3D_CUDA)
rec_id = astra.data3d.create('-vol', vol_geom)
cfg_bp = astra.astra_dict('BP3D_CUDA')
cfg_bp['ProjectionDataId'] = sino_id
cfg_bp['ReconstructionDataId'] = rec_id
alg_bp = astra.algorithm.create(cfg_bp)

t0 = time.perf_counter()
for _ in range(N_RUNS):
    astra.algorithm.run(alg_bp)
t_bp_avg = (time.perf_counter() - t0) / N_RUNS

# 3. Measure 150 Iterations of Pure SIRT3D_CUDA (Combined FP + BP + Preconditioned Update)
rec_sirt_id = astra.data3d.create('-vol', vol_geom)
cfg_sirt = astra.astra_dict('SIRT3D_CUDA')
cfg_sirt['ProjectionDataId'] = sino_id
cfg_sirt['ReconstructionDataId'] = rec_sirt_id
cfg_sirt['option'] = {'MinConstraint': 0}
alg_sirt = astra.algorithm.create(cfg_sirt)

t0 = time.perf_counter()
astra.algorithm.run(alg_sirt, 150)
t_sirt_total = time.perf_counter() - t0
t_sirt_per_iter = t_sirt_total / 150.0

# 4. Measure 1 Pass of 3D TV Proximal Regularization (Chambolle)
curr_vol = astra.data3d.get(rec_sirt_id)
t0 = time.perf_counter()
_ = denoise_tv_chambolle(curr_vol, weight=0.05)
t_tv_single = time.perf_counter() - t0

print("\n" + "=" * 75)
print("⏱️ ASTRA GPU PROFILING RESULTS (300x300x300 Grid, 9 Views)")
print("=" * 75)
print(f"Average Forward-Projection (FP) time per iter : {t_fp_avg*1000:.2f} ms ({t_fp_avg:.4f} s)")
print(f"Average Back-Projection (BP) time per iter    : {t_bp_avg*1000:.2f} ms ({t_bp_avg:.4f} s)")
print(f"Sum of Standalone (FP + BP) per iter          : {(t_fp_avg + t_bp_avg)*1000:.2f} ms ({(t_fp_avg + t_bp_avg):.4f} s)")
print(f"Pure SIRT3D_CUDA time per iter (150 iters)    : {t_sirt_per_iter*1000:.2f} ms ({t_sirt_per_iter:.4f} s)")
print(f"Pure SIRT3D_CUDA total time (150 iters)       : {t_sirt_total:.2f} s")
print(f"3D TV Denoising (Chambolle) time per call     : {t_tv_single:.2f} s")
print("=" * 75)

astra.algorithm.delete([alg_fp, alg_bp, alg_sirt])
astra.data3d.delete([sino_id, vol_id, rec_id, rec_sirt_id])
astra.clear()