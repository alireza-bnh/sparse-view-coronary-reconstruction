import astra
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import sys
import time
import os
import nibabel as nib
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import threshold_otsu

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Ensure your .vtp phantom files are placed in a 'data' subdirectory
INPUT_FILE = 'data/Normal_3.vtp'
RES = 300              
NUM_VIEWS = 9          

# ==============================================================================
# 2. ADVANCED METRICS ENGINE
# ==============================================================================
def calculate_metrics(gt, rec, threshold):
    rec_bin = (rec > threshold).astype(bool)
    gt_bin = (gt > 0.5).astype(bool)
    
    intersection = np.sum(rec_bin & gt_bin)
    union = np.sum(rec_bin | gt_bin)
    rec_vol = np.sum(rec_bin)
    gt_vol = np.sum(gt_bin)
    
    # 1. Dice
    dice = 2.0 * intersection / (rec_vol + gt_vol) if (rec_vol + gt_vol) > 0 else 0.0
    
    # 2. Jaccard (IoU)
    jaccard = intersection / union if union > 0 else 0.0
    
    # 3. Volume Error (Ratio) - How bloated is it?
    vol_error = rec_vol / gt_vol if gt_vol > 0 else 0.0
    
    return dice, jaccard, vol_error

def calculate_rmse(vol1, vol2):
    return np.sqrt(np.mean((vol1 - vol2) ** 2))

# ==============================================================================
# 3. SIMULATION & RECONSTRUCTION
# ==============================================================================
print("🔹 Loading and Voxelizing...")
try:
    mesh = pv.read(INPUT_FILE)
    try: 
        grid = pv.ImageData(dimensions=(RES, RES, RES))
    except: 
        grid = pv.UniformGrid(dimensions=(RES, RES, RES))
    
    grid.spacing = (mesh.length/RES, mesh.length/RES, mesh.length/RES)
    
    print(f"   ---> CALCULATED VOXEL SIZE: {grid.spacing[0]:.3f} mm")
    print(f"   ---> CALCULATED FOV: {mesh.length:.1f}^3 mm^3")
    
    grid.origin = mesh.center - np.array(grid.spacing) * RES / 2
    selection = grid.select_enclosed_points(mesh, tolerance=0.0)
    vol_gt = selection.point_data['SelectedPoints'].reshape((RES, RES, RES), order='F').astype(np.float32)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit()

print(f"🔹 Simulating 9-View Sparse Data...")
vol_geom = astra.create_vol_geom(RES, RES, RES)
angles = np.linspace(0, np.pi, NUM_VIEWS, False)

# --- CLINICAL CONE-BEAM GEOMETRY ---
SOD = 750.0  # Source-to-Object Distance in mm
SDD = 1200.0 # Source-to-Detector Distance in mm
magnification = SDD / SOD  
det_pixel_size = 1.0 * magnification
ODD = SDD - SOD

proj_geom = astra.create_proj_geom('cone', det_pixel_size, det_pixel_size, RES, RES, angles, SOD, ODD)

# 1. Create clean sinogram
clean_proj_id, clean_sinogram = astra.create_sino3d_gpu(vol_gt, proj_geom, vol_geom)

# 2. Simulate Low-Dose Poisson Noise
print("🔹 Injecting Clinical Poisson Noise...")
I0 = 5000.0  
transmission = I0 * np.exp(-clean_sinogram)
noisy_transmission = np.random.poisson(transmission)
noisy_transmission[noisy_transmission == 0] = 1  
noisy_sinogram = -np.log(noisy_transmission / I0)

proj_id = astra.data3d.create('-sino', proj_geom, noisy_sinogram)

# --- Method 1: Discrete (BP) ---
print("🔸 Running Discrete BP (on noisy data)...")
start_time_bp = time.time()
rec_id_bp = astra.data3d.create('-vol', vol_geom)
cfg_bp = astra.astra_dict('BP3D_CUDA')
cfg_bp['ProjectionDataId'] = proj_id
cfg_bp['ReconstructionDataId'] = rec_id_bp
astra.algorithm.run(astra.algorithm.create(cfg_bp))
vol_bp = astra.data3d.get(rec_id_bp)
if vol_bp.max() > vol_bp.min(): 
    vol_bp = (vol_bp - vol_bp.min()) / (vol_bp.max() - vol_bp.min())
print(f"⏱️ BP Reconstruction Time: {time.time() - start_time_bp:.2f} seconds")

# --- Method 2: Weighted (SIRT + TV) ---
print("🔸 Running Weighted SIRT with Total Variation (TV)...")
start_time_sirt = time.time()

rec_id_sirt = astra.data3d.create('-vol', vol_geom)
cfg_sirt = astra.astra_dict('SIRT3D_CUDA')
cfg_sirt['ProjectionDataId'] = proj_id
cfg_sirt['ReconstructionDataId'] = rec_id_sirt
cfg_sirt['option'] = {'MinConstraint': 0}
alg_id_sirt = astra.algorithm.create(cfg_sirt)

iterations = 150
tv_frequency = 10  
lambda_tv = 0.05   # Set to 0.00 to run the Pure SIRT baseline
outer_loops = iterations // tv_frequency

for i in range(outer_loops):
    astra.algorithm.run(alg_id_sirt, tv_frequency)
    current_vol = astra.data3d.get(rec_id_sirt)
    
    if lambda_tv > 0.0:
        reg_vol = denoise_tv_chambolle(current_vol, weight=lambda_tv)
        astra.data3d.store(rec_id_sirt, reg_vol)
    
    current_iter = (i + 1) * tv_frequency
    print(f"   ... Completed {current_iter}/{iterations} iterations")

vol_sirt = astra.data3d.get(rec_id_sirt)
if vol_sirt.max() > vol_sirt.min():
    vol_sirt = (vol_sirt - vol_sirt.min()) / (vol_sirt.max() - vol_sirt.min())
    
astra.algorithm.delete(alg_id_sirt)
print(f"⏱️ SIRT(+TV) Total Reconstruction Time: {time.time() - start_time_sirt:.2f} seconds")

# ==============================================================================
# 4. SENSITIVITY ANALYSIS
# ==============================================================================
print("\n📊 Generating Sensitivity Graph...")
thresholds = np.linspace(0.01, 0.50, 20)
bp_dices = []
sirt_dices = []

for t in thresholds:
    d_bp, _, _ = calculate_metrics(vol_gt, vol_bp, t)
    d_sirt, _, _ = calculate_metrics(vol_gt, vol_sirt, t)
    bp_dices.append(d_bp)
    sirt_dices.append(d_sirt)

best_t_bp = thresholds[np.argmax(bp_dices)]
best_t_sirt = thresholds[np.argmax(sirt_dices)]
print(f"🌟 Optimal Threshold for Binary (BP): {best_t_bp:.2f}")
print(f"🌟 Optimal Threshold for Weighted (SIRT): {best_t_sirt:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(thresholds, bp_dices, 'r-o', label='Discrete (BP)', linewidth=2)
plt.plot(thresholds, sirt_dices, 'y-o', label='Weighted (SIRT+TV)', linewidth=2)
plt.title(f'Isosurface Sensitivity Analysis ({NUM_VIEWS} Views with Poisson Noise)')
plt.xlabel('Threshold Value')
plt.ylabel('Dice Similarity Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("sensitivity_graph.png")
print("   ✅ Graph saved as 'sensitivity_graph.png'")

# ==============================================================================
# 5. FINAL TABLE GENERATION (Using Global Otsu)
# ==============================================================================
print("\n📊 Calculating Objective Global Otsu Thresholds...")

# Let Otsu decide the perfect threshold objectively
otsu_thresh_bp = threshold_otsu(vol_bp)
otsu_thresh_sirt = threshold_otsu(vol_sirt)

# Calculate metrics using these objective thresholds
d_bp, j_bp, v_bp = calculate_metrics(vol_gt, vol_bp, otsu_thresh_bp)
d_sirt, j_sirt, v_sirt = calculate_metrics(vol_gt, vol_sirt, otsu_thresh_sirt)

rmse_bp = calculate_rmse(vol_gt, vol_bp)
rmse_sirt = calculate_rmse(vol_gt, vol_sirt)

print("\n" + "="*65)
print("📊 EXTENDED RESULTS TABLE (NOISY SCENARIO - OTSU IMPLEMENTED)")
print("="*65)
print(f"Computed BP Otsu Threshold:   {otsu_thresh_bp:.4f}")
print(f"Computed SIRT Otsu Threshold: {otsu_thresh_sirt:.4f}")
print("-" * 65)
print(f"{'Metric':<20} | {'Discrete (BP)':<20} | {'Weighted (SIRT+TV)':<20}")
print("-" * 65)
print(f"{'Otsu Dice':<20} | {d_bp:.4f}                | {d_sirt:.4f}")
print(f"{'Jaccard (IoU)':<20} | {j_bp:.4f}                | {j_sirt:.4f}")
print(f"{'Volume Ratio':<20} | {v_bp:.2f}x                 | {v_sirt:.2f}x")
print(f"{'RMSE':<20} | {rmse_bp:.4f}                | {rmse_sirt:.4f}")
print("="*65)

# ==============================================================================
# 6. EXPORT RAW VOLUMES AND MESHES (.nii and .vtp)
# ==============================================================================
output_dir = "reconstruction_outputs"
os.makedirs(output_dir, exist_ok=True)

# Export NIfTI (.nii)
affine = np.eye(4)
nib.save(nib.Nifti1Image(vol_bp.astype(np.float32), affine), os.path.join(output_dir, "Discrete_BP_raw.nii"))
nib.save(nib.Nifti1Image(vol_sirt.astype(np.float32), affine), os.path.join(output_dir, "Weighted_SIRT_raw.nii"))
print(f"✅ Saved NIfTI volumes to: {output_dir}")

# Export VTP (.vtp)
grid_bp_save = grid.copy()
grid_bp_save.point_data["values"] = vol_bp.flatten(order="F")
grid_bp_save.contour([otsu_thresh_bp], scalars="values").save(os.path.join(output_dir, "Discrete_BP_otsu.vtp"), binary=True)

grid_sirt_save = grid.copy()
grid_sirt_save.point_data["values"] = vol_sirt.flatten(order="F")
grid_sirt_save.contour([otsu_thresh_sirt], scalars="values").save(os.path.join(output_dir, "Weighted_SIRT_otsu.vtp"), binary=True)
print(f"✅ Saved VTP meshes to: {output_dir}")

# ==============================================================================
# 7. HISTOGRAM VISUALIZATION 
# ==============================================================================
print("\n📊 Generating Intensity Histograms...")

bp_data = vol_bp[vol_bp > 0.005].flatten()
sirt_data = vol_sirt[vol_sirt > 0.005].flatten()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Discrete BP Histogram (Log Scale)
axes[0].hist(bp_data, bins=100, color='crimson', alpha=0.7, log=True)
axes[0].axvline(otsu_thresh_bp, color='black', linestyle='dashed', linewidth=2, label=f'Otsu Threshold ({otsu_thresh_bp:.4f})')
axes[0].set_title(f'Discrete BP Intensity Distribution\n(Artifacts obscure the bimodal valley)')
axes[0].set_xlabel('Attenuation Density')
axes[0].set_ylabel('Voxel Count (Log Scale)')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# Plot 2: Weighted SIRT+TV Histogram (Log Scale)
axes[1].hist(sirt_data, bins=100, color='goldenrod', alpha=0.7, log=True)
axes[1].axvline(otsu_thresh_sirt, color='black', linestyle='dashed', linewidth=2, label=f'Otsu Threshold ({otsu_thresh_sirt:.4f})')
axes[1].set_title(f'Weighted SIRT+TV Intensity Distribution\n(Clear bimodal separation recovered)')
axes[1].set_xlabel('Attenuation Density')
axes[1].set_ylabel('Voxel Count (Log Scale)')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histogram_comparison_log.png'), dpi=300)
print(f"✅ Saved log histogram to: {output_dir}/histogram_comparison_log.png")
plt.show(block=False)

# ==============================================================================
# 7.5 VISUALIZE AND EXPORT THE 9 INPUT PROJECTIONS
# ==============================================================================
print("\n📊 Generating 9-View Projection Grid...")

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('Simulated 9-View C-Arm Projections\n(Low-Dose Poisson Noise, I0=5000)', fontsize=16, y=0.98)

for i, ax in enumerate(axes.flatten()):
    if i < NUM_VIEWS:
        proj_2d = noisy_sinogram[:, i, :]
        ax.imshow(proj_2d, cmap='gray')
        angle_deg = angles[i] * 180 / np.pi
        ax.set_title(f'View {i+1} ({angle_deg:.1f}°)', fontsize=12)
        ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '9_input_projections.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved 9-view projections to: {output_dir}/9_input_projections.png")
plt.show(block=False)

# ==============================================================================
# 7.6 2D CROSS-SECTIONAL PLANE (SLICE) VISUALIZATION
# ==============================================================================
print("\n📊 Generating 2D Cross-Sectional Planes...")

mid_z = RES // 2
slice_gt = vol_gt[:, :, mid_z]     
slice_bp = vol_bp[:, :, mid_z]
slice_sirt = vol_sirt[:, :, mid_z]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Cross-Sectional Plane Comparison (Z = {mid_z})', fontsize=16)

axes[0].imshow(slice_gt, cmap='gray')
axes[0].set_title('Ground Truth', fontsize=14)
axes[0].axis('off')

axes[1].imshow(slice_bp, cmap='gray', vmin=0, vmax=np.percentile(slice_bp, 99))
axes[1].set_title('Discrete (BP) - Severe Streak Artifacts', fontsize=14)
axes[1].axis('off')

axes[2].imshow(slice_sirt, cmap='gray', vmin=0, vmax=np.percentile(slice_sirt, 99))
axes[2].set_title('Weighted (SIRT+TV)', fontsize=14)
axes[2].axis('off')

plt.tight_layout()
slice_path = os.path.join(output_dir, 'cross_sectional_planes.png')
plt.savefig(slice_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved cross-sectional planes to: {slice_path}")
plt.show(block=False)

# ==============================================================================
# 8. 3D VISUALIZATION
# ==============================================================================
p = pv.Plotter(shape=(2, 3), window_size=[1800, 1200])

p.subplot(0, 0)
p.add_text("A. Ground Truth", font_size=12)
p.add_mesh(mesh, color="lightblue", opacity=0.3)
p.add_mesh(mesh, style='wireframe', color="black", opacity=0.1)

p.subplot(0, 1)
p.add_text(f"B. Discrete (BP) Isosurface\nOtsu: {otsu_thresh_bp:.4f} | Dice: {d_bp:.2f}", font_size=12)
grid_bp_iso = grid.copy()
grid_bp_iso.point_data['values'] = vol_bp.flatten(order='F')
iso_bp = grid_bp_iso.contour([otsu_thresh_bp], scalars='values')
p.add_mesh(iso_bp, color="crimson", opacity=0.6, name="bp_mesh")

def update_bp(thresh):
    p.subplot(0, 1)
    new_iso = grid_bp_iso.contour([thresh], scalars='values')
    p.add_mesh(new_iso, color="crimson", opacity=0.6, name="bp_mesh")

p.add_slider_widget(update_bp, [0.01, 0.50], value=otsu_thresh_bp, title="BP Threshold", pointa=(0.05, 0.9), pointb=(0.3, 0.9))

p.subplot(0, 2)
p.add_text(f"C. Weighted (SIRT+TV) Isosurface\nOtsu: {otsu_thresh_sirt:.4f} | Dice: {d_sirt:.2f}", font_size=12)
grid_sirt_iso = grid.copy()
grid_sirt_iso.point_data['values'] = vol_sirt.flatten(order='F')
iso_sirt = grid_sirt_iso.contour([otsu_thresh_sirt], scalars='values')
p.add_mesh(iso_sirt, color="gold", smooth_shading=True, name="sirt_mesh")

def update_sirt(thresh):
    p.subplot(0, 2)
    new_iso = grid_sirt_iso.contour([thresh], scalars='values')
    p.add_mesh(new_iso, color="gold", smooth_shading=True, name="sirt_mesh")

p.add_slider_widget(update_sirt, [0.01, 0.50], value=otsu_thresh_sirt, title="SIRT+TV Threshold", pointa=(0.6, 0.9), pointb=(0.85, 0.9))

p.subplot(1, 0)
p.add_text("D. Ground Truth (Repeated)", font_size=12)
p.add_mesh(mesh, color="lightblue", opacity=0.3)
p.add_mesh(mesh, style='wireframe', color="black", opacity=0.1)

p.subplot(1, 1)
p.add_text("E. Discrete (BP) - Raw Density Map\n(No Threshold)", font_size=12)
grid_bp_vol = grid.copy()
grid_bp_vol.point_data['values'] = vol_bp.flatten(order='F')
p.add_volume(grid_bp_vol, scalars='values', cmap="Reds", opacity="linear")

p.subplot(1, 2)
p.add_text("F. Weighted (SIRT+TV) - Raw Density Map\n(No Threshold)", font_size=12)
grid_sirt_vol = grid.copy()
grid_sirt_vol.point_data['values'] = vol_sirt.flatten(order='F')
p.add_volume(grid_sirt_vol, scalars='values', cmap="YlOrBr", opacity="linear")

print("✅ PyVista 3D Visualization Started.")
p.link_views()
p.show()

astra.clear()