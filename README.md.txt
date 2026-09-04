# Sparse-View 3D Coronary Reconstruction (SIRT+TV)

This repository contains the Python implementation of the 3D coronary artery reconstruction framework utilizing the Simultaneous Iterative Reconstruction Technique (SIRT) coupled with Total Variation (TV) regularization. 

This code was developed to evaluate the geometric and morphological impact of discrete (Bresenham) versus weighted (Siddon) ray-tracing projectors on extreme sparse-view (9 projections) C-arm X-ray acquisitions under high Poisson noise conditions ($I_0 = 5000$).

## Features
* **Sparse-View Cone-Beam Simulation:** Generates 9-view 2D X-ray projections from 3D `.vtp` meshes using the ASTRA Toolbox.
* **Poisson Noise Modeling:** Simulates realistic clinical low-dose quantum mottle.
* **Dual Projector Evaluation:** Compares the discrete Bresenham algorithm against the sub-voxel accurate weighted Siddon algorithm.
* **Automated Segmentation:** Utilizes 3D global Otsu thresholding for objective volumetric binarization.
* **Metrics Engine:** Calculates Dice Similarity Coefficient (DSC), Jaccard Index (IoU), Volume Ratio (VR), and RMSE.
* **Visualization:** Outputs 2D cross-sectional planes, intensity histograms, and interactive 3D PyVista isosurfaces.

## Prerequisites

This pipeline requires Python 3.x and relies heavily on GPU acceleration via the ASTRA Toolbox. 

### Dependencies
* `numpy`
* `astra-toolbox` (Requires a CUDA-enabled NVIDIA GPU)
* `pyvista`
* `scikit-image`
* `matplotlib`
* `nibabel`

## Installation

It is recommended to run this code within a Conda virtual environment. 

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/sparse-view-coronary-reconstruction.git](https://github.com/yourusername/sparse-view-coronary-reconstruction.git)
   cd sparse-view-coronary-reconstruction