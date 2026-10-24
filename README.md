# Sparse-View 3D Coronary Reconstruction (SIRT+TV)

This repository contains the Python implementation of a 3D coronary artery reconstruction framework utilizing the Simultaneous Iterative Reconstruction Technique (SIRT) coupled with Total Variation (TV) regularization. 

This code was developed to evaluate the geometric and morphological impact of discrete (Bresenham) versus weighted (Siddon) ray-tracing projectors on sparse-view (9 projections) C-arm X-ray acquisitions under simulated Poisson-noise conditions ($I_0 = 5000$).

## Features
- **Sparse-View Cone-Beam Simulation:** Generates 9-view 2D X-ray projections from 3D `.vtp` volumes using the ASTRA Toolbox.
- **Poisson Noise Modeling:** Simulates realistic clinical quantum fluctuations ($I_0 = 5000$ photons per projection pixel).
- **Dual Projector Evaluation:** Compares the discrete Bresenham algorithm against the sub-voxel accurate weighted Siddon algorithm.
- **Unregularized & Regularized Solvers:** Evaluates both pure SIRT (unregularized) baseline matrices and SIRT+TV pipelines.
- **Dual-Track Segmentation:** Utilizes parameter-free 3D global Otsu thresholding alongside ground-truth-based IoU-maximized threshold optimization for rigorous geometric evaluation.
- **Metrics Engine:** Calculates the Dice-Sørensen Coefficient (DSC), Jaccard Index (IoU), Volume Ratio (VR), and Root Mean Square Error (RMSE).
- **Computational Profiling:** Micro-benchmarking for standalone GPU forward- and back-projection operations.

## Repository Structure

- `reconstruction_pipeline.py`: The primary pipeline for generating noisy projections, running the SIRT/SIRT+TV solvers, applying global Otsu thresholding, and exporting 2D/3D visualizations.
- `no_tv_sirt.py`: Evaluates and compares the unregularized (pure SIRT) baseline matrices against the TV-regularized pipelines across all anatomical models.
- `profile_projectors.py`: Standalone ASTRA GPU forward- and back-projection micro-benchmarking over 150 iterations.
- `SparseViewIoU.py`: Computes the optimal segmentation threshold via a 256-step binarization sweep, maximizing the Jaccard index (IoU) between the reconstructed and ground-truth volumes.
- `SparseViewIoU.txt`: The raw thresholding sensitivity outputs used to generate the data tables and plots in the manuscript.

## Prerequisites

This pipeline requires Python 3.x and relies heavily on GPU acceleration via the ASTRA Toolbox. 

### Dependencies
- `numpy`
- `astra-toolbox` (Requires a CUDA-enabled NVIDIA GPU)
- `pyvista`
- `scikit-image`
- `matplotlib`
- `nibabel`
- `itk` (Required for the `SparseViewIoU.py` threshold optimization script)

## Installation

It is recommended to run this code within a Conda virtual environment. 

1. Clone the repository:
   ```bash
   git clone [https://github.com/alireza-bnh/sparse-view-coronary-reconstruction.git](https://github.com/alireza-bnh/sparse-view-coronary-reconstruction.git)
   cd sparse-view-coronary-reconstruction