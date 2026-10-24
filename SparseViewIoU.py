# =============================================================================
# Sparse-View Reconstruction Evaluation
#
# This code computes optimal grayscale thresholds and similarity measures
# between the ground-truth model and models reconstructed from sparse views.
#
# Author: Piotr M. Szczypiński
#
# MIT License
#
# Copyright (c) 2026 Piotr M. Szczypiński
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# =============================================================================

import itk
import matplotlib.pyplot as plt
import numpy as np
import math

# ------------------------------------------------------------------
# Parametry
# ------------------------------------------------------------------

Dimension = 3
PixelTypeReference = itk.F #itk.UC      # obraz binarny
PixelTypeReconstruction = itk.F  # obraz szaroskalowy

ReferenceImageType = itk.Image[PixelTypeReference, Dimension]
ReconstructionImageType = itk.Image[PixelTypeReconstruction, Dimension]
BinaryImageType = itk.Image[itk.UC, Dimension]

reference_file = "gt_filename"
reconstruction_file = "reconst_filename.nii"

def calculate_metrics(best_threshold, max_val, reconstruction, binary_reference):

    thresholder = itk.BinaryThresholdImageFilter[
        ReconstructionImageType,
        BinaryImageType
    ].New()

    thresholder.SetInput(reconstruction)
    thresholder.SetLowerThreshold(best_threshold)
    thresholder.SetUpperThreshold(max_val)
    thresholder.SetInsideValue(1)
    thresholder.SetOutsideValue(0)
    thresholder.Update()

    binary_reconstruction = thresholder.GetOutput()

    and_filter = itk.AndImageFilter[
        BinaryImageType,
        BinaryImageType,
        BinaryImageType
    ].New()
    stats_intersection = itk.StatisticsImageFilter[BinaryImageType].New()

    and_filter.SetInput1(binary_reference)
    and_filter.SetInput2(binary_reconstruction)
    and_filter.Update()
    stats_intersection.SetInput(and_filter.GetOutput())
    stats_intersection.Update()
    count_tp = stats_intersection.GetSum()

    not_filter = itk.NotImageFilter[
        BinaryImageType,
        BinaryImageType
    ].New()
    not_filter.SetInput(binary_reconstruction)
    not_filter.Update()

    and_filter.SetInput1(binary_reference)
    and_filter.SetInput2(not_filter.GetOutput())
    and_filter.Update()
    stats_intersection.SetInput(and_filter.GetOutput())
    stats_intersection.Update()
    count_fn = stats_intersection.GetSum()

    not_filter = itk.NotImageFilter[
        BinaryImageType,
        BinaryImageType
    ].New()
    not_filter.SetInput(binary_reference)
    not_filter.Update()

    and_filter.SetInput1(not_filter.GetOutput())
    and_filter.SetInput2(binary_reconstruction)
    and_filter.Update()
    stats_intersection.SetInput(and_filter.GetOutput())
    stats_intersection.Update()
    count_fp = stats_intersection.GetSum()

    region = binary_reference.GetLargestPossibleRegion()
    count_all = region.GetNumberOfPixels()
    count_union = count_fn + count_fp + count_tp
    count_tn = count_all - count_union

    cast_binary_to_float = itk.CastImageFilter[BinaryImageType, ReferenceImageType].New()
    cast_binary_to_float.SetInput(binary_reference)
    cast_binary_to_float.Update()
    #    cast.GetOutput()

    subtract_image_filter = itk.SubtractImageFilter[
        ReconstructionImageType,
        ReconstructionImageType,
        ReconstructionImageType
    ].New()
    diff = itk.subtract_image_filter(reconstruction, cast_binary_to_float.GetOutput())

    stats_float = itk.StatisticsImageFilter[ReconstructionImageType].New()
    stats_float.SetInput(diff)
    stats_float.Update()
    mean = stats_float.GetMean()
    mse = stats_float.GetVariance() + mean*mean
    rmse = math.sqrt(mse)

    vr = (count_tp+count_fp) / (count_tp+count_fn) if (count_tp+count_fn) > 0 else 0.0
    iou = count_tp / count_union if count_union > 0 else 0.0
    dsc = 2*count_tp / (count_union+count_tp) if count_union > 0 else 0.0

    return iou, dsc, rmse, vr



def calculate_funcion():
    threshold_values = []
    iou_values = []
    print(reconstruction_file)
    print(reference_file)

    # ------------------------------------------------------------------
    # Wczytanie obrazów
    # ------------------------------------------------------------------
    referenceo = itk.imread(reference_file, PixelTypeReference)
    reconstruction = itk.imread(reconstruction_file, PixelTypeReconstruction)
    reconstruction.SetOrigin(referenceo.GetOrigin())
    reconstruction.SetSpacing(referenceo.GetSpacing())
    reconstruction.SetDirection(referenceo.GetDirection())

    thresholderr = itk.BinaryThresholdImageFilter[
        ReferenceImageType,
        BinaryImageType
    ].New()

    thresholderr.SetInput(referenceo)
    thresholderr.SetLowerThreshold(0.5)
    thresholderr.SetUpperThreshold(255)
    thresholderr.SetInsideValue(1)
    thresholderr.SetOutsideValue(0)
    thresholderr.Update()
    binary_reference = thresholderr.GetOutput()

    # ------------------------------------------------------------------
    # Wyznaczenie min i max jasności obrazu rekonstrukcji
    # ------------------------------------------------------------------
    stats = itk.StatisticsImageFilter[ReconstructionImageType].New()
    stats.SetInput(reconstruction)
    stats.Update()

    min_val = stats.GetMinimum()
    max_val = stats.GetMaximum()

    print(f"Min intensity = {min_val}")
    print(f"Max intensity = {max_val}")
    print()

    # ------------------------------------------------------------------
    # 256 progów
    # ------------------------------------------------------------------
    for i in range(256):

        threshold = min_val + i * (max_val - min_val) / 255.0

        # Binaryzacja rekonstrukcji
        thresholder = itk.BinaryThresholdImageFilter[
            ReconstructionImageType,
            BinaryImageType
        ].New()

        thresholder.SetInput(reconstruction)
        thresholder.SetLowerThreshold(threshold)
        thresholder.SetUpperThreshold(max_val)
        thresholder.SetInsideValue(1)
        thresholder.SetOutsideValue(0)
        thresholder.Update()

        binary_reconstruction = thresholder.GetOutput()

        and_filter = itk.AndImageFilter[
            BinaryImageType,
            BinaryImageType,
            BinaryImageType
        ].New()
        stats_intersection = itk.StatisticsImageFilter[BinaryImageType].New()

        and_filter.SetInput1(binary_reference)
        and_filter.SetInput2(binary_reconstruction)
        and_filter.Update()
        stats_intersection.SetInput(and_filter.GetOutput())
        stats_intersection.Update()
        count_tp = stats_intersection.GetSum()

        not_filter = itk.NotImageFilter[
            BinaryImageType,
            BinaryImageType
        ].New()
        not_filter.SetInput(binary_reconstruction)
        not_filter.Update()

        and_filter.SetInput1(binary_reference)
        and_filter.SetInput2(not_filter.GetOutput())
        and_filter.Update()
        stats_intersection.SetInput(and_filter.GetOutput())
        stats_intersection.Update()
        count_fn = stats_intersection.GetSum()

        not_filter = itk.NotImageFilter[
            BinaryImageType,
            BinaryImageType
        ].New()
        not_filter.SetInput(binary_reference)
        not_filter.Update()

        and_filter.SetInput1(not_filter.GetOutput())
        and_filter.SetInput2(binary_reconstruction)
        and_filter.Update()
        stats_intersection.SetInput(and_filter.GetOutput())
        stats_intersection.Update()
        count_fp = stats_intersection.GetSum()

        region = binary_reference.GetLargestPossibleRegion()
        count_union = count_fn + count_fp + count_tp
        iou = count_tp / count_union if count_union > 0 else 0.0
        dsc = 2*count_tp / (count_union+count_tp) if count_union > 0 else 0.0
        print(f"{i}\t{threshold:.6f}\t{iou:.6f}\t{dsc:.6f}")
        threshold_values.append(threshold)
        iou_values.append(iou)

    # ------------------------------------------------------------------
    # Wykres IoU(threshold)
    # ------------------------------------------------------------------
    best_idx = iou_values.index(max(iou_values))
    best_threshold = threshold_values[best_idx]
    best_iou = iou_values[best_idx]
    iou, dsc, rmse, vr = calculate_metrics(best_threshold, max_val, reconstruction, binary_reference)
    print(f"--------------------------------------------------")
    print(reconstruction_file)
    print(f"i\tthreshold\tiou\tdsc\tvr\trmse")
    print(f"{best_idx}\t{best_threshold:.6f}\t{iou:.6f}\t{dsc:.6f}\t{vr:.6f}\t{rmse:.6f}")
    #scaled = itk.multiply_image_filter(reconstruction, constant=255.0)
    # imageUC = itk.cast_image_filter(scaled, ttype=(type(scaled), itk.Image[itk.UC, 3]) )
    otsu = itk.OtsuThresholdImageFilter.New(Input=reconstruction)
    otsu.SetInsideValue(255)
    otsu.SetOutsideValue(0)
    otsu.SetNumberOfHistogramBins(256)
    otsu.Update()
    otsu_threshold = otsu.GetThreshold()
    otsu_index = int(otsu_threshold * 256)
    iou, dsc, rmse, vr = calculate_metrics(otsu_threshold, max_val, reconstruction, binary_reference)
    print(f"{otsu_index}\t{otsu_threshold:.6f}\t{iou:.6f}\t{dsc:.6f}\t{vr:.6f}\t{rmse:.6f}")
    print(f"")
    print(f"==================================================")
    print(f"")
    return

reference_file = "Model1_GroundTruth.nii"
reconstruction_file = "Model1_Bresenham_SIRT.nii"
calculate_funcion()
reconstruction_file = "Model1_Bresenham_SIRT_TV.nii"
calculate_funcion()
reconstruction_file = "Model1_Siddon_SIRT.nii"
calculate_funcion()
reconstruction_file = "Model1_Siddon_SIRT_TV.nii"
calculate_funcion()

reference_file = "Model2_GroundTruth.nii"
reconstruction_file = "Model2_Bresenham_SIRT.nii"
calculate_funcion()
reconstruction_file = "Model2_Bresenham_SIRT_TV.nii"
calculate_funcion()
reconstruction_file = "Model2_Siddon_SIRT.nii"
calculate_funcion()
reconstruction_file = "Model2_Siddon_SIRT_TV.nii"
calculate_funcion()

reference_file = "Model3_GroundTruth.nii"
reconstruction_file = "Model3_Bresenham_SIRT.nii"
calculate_funcion()
reconstruction_file = "Model3_Bresenham_SIRT_TV.nii"
calculate_funcion()
reconstruction_file = "Model3_Siddon_SIRT.nii"
calculate_funcion()
reconstruction_file = "Model3_Siddon_SIRT_TV.nii"
calculate_funcion()

