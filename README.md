# Post_processing

This repsoitory contains a step-by-step jupyter notebook script for post-processing EELS spectrum images with reference image according to the paper:

The script enables fast mapping of fine structure by multi-frame averaging and PCA denoising including control mechanism by clustering the spectrum image and calculating L2-norm.

The script is structured in:
  - Loading required packages, spectrum image and reference image
  - Crop spectra to region of interest (to reduce computational cost)
  - Unsupervised clustering of the spectrum image (according to https://doi.org/10.1016/j.ultramic.2021.113314)
  - Determining atom positions by Atomap for drift correction
  - Linear drift correction
  - Selecting atoms, which are used as center for the cropped SIs
  - Aligning cells by SmartAlign-algorithm
  - Calculating L2-norm to exclude "bad" slices
  - Averaging spectrum image
  - PCA denoising and mapping of the fine structure (saving images, saving denoised spectrum image for GMS)
  - Saving complete notebook for documentation (nbconvert package have to be installed) 


If you use this script please cite:
