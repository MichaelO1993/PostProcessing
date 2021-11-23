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
  - Aligning cells by SmartAlign (Matlab-engine required)
  - Calculating L2-norm to exclude "bad" slices
  - Averaging spectrum image
  - PCA denoising and mapping of the fine structure (saving images)
  - Saving complete notebook for documentation (nbconvert package have to be installed) 

<br/><br/>

For using SmartAlign:
  - Download it as matlab version from: http://lewysjones.com/software/smart-align/
  - Copy these files in the *matlab* folder from this repository 
  - Insert before the option lines in the SmartAlign_Wrapper_1_6.m script following lines:  
	*addpath('./Smart_Align_1_6');  
	cd('./Matlab_files') 
	global number_stacks;  
	global pixels;  
	global image_stack_translation;  
	pixels = number_pixels;  
	image_stack_translation = cell([2 number_stacks]);  
	% Save translation matrix  
	global i_count;  
	global indexing;  
	indexing = 0;  
	i_count = cell([1 number_stacks]);  
	% Avoid error with trimming (if crop_core is off)  
	global y_trim;  
	global x_trim;  
	y_trim = 0;  
	x_trim = 0;*
  - Insert at the end of the script:  
	*close all*
  - Replace the downloaded movepixels_2d_double.m file with the movepixels_2d_double.m from this repository


<br/><br/>


If you use this script please cite:
