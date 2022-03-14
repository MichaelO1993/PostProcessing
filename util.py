# Import required packages

from ipywidgets import *
import hyperspy.api as hs
import os
import subprocess
import numpy as np
import atomap.api as am
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import matplotlib.animation as animation
from tkinter.filedialog import askopenfilename
import math
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import pandas as pd
import random
from tqdm.notebook import tqdm_notebook
from PIL import Image
import sys
import time
import copy
from lmfit import Model, Parameters
from numpy.lib.stride_tricks import as_strided
import cv2
import scipy.ndimage as ndimage
from scipy.fft import fftn, ifftn, fftfreq


class PostProcessor:
    def __init__(self):
        self.s_EELS = None
        self.s_darkfield = None
        self.possion_noise = None
        self.roi_crop = None
        self.optics_model = None
        self.colors = None
        self.newcmp = None
        self.labels_shaped = None
        self.df = None
        self.newcmp = None
        self.n_denoise_cluster = None
        self.label_eps = None
        self.atom_positions = None
        self.s_low = None
        self.sublattice_A = None
        self.darkfield_drift = None
        self.labels_shaped_drift = None
        self.atom_position_drift = None
        self.EELS_data_drift = None
        self.index_image_stack = None
        self.dark_field_stack = None
        self.EELS = None
        self.EELS_sum_aligned = None
        self.darkfield_aligned = None
        self.L2_norm = None
        self.roi_signal = None
        self.roi_background = None
        
    # Load data
    def load_data(self, poisson_noise = True):
        self.poisson_noise = poisson_noise
        # Load spectrum image
        filename_EELS = askopenfilename(title='Please select spectrum image.')
        # Load HAADF image
        filename_darkfield = askopenfilename(title='Please select reference image.')
        
        # Path for saving results
        self.path_EELS = os.path.dirname(filename_EELS)
        
        # Load spectrum image
        self.s_EELS = hs.load(filename_EELS)
        self.s_EELS.set_signal_type("EELS")
        print(f'Loaded SI from {filename_EELS}')

        # Load dark field image
        self.s_darkfield = hs.load(filename_darkfield)
        self.s_darkfield.metadata.General.name = 'dark_field_image'
        print(f'Loaded reference image from {filename_darkfield}')
        
    # Interactive crop region selection 
    def select_crop(self):
        # Average spectra --> new EELS-set
        s_crop = hs.signals.Signal1D(np.sum(self.s_EELS.data,axis=(0,1)))
        # Define axes manager
        s_crop.axes_manager[0].name = self.s_EELS.axes_manager[2].name
        s_crop.axes_manager[0].scale = self.s_EELS.axes_manager[2].scale
        s_crop.axes_manager[0].offset = self.s_EELS.axes_manager[2].offset
        # Initialize crop region
        self.roi_crop = hs.roi.SpanROI(left=1.05*s_crop.axes_manager[0].offset, right= 0.95 * (s_crop.axes_manager[0].offset + s_crop.axes_manager[0].scale * s_crop.axes_manager[0].size))

        # Interactive plot
        s_crop.plot()
        plt.gca().set_title("")
        textstr =  'Crop spetrum (use green area)'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        text = plt.gcf().text(0.5, 0.98, textstr, fontsize=8, horizontalalignment='center', verticalalignment='top', bbox=props)
        self.roi_crop.interactive(s_crop, color='green')
       
    # Crop EELS data
    def crop(self, roi_crop_in = None):
        if roi_crop_in == None:
            left_value = self.roi_crop.left
            right_value = self.roi_crop.right
        else:
            left_value = np.amin(roi_crop_in)
            right_value = np.amax(roi_crop_in)
        # Crop EELS data
        self.s_EELS.crop(2,start = left_value, end = right_value)
        print(f'Crop Region from {left_value:.2f} eV to {right_value:.2f} eV')
     
    def decompose_hs(self, EELS):
        # for poisson no negative values allowed
        if self.poisson_noise:
            EELS.data[EELS.data < 0] = 0
            #EELS -= np.amin(s_EELS.data) # shift upwards
            
        # Decompose spectra
        EELS.decomposition(normalize_poissonian_noise=self.poisson_noise)
        
        return EELS
        
    def cluster_pca(self, n_plot = 60):
        # Decompose
        self.s_EELS = self.decompose_hs(self.s_EELS)
        
        # Plot scree plot
        ax = self.s_EELS.plot_explained_variance_ratio(n=n_plot,vline=False)
        ax.set_title('Scree Plot')
        
    def clustering_init(self, n_denoise_cluster = 10, perplexity_tsne = 30):
        self.n_denoise_cluster = n_denoise_cluster
    
        # Extracting the factor-matrix from the hyperspy framework for further clustering
        factor_matrix = self.s_EELS.get_decomposition_loadings()

        # Select only factors, which are used for reconstruction
        factor_tsne = factor_matrix.data[0:self.n_denoise_cluster+1,:,:] 
        factor_reshaped = factor_tsne.reshape(factor_tsne.data.shape[0],-1)

        # Reduce dimensionality by t-sne with sklearn
        tsne = TSNE(perplexity=perplexity_tsne, random_state=0, init='pca', method='barnes_hut') # method... 'excact' or ’barnes_hut’ --> faster
        tsne_results = tsne.fit_transform(factor_reshaped.transpose())


        # Plot the results from the OPTICS algorithm without assigning to clusters

        # Numpy to Pandas Dataframe
        column = []
        for i in range(0,tsne_results.shape[1]):
            column = np.append(column,"tsne-"+str(i))
        self.df = pd.DataFrame(data=tsne_results, columns=column)   

        # Separation with OPTICS
        self.optics_model = OPTICS(cluster_method = "dbscan", min_samples = 2)
        self.optics_model.fit(self.df) 


        # Plot reachability
        fig, ax = plt.subplots()
        ax.plot(self.optics_model.reachability_[self.optics_model.ordering_], 'k.', alpha = 0.3) 
        ax.set_ylabel('Reachability Distance') 
        ax.set_title('Reachability Plot') 
        
    def clustering(self, eps_optics = 1, cmap = 'tab10', shuffle = True):
        # Cluster spectra with given eps
        self.labels_eps = cluster_optics_dbscan(reachability=self.optics_model.reachability_,
                                           core_distances=self.optics_model.core_distances_,
                                           ordering=self.optics_model.ordering_, eps=eps_optics)

        # Storing the reachability distance of each point 
        reachability = self.optics_model.reachability_[self.optics_model.ordering_] 


        # Creating a numpy array with numbers at equal spaces till the specified range for colouring
        space = np.arange(len(self.df)) 

        # Storing the cluster labels of each point 
        labels = self.labels_eps[self.optics_model.ordering_]
        
        
        # If smaller than 0, no clusters are found
        if np.amax(labels) < 0:
            n_samples = 0
        else:
            n_samples = np.amax(labels)

        #Creating colors from specific color map (colormap can be changed)
        cmap = plt.get_cmap(cmap, n_samples+1) # n_samples+1 to consider the no-cluster
        self.colors = [cmap(i) for i in np.linspace(0, 1, n_samples+1)]
        if shuffle:
            random.shuffle(self.colors) # shuffle color for better visualization
        self.colors.insert(0,[0, 0, 0, 1.0]) # add black for the no-cluster

        # Plotting results (coloured reachability and coloured t-SNE plot)
        fig1, (ax11, ax12) = plt.subplots(1, 2)
        fig1.suptitle('OPTICS result')

        for Class, colour  in zip(range(-1, n_samples), self.colors): 
            # Coloured reachability-distance plot 
            Xk_r = space[labels == Class] 
            Rk = reachability[labels == Class] 

            # Coloured OPTICS Clustering 
            Xk_o = self.df[self.labels_eps == Class]   

            if Class == -1:
                ax11.plot(Xk_r, Rk, color=colour, alpha = 0.3,linestyle="",marker=".")
                ax12.plot(Xk_o.iloc[:, 0], Xk_o.iloc[:, 1], color=colour, alpha = 0.3,linestyle="",marker=".")  
            else:
                ax11.plot(Xk_r, Rk, color=colour, alpha = 0.5,linestyle="",marker=".") 
                ax12.plot(Xk_o.iloc[:, 0], Xk_o.iloc[:, 1], color=colour, alpha = 0.5,linestyle="",marker=".")  

        ax11.set_ylabel('Reachability Distance') 
        ax11.set_title('Reachability Plot') 

        ax12.set_title('OPTICS Clustering') 
        
        # Plot coloured map with corresponding color of the clustering
        
        # Reshape labels to image shape
        self.labels_shaped = np.reshape(self.labels_eps, (self.s_EELS.data.shape[0], self.s_EELS.data.shape[1]))
        
        # Create new colormap to match the color from before
        self.newcmp = ListedColormap(self.colors)

        # Plotting clustered SI
        fig2, (ax21, ax22) = plt.subplots(1, 2, sharex=True, sharey=True)

        ax21.matshow(self.labels_shaped,cmap=self.newcmp)
        ax21.axis('off')
        ax21.set_title('Clustered SI')

        ax22.matshow(self.s_darkfield.data)
        ax22.axis('off')
        ax22.set_title('Dark field image')

        fig2.tight_layout()
        
    def clustering_spectra(self, k_min = 500):
        # Average over all pixels from the denoised SI by PCA
        sc = self.s_EELS.get_decomposition_model(self.n_denoise_cluster)
        sc_averaged = np.mean(sc.data,axis=(0,1))
        sc_averaged = sc_averaged[:,np.newaxis]

        # Average over all pixels from the original SI
        s_averaged = np.mean(self.s_EELS.data,axis=(0,1))
        s_averaged = s_averaged[:,np.newaxis]

        # First element is the over all averaged spectra in black
        label_color = [-1]

        # Loop over all labels
        labels_unique = np.unique(self.labels_shaped)

        for idx, label_unique in enumerate(labels_unique):

            # Loop only over labels, which have mor then k_min pixels and are not the no-cluster
            if len(self.labels_shaped[self.labels_shaped == label_unique]) >= k_min and label_unique != -1:

                # Add label to the color list
                label_color = np.append(label_color,label_unique)

                # Average all spectra with the same label (PCA denoised)
                sc_vec = np.reshape(sc.data, ( self.labels_eps.shape[0], sc.data.shape[2]))
                a = np.mean(sc_vec[self.labels_eps == label_unique],axis=0)
                a = a[:,np.newaxis]
                sc_averaged = np.append(sc_averaged,a,axis=1)

                # Average all spectra with the same label (original)
                s_vec = np.reshape(self.s_EELS.data, ( self.labels_eps.shape[0], self.s_EELS.data.shape[2]))
                a = np.mean(s_vec[self.labels_eps == label_unique],axis=0)
                a = a[:,np.newaxis]
                s_averaged = np.append(s_averaged,a,axis=1)

        print(f'Number of clusters for plotting: {(sc_averaged.shape[1]-1)}')
        
        # Extract the energy axes for a correct x-axes
        energy_axes = np.linspace(self.s_EELS.axes_manager["Energy loss"].offset,self.s_EELS.axes_manager["Energy loss"].offset+self.s_EELS.axes_manager["Energy loss"].scale*self.s_EELS.axes_manager["Energy loss"].size,self.s_EELS.axes_manager["Energy loss"].size)

        # Plotting averaged spectra (black in the spectra)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.suptitle('Averaged EELS with same clusters')

        for i in range(0,sc_averaged.shape[1]):
            # Select correct color
            if label_color[i] == -1: # aveaged spectra in black
                colour = self.colors[0]
            else:
                colour = self.colors[int(label_color[i])+1]
            ax1.plot(energy_axes, sc_averaged[:,i], color = colour)
            ax2.plot(energy_axes, s_averaged[:,i], color = colour)

        ax1.set_title('PCA denoised') 
        ax1.set_xlabel('Energy loss / eV')
        ax1.set_ylabel('Intensity / a.u.')
        ax1.axes.yaxis.set_ticklabels([])
        ax2.set_title('Original') 
        ax2.set_xlabel('Energy loss / eV')
        ax2.set_ylabel('Intensity / a.u.')
        
    def atom_positioning(self, s_low = 4):
        self.s_low = s_low
        
        # Get atom position
        self.atom_positions = am.get_atom_positions(self.s_darkfield, separation=s_low, pca=True, subtract_background=True)
       
    def atom_positioning_single(self, s_darkfield, atom_positions, stacking = False):
        # Add or remove indvidual atoms - close window if you finished
        if stacking:
            self.atom_position_stacking = am.add_atoms_with_gui(s_darkfield, atom_positions,distance_threshold=self.s_low)
        else:
            self.atom_positions = am.add_atoms_with_gui(s_darkfield, atom_positions,distance_threshold=self.s_low)
        
        plt.gca().set_title("")
        textstr =  'Click to add/remove atoms. Close figure after selection!'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        text = plt.gcf().text(0.5, 0.98, textstr, fontsize=8, horizontalalignment='center', verticalalignment='top', bbox=props)
        
    def refine_positions(self):
        # Refine atom position by center of mass and 2d gaussians
        self.sublattice_A = am.Sublattice(self.atom_positions, image=self.s_darkfield.data, color='r')
        self.sublattice_A.find_nearest_neighbors()
        self.sublattice_A.refine_atom_positions_using_center_of_mass()
        self.sublattice_A.refine_atom_positions_using_2d_gaussian() # Warnings can be ignored

        # Generate atom lattice
        atom_lattice = am.Atom_Lattice(image=self.s_darkfield.data, sublattice_list=[self.sublattice_A])

        # Plot refined atom positions

        # Get atom positions for plotting
        x_px_plot = atom_lattice.x_position
        y_px_plot = atom_lattice.y_position

        fig, ax = plt.subplots()
        ax.imshow(self.s_darkfield.data)
        scatter_atom = ax.scatter(x_px_plot,y_px_plot, c='r', s=4)
        ax.set_title('Refined atom positions')


    def lattice_calc(self):
        # Construct planes and navigated through it
        self.sublattice_A.construct_zone_axes()
        self.sublattice_A.plot_planes()

    # Transformation matrix
    def transform_matrix(self, alpha, slope_y, sx, sy):
        slope_x = 0
        
        # Matrix, which rotates the crystal, shears and scales it
        matrix = np.array([
        [sx*(np.cos(alpha) + slope_y*np.sin(alpha)) , sy * (slope_x*np.cos(alpha) + np.sin(alpha)), 0],
        [sx * (-np.sin(alpha) + slope_y*np.cos(alpha)),  sy * (-slope_x*np.sin(alpha) + np.cos(alpha)), 0],
        [0,0, 1]
        ])

        return matrix
        
    def drift_correction(self, vector_ind, crystal_x = [1,0], crystal_y = [0,1], scaling = False):
        
        # Calculate crystal vectors
        measured_x = self.sublattice_A.zones_axis_average_distances[vector_ind[0]]
        measured_y = self.sublattice_A.zones_axis_average_distances[vector_ind[1]]

        # Get dark field image and spectrum image as numpy array
        dark_field_image = self.s_darkfield.data[:,:,np.newaxis]
        EELS_data = self.s_EELS.data
        labels_shaped_3d = self.labels_shaped[:,:,np.newaxis]

        # Transform atom position from atomap
        x_px_A = self.sublattice_A.x_position
        y_px_A = self.sublattice_A.y_position
        atom_position_a = np.transpose(np.column_stack((x_px_A,y_px_A)))

        # Angle x-measured to x-crystal
        alpha = -np.arccos(np.dot(measured_x/np.linalg.norm(measured_x), crystal_x/np.linalg.norm(crystal_x)))
        
        # Shear y
        slope_y = measured_y[0]/measured_y[1] - crystal_y[0]/crystal_y[1] + np.tan(alpha)
        
        if scaling:
            # Calculate stretch in y-vector (corrected in x-axis)
            ratio = np.linalg.norm(crystal_y)/np.linalg.norm(crystal_x)*(np.linalg.norm(measured_x)/np.linalg.norm(measured_y))
            # Only stretch smaller axis, other axis is not scaled
            crystal_xnorm = np.linalg.norm(crystal_x)
            crystal_ynorm = np.linalg.norm(crystal_y)
            if ratio > 1:
                sy = (crystal_x[0]/crystal_xnorm + crystal_y[0]/crystal_ynorm)*ratio
                sx = 1
            elif ratio < 1:
                sx = (crystal_x[1]/crystal_xnorm + crystal_y[1]/crystal_ynorm)/ratio
                sy = 1
            else:
                sx = 1
                sy = 1
        else:
            sx = 1
            sy = 1
        
        
                # Calculate transformation matrix
        transformation_matrix = self.transform_matrix(alpha, slope_y, sx, sy)


        # Define offset for the transformation
        offset = (-int(dark_field_image.shape[0]/2), -int(dark_field_image.shape[1]/2), 0)

        # Transform images
        dark_field_image_trans = ndimage.affine_transform(dark_field_image, transformation_matrix, offset=offset, order=3, mode='constant',
                                                          cval=np.NaN, output_shape = (int(2*dark_field_image.shape[0]), 
                                                                                       int(2*dark_field_image.shape[1]), 
                                                                                       dark_field_image.shape[2]), prefilter=True)
        EELS_data = ndimage.affine_transform(EELS_data, transformation_matrix, offset=offset, order=3, mode='constant',
                                                          cval=np.NaN, output_shape = (int(2*dark_field_image.shape[0]), 
                                                                                       int(2*dark_field_image.shape[1]), 
                                                                                       EELS_data.shape[2]), prefilter=True)
        labels_shaped_transformed = ndimage.affine_transform(labels_shaped_3d.astype(float), transformation_matrix, offset=offset, order=0,
                                                             mode='constant',cval=np.NaN, 
                                                             output_shape = (int(2*labels_shaped_3d.shape[0]),
                                                                             int(2*labels_shaped_3d.shape[1]),
                                                                             labels_shaped_3d.shape[2]), prefilter=True)        


        # Identify the translation required for the atom positions
        for i in range(0,dark_field_image_trans.shape[0]):
            k_y = i
            if ~np.isnan(dark_field_image_trans[i,:]).all(axis=(0,1)):
                break
        for i in range(0,dark_field_image_trans.shape[1]):
            k_x = i
            if ~np.isnan(dark_field_image_trans[:,i]).all(axis=(0,1)):
                break

        # Remove nan-margin        
        dark_field_image_trans = dark_field_image_trans[:, ~np.isnan(dark_field_image_trans).all(axis=(0,2))]
        dark_field_image = dark_field_image_trans[~np.isnan(dark_field_image_trans).all(axis=(1,2)),:]   

        EELS_data = EELS_data[:, ~np.isnan(EELS_data).all(axis=(0,2))]
        EELS_data = EELS_data[~np.isnan(EELS_data).all(axis=(1,2)),:]   

        labels_shaped_transformed = labels_shaped_transformed[:, ~np.isnan(labels_shaped_transformed).all(axis=(0,2))]
        labels_shaped_transformed = labels_shaped_transformed[~np.isnan(labels_shaped_transformed).all(axis=(1,2)),:]   

        # 3D points (affine transformation)
        atom_position_a = np.r_[ atom_position_a, np.ones(atom_position_a.shape[1])[np.newaxis,:] ]

        # Transformation matrix for points
        transformation_matrix_pts = self.transform_matrix(-alpha, -slope_y, 1/sy, 1/sx)

        # Use same offset as at the image trasnformation --> add it to the points
        atom_position_a = atom_position_a + np.tile(np.array([-offset[1], -offset[0], 0])[:,np.newaxis],(1,atom_position_a.shape[1]))

        # Transform
        atom_position_a_transformed = np.matmul(transformation_matrix_pts.T, atom_position_a)

        # 2D points
        atom_position_a_transformed = atom_position_a_transformed[0:2,:]

        # Remove same margin as at the image
        atom_position_a_transformed = atom_position_a_transformed - np.tile(np.array([k_x, k_y])[:,np.newaxis],
                                                                            (1,atom_position_a_transformed.shape[1]))
        atom_position_a_transformed = np.swapaxes(atom_position_a_transformed,0,1)

        dark_field_image = dark_field_image[:,:,0]
        labels_shaped_transformed = labels_shaped_transformed[:,:,0]
        
        self.darkfield_drift = dark_field_image
        self.EELS_data_drift = EELS_data
        self.labels_shaped_drift = labels_shaped_transformed
        self.atom_position_drift = atom_position_a_transformed
        
        # Plot drift corrected images
        plt.figure()
        plt.imshow(dark_field_image)
        plt.scatter(atom_position_a_transformed[:,0],atom_position_a_transformed[:,1], c='r', s=4)
        plt.title('Drift corrected dark field image')

    def drift_on_off(self, drift_corr = True):
        if drift_corr:
            self.darkfield_stacking = self.darkfield_drift
            self.atom_position_stacking = np.array(self.atom_position_drift)
            self.labels_shaped_stacking = self.labels_shaped_drift
            self.EELS_data_stacking = self.EELS_data_drift
        else:
            self.darkfield_stacking = self.s_darkfield.data
            self.atom_position_stacking = np.array(self.atom_positions)
            self.labels_shaped_stacking = self.labels_shaped
            self.EELS_data_stacking = self.s_EELS.data            

    def stacking(self, width, height, shift_x = 0, shift_y = 0):
        dark_field = self.darkfield_stacking
        atom = self.atom_position_stacking
        labels = self.labels_shaped_stacking
        self.EELS = self.EELS_data_stacking
        
        # Shuffle the stacks (better alignment with SmartAlign)
        shuffle = 1 

        # Get shapes
        image_dim = np.shape(dark_field)

        # Get pixel
        x_px = np.round(np.array(atom)[:,0]) + shift_x
        y_px = np.round(np.array(atom)[:,1]) + shift_y


        # Check if every atom can be cropped with the given width and height. If not, the atoms are excluded
        del_px = []
        for i in range(0,len(x_px)):

            x_low = int(x_px[i]-width/2)
            x_high = int(x_px[i]+width/2)
            y_low = int(y_px[i]-height/2)
            y_high = int(y_px[i]+height/2)  

            # Atoms too close to the border are removed
            if x_px[i] + width/2 >= image_dim[1] or x_px[i] - width/2 < 0:
                del_px.append(i)
            elif y_px[i] + height/2 >= image_dim[0] or y_px[i] - height/2 < 0:
                del_px.append(i)
            elif np.isnan(np.sum(dark_field[y_low:y_high+1, x_low:x_high+1])) == True:
                del_px.append(i)

        # Atoms, which can be used for cropping
        x_px_within_image = np.delete(x_px,del_px,0)
        y_px_within_image = np.delete(y_px,del_px,0)

        # Random vector for shuffling
        if shuffle == 1:
            arr_shuffle = np.arange(len(x_px_within_image))
            np.random.shuffle(arr_shuffle) 

            x_px_within_image = x_px_within_image[arr_shuffle]
            y_px_within_image = y_px_within_image[arr_shuffle]


        # Different color for each rectangles
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, len(y_px_within_image))]
        if shuffle == 1:
            colors = [colors[i] for i in range(0, len(y_px_within_image))]


        # Create empty stacks for dark field image and the cropping positions
        self.dark_field_stack = np.zeros([height, width, len(y_px_within_image)])
        self.index_image_stack = []

        # Cropping rectangulars from the dark field image and stack them & plot dark field image incl the cells
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1.imshow(dark_field)
        ax2.matshow(labels,cmap=self.newcmp)

        for i in range(0, len(y_px_within_image)):
            # Defining boundaries of the cells
            x_low = int(x_px_within_image[i]-width/2)
            x_high = int(x_px_within_image[i]+width/2)
            y_low = int(y_px_within_image[i]-height/2)
            y_high = int(y_px_within_image[i]+height/2)

            # Adding cells
            self.dark_field_stack[:,:,i] = dark_field[y_low:y_high, x_low:x_high]
            self.index_image_stack.append([y_low, x_low])

            # Plotting results

            # Create patches
            rect1 = patches.Rectangle((x_low,y_low),width,height,linewidth=1,edgecolor=colors[i],facecolor='none')
            point1 = patches.Circle((x_px_within_image[i],y_px_within_image[i]),1,linewidth=1,edgecolor=colors[i],facecolor=colors[i])
            rect2 = patches.Rectangle((x_low,y_low),width,height,linewidth=1,edgecolor=colors[i],facecolor='none')
            point2 = patches.Circle((x_px_within_image[i],y_px_within_image[i]),1,linewidth=1,edgecolor=colors[i],facecolor=colors[i])

            # Add the patch to the Axes
            ax1.add_patch(rect1)
            ax1.add_patch(point1)

            ax2.add_patch(rect2)
            ax2.add_patch(point2)    

        print(f'Number of cells: {len(y_px_within_image)}')
        
    def normalize_L2(self, img_arr, norming):
        # check if input is a stack or a single input
        if len(img_arr.shape) > 2:
            img_arr_norm = np.zeros(img_arr.shape)
            for i in range(0,img_arr.shape[2]):
                if norming == 'max':
                    img_arr_norm[:,:,i] = img_arr[:,:,i]/np.amax(img_arr[:,:,i])
                elif norming == 'mean':
                    img_arr_norm[:,:,i] = img_arr[:,:,i]/np.mean(img_arr[:,:,i])
                elif norming == 'kernel':
                    img_arr_norm[:,:,i] = ndimage.median_filter(img_arr[:,:,i], size = (3,3))
                elif norming == 'maxmin':
                    img_arr_norm[:,:,i] = (img_arr[:,:,i] - np.amin(img_arr[:,:,i]))/(np.amax(img_arr[:,:,i]) - np.amin(img_arr[:,:,i]))
                elif norming == 'clahe':
                    clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize=(4, 4))
                    img_arr_norm[:,:,i] = clahe.apply((255*(img_arr[:,:,i]/np.amax(img_arr[:,:,i]))).astype(np.uint8))
                elif norming == 'gaussian':
                    img_gauss = ndimage.gaussian_filter(img_arr[:,:,i], sigma=1)
                    img_arr_norm[:,:,i] = img_gauss/np.mean(img_gauss)
        else:
                if norming == 'max':
                    img_arr_norm = img_arr/np.amax(img_arr)
                elif norming == 'mean':
                    img_arr_norm = img_arr/np.mean(img_arr)
                elif norming == 'kernel':
                    img_arr_norm = ndimage.median_filter(img_arr, size = (3,3))
                elif norming == 'maxmin':
                    img_arr_norm = (img_arr - np.amin(img_arr))/(np.amax(img_arr) - np.amin(img_arr))
                elif norming == 'clahe':
                    clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize=(4, 4))
                    img_arr_norm = clahe.apply( (255*(img_arr/np.amax(img_arr))).astype(np.uint8))      
                elif norming == 'gaussian':
                    img_gauss = ndimage.gaussian_filter(img_arr, sigma=1)
                    img_arr_norm = img_gauss/np.mean(img_gauss)                    
        return img_arr_norm

    def L2_init(self, norming = 'max', exponent = 2):
        # Normalize images
        darkfield_aligned_norm = self.normalize_L2(self.darkfield_aligned, norming)
        darkfield_aligned_averaged_norm = self.normalize_L2(np.sum(self.darkfield_aligned, axis = -1), norming)

        # Calculate L2 norm
        self.L2_norm = []
        for i, cell in enumerate(np.moveaxis(darkfield_aligned_norm,2,0)):
            self.L2_norm.append(np.sum(np.power(np.abs(darkfield_aligned_averaged_norm-cell), exponent)))
            
        return darkfield_aligned_norm
    
    def L2_norm_process(self, n_ratio = 0.1, norming = 'max', exponent = 2):
        # Stacks, where the slices will be thrown out
        darkfield_aligned_norm_L2 = self.normalize_L2(self.darkfield_aligned, norming)
        L2_excluded = []
        
        # Number of images, which are removed
        n_abs = np.round(darkfield_aligned_norm_L2.shape[2]*(1-n_ratio)).astype(np.int32)

        for i in range(0,n_abs):
            # Remove slice with the largest L2 norm
            i_remove = np.argsort(np.array(self.L2_norm))[-1]

            # Remove slice from all stacks
            darkfield_aligned_norm_L2 = np.delete(darkfield_aligned_norm_L2,i_remove,axis=2)
            L2_excluded.append(i_remove)
        
            # New averaged image
            darkfield_aligned_averaged_norm_L2 = np.sum(darkfield_aligned_norm_L2, axis=2)
            darkfield_aligned_averaged_norm_L2 = self.normalize_L2(darkfield_aligned_averaged_norm_L2, norming)

            # Calculate new L2 norm
            self.L2_norm = []
            for i, cell in enumerate(np.moveaxis(darkfield_aligned_norm_L2,2,0)):
                self.L2_norm.append(np.sum(np.power(np.abs(darkfield_aligned_averaged_norm_L2-cell), exponent)))

        self.slice_L2_excluded = L2_excluded
        
        print(f'Number of total slices: {self.darkfield_aligned.shape[2]}')
        print(f'Number of slices after L2-norm: {self.darkfield_aligned.shape[2]-n_abs}')

        # Plotting results

        # Compare new averaged image and the old one
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        
        self.darkfield_plot = np.zeros(self.darkfield_aligned.shape[0:2])
        for i in range(0, self.darkfield_aligned.shape[-1]):
             if not i in L2_excluded:
                self.darkfield_plot += self.darkfield_aligned[:,:,i]
        
        ax2.imshow(self.darkfield_plot)
        ax2.set_title('Reduced')

        ax1.imshow(np.sum(self.darkfield_aligned,axis = -1))
        ax1.set_title('All')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])        
     
    
    
    def EELS_region(self):
        
        # Generate a hyperspy-class from the average spectrum image to apply the PCA
        self.s_averaged_orig = hs.signals.Signal2D(self.EELS_sum_aligned)
        # Adjust axes
        self.s_averaged_orig.axes_manager = self.s_EELS.axes_manager
        self.s_averaged_orig.axes_manager['x'].size = self.EELS_sum_aligned.data.shape[1]
        self.s_averaged_orig.axes_manager['y'].size = self.EELS_sum_aligned.data.shape[0]

        # Average whole averaged spectrum image for setting signal and background range
        self.s = hs.signals.Signal1D(np.sum(self.EELS_sum_aligned,axis=(0,1)))
        # Adjust axes
        self.s.axes_manager[0].name = self.s_averaged_orig.axes_manager[2].name
        self.s.axes_manager[0].scale = self.s_averaged_orig.axes_manager[2].scale
        self.s.axes_manager[0].offset = self.s_averaged_orig.axes_manager[2].offset
        self.s.axes_manager[0].size = self.s_averaged_orig.axes_manager[2].size

        # Span initial region of interest
        if self.roi_background == None:
            self.roi_background = hs.roi.SpanROI(left=self.s.axes_manager[0].offset, right=self.s.axes_manager[0].offset + self.s.axes_manager[0].scale*self.s.axes_manager[0].size/4)
            self.roi_signal = hs.roi.SpanROI(left=self.s.axes_manager[0].offset + self.s.axes_manager[0].scale*self.s.axes_manager[0].size/3, right=self.s.axes_manager[0].offset + self.s.axes_manager[0].scale*self.s.axes_manager[0].size*2/4)


        # Select background for removal and signal for integration
        self.s.plot()

        plt.gca().set_title("")
        textstr_blue =  'Background'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.1)
        text = plt.gcf().text(0.33, 1.02, textstr_blue, fontsize=12, horizontalalignment='center', verticalalignment='bottom', color = 'blue', bbox=props, transform=plt.gca().transAxes)

        textstr_green =  'Signal'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.1)
        text = plt.gcf().text(0.66, 1.02, textstr_green, fontsize=12, horizontalalignment='center', verticalalignment='bottom', color = 'green', bbox=props, transform=plt.gca().transAxes)
        
        # Get values for the background and signal
        self.roi_background.interactive(self.s, color='blue')
        self.roi_signal.interactive(self.s, color='green')

    # Define background function
    def powerlaw(self, x, A, r):
        return A * x**(-r)
    def exponential(self, x, A, r):
        return A * np.exp(-r*x)    
    
    def EELS_background(self, background_fun = 'Powerlaw'):

        # Define initial parameters and creating the fitting model, depending on the function
        if background_fun == 'Powerlaw':
            # Create fitting model
            self.gmodel = Model(self.powerlaw)

            # Add required parameters (incl. constraints)
            params = Parameters()
            params.add('A', value=1000, min=0, max=np.inf, vary = True)
            params.add('r', value=1, min=0, max=np.inf, vary = True)

        elif background_fun == 'Exponential':
            # Create fitting model
            self.gmodel = Model(self.exponential)

            # Add required parameters (incl. constraints)
            params = Parameters()
            params.add('A', value=450, min=0, max=np.inf, vary = True)
            params.add('r', value=0.01, min=0, max=np.inf, vary = True)

        # Calculate x-axes vector
        x_axes_background = np.arange(self.roi_background.left, self.roi_background.right, self.s.axes_manager[0].scale)
        x_axes_signal = np.arange(self.roi_signal.left, self.roi_signal.right, self.s.axes_manager[0].scale)
        x_axes = np.arange(self.s.axes_manager[0].offset, self.s.axes_manager[0].offset + self.s.axes_manager[0].scale*self.s.axes_manager[0].size, self.s.axes_manager[0].scale)

        # Crop spectrum to background roi
        s_fit = self.s.isig[self.roi_background].data/(self.EELS_sum_aligned.shape[0]*self.EELS_sum_aligned.shape[1])
        # Fit averaged spectrum to get better estimation for initial parameters for each pixel
        self.result_fit = self.gmodel.fit(s_fit, params, x=x_axes_background)

        # Selected energy ranges
        print(f'Background: {(self.roi_background.right - self.roi_background.left):.2f} eV from {self.roi_background.left:.2f} eV to {self.roi_background.right:.2f} eV')
        print(f'Signal: {(self.roi_signal.right - self.roi_signal.left):.2f} eV from {self.roi_signal.left:.2f} eV to {self.roi_signal.right:.2f} eV')

        # Plot signal and fitted background
        figure, ax = plt.subplots(1, 1)
        ax.plot(x_axes,self.s.data/(self.EELS_sum_aligned.shape[0]*self.EELS_sum_aligned.shape[1]), label = 'EELS')
        ax.plot(x_axes,self.gmodel.eval(self.result_fit.params, x=x_axes), label = 'Background')
        ax.plot(x_axes,self.s.data/(self.EELS_sum_aligned.shape[0]*self.EELS_sum_aligned.shape[1])-self.gmodel.eval(self.result_fit.params, x=x_axes), label = 'Residual')
        ax.fill_between(x_axes_signal,self.s.isig[self.roi_signal].data/(self.EELS_sum_aligned.shape[0]*self.EELS_sum_aligned.shape[1])-self.gmodel.eval(self.result_fit.params, x=x_axes_signal), alpha = 0.3, facecolor = 'green')
        ax.plot([np.amin(x_axes), np.amax(x_axes)],[0, 0],'k--')
        ax.plot(x_axes_background,self.result_fit.init_fit,'r--', label = 'Initial guess')
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(bbox_to_anchor=(1,1), loc="upper left")
        ax.set_xlabel('Energy loss / eV')
        ax.set_ylabel('Intensity / a.u.')



        
class atom_selector:
    def __init__(self, atom_positions, s_darkfield, label = None, newcmp = None, map_label = False):
        self.atom_positions = atom_positions
        self.s_darkfield = s_darkfield
        self.label = label
        self.newcmp = newcmp
        self.map_label = map_label

        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.scatter_atom = None
        self.scatter_atom_2 = None
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.scatter_atom = None
        self.scatter_atom_2 = None

        if not self.map_label:
            self.fig, self.ax1 = plt.subplots()
        else:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
            # Plot drift corrected clustered image
            self.ax2.matshow(self.label, cmap=self.newcmp)
            self.scatter_atom_2 = self.ax2.scatter(self.atom_positions[:,0],self.atom_positions[:,1], c='k', edgecolors = 'w', s=18)
            
        self.ax1.imshow(self.s_darkfield.data)
        self.scatter_atom = self.ax1.scatter(self.atom_positions[:,0],self.atom_positions[:,1], c='r', s=4)
        
        textstr =  'Click and drag to draw a rectangle.\nPress "d" to remove atoms.\nPress "t" to toggle the selector on and off.'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        text = self.ax1.text(0.5, 1.02, textstr, fontsize=8, horizontalalignment='center', verticalalignment='bottom', bbox=props, transform=self.ax1.transAxes)
        

        # drawtype is 'box' or 'line' or 'none'
        self.RS = RectangleSelector(self.ax1, self.line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1, 3],  # disable middle button
                                               minspanx=2, minspany=2,
                                               spancoords='pixels',
                                               interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', self.toggle_selector)
        plt.show()                

    def line_select_callback(self, eclick, erelease):
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata
        return

    def toggle_selector(self, event):
        if event.key == 't':
            if self.RS.active:
                self.RS.set_active(False)
            else:
                self.RS.set_active(True)
        if event.key == 'd':
            index = []
            for i, position in enumerate(self.atom_positions):
                if (position[0] >= self.x1 and position[0] <= self.x2 and position[1] >= self.y1 and position[1] <= self.y2):
                    index.append(i)
            self.atom_positions = np.delete(self.atom_positions,index, axis=0)
            self.scatter_atom.set_offsets(np.c_[self.atom_positions[:,0],self.atom_positions[:,1]])
            if self.map_label:
                self.scatter_atom_2.set_offsets(np.c_[self.atom_positions[:,0],self.atom_positions[:,1]])
            self.fig.canvas.draw_idle()
        return


class L2_Selector():
    def __init__(self, darkfield_aligned_norm, L2_norm):
        self.darkfield_aligned_norm = darkfield_aligned_norm
        self.L2_norm = L2_norm
        self.x_points = np.linspace(0,1,self.darkfield_aligned_norm.shape[2])
        
        # Sorting
        L2_sorted_idx = np.argsort(self.L2_norm)
        self.darkfield_aligned_norm = self.darkfield_aligned_norm[:,:,L2_sorted_idx]
        self.L2_sorted = np.asarray(self.L2_norm)[L2_sorted_idx]
        
        # Set initial index
        self.idx = 0
        # Differentiate between click and drag (for zooming)
        self.MAX_CLICK_LENGTH = 0.2
        
        # Plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        
        # Plot L2-norm
        self.lineplot = self.ax1.plot(self.x_points,self.L2_sorted, linestyle='--', marker='o', color='b', zorder=0)
        self.scatter_2, = self.ax1.plot(self.x_points[self.idx],self.L2_sorted[self.idx] , marker='o', color='r')
        
        textstr =  'Click to show specific cell.\nPress "w" or "e" for plus/muinus.'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        text = self.ax1.text(0.5, 1.02, textstr, fontsize=8, horizontalalignment='center', verticalalignment='bottom', bbox=props, transform=self.ax1.transAxes)
        
        
        # Set titles
        self.ax2.set_title('Cell:' + str(self.idx + 1))

        # Plot cells (initialize with first cell)
        self.l2_plot = self.ax2.imshow(self.darkfield_aligned_norm[:,:,self.idx])
        #self.l2_plot.set_clim(vmin=0, vmax=1)

        # Connect
        self.cid = self.ax1.figure.canvas.mpl_connect("button_press_event",self.select_change_press)
        self.cid = self.ax1.figure.canvas.mpl_connect("button_release_event",self.select_change_release)
        self.cid = self.ax1.figure.canvas.mpl_connect("key_press_event",self.select_change_key)
        
    def select_change_press(self, event):
        self.time_onclick = time.time()
        
        return

    def select_change_release(self, event):  
        if (time.time() - self.time_onclick) < self.MAX_CLICK_LENGTH:
            # Get position of mouse
            x = event.xdata

            # Find nearest point
            array = np.asarray(self.x_points)
            self.idx = (np.abs(array - x)).argmin()

            # Plot
            self.plot()
        else:
            pass
        
        return
    
    def select_change_key(self, event):   
        # Detect key
        if event.key == "w":
            self.idx -= 1
        elif event.key == "e":
            self.idx += 1
        
        # Reset idx,if out of boundary
        if self.idx < 0:
            self.idx = 0
        elif self.idx > self.darkfield_aligned_norm.shape[2]-1:
            self.idx = self.darkfield_aligned_norm.shape[2]-1

        # Plot
        self.plot()
        
        return
    
    def plot(self):
        # Set new cell
        self.l2_plot.set_data(self.darkfield_aligned_norm[:,:,self.idx])
        
        # Set new point
        self.scatter_2.set_data(self.x_points[self.idx],self.L2_sorted[self.idx])
        
        # Set new title
        self.ax2.set_title('Cell:' + str(self.idx + 1))

        # Update figure
        self.fig.canvas.draw_idle()
        
        return    
    

class Aligner:
    def __init__(self, images):
        self.images = images

    def rigid_align(self, i_rigid=1, max_shift = None):
        # i_rigid -> Iteration of rigid transformation
        self.images_rigid = np.copy(self.images)
        
        # Save rigid transformation
        self.transf_field_rigid = np.zeros((2,self.images_rigid.shape[-1]))

        # Rigid registration

        # Several iterations
        for i in range(i_rigid):
            #images_avg = self.reference(self.images)
            images_avg = ndimage.median_filter(self.reference(self.images), size = (3,3))
            # Take first image as reference
            #images_avg = ndimage.median_filter(self.images[:, :, 0], size = (3,3))
            print(str(i + 1) + ' of ' + str(i_rigid) + ' rigid registration')
            for j in range(0, self.images.shape[-1]):
                #img_slice = self.images[:, :, j]
                img_slice =self.images[:, :, j]
                #img_slice = (img_slice - np.amin(img_slice))/(np.amax(img_slice)-np.amin(img_slice))
                # Cross correlation
                shift = self.phase_cross_correlation(images_avg, ndimage.median_filter(img_slice, size = (3,3)),
                                                                  upsample_factor=10, max_shift =  max_shift)
                    
                self.transf_field_rigid[:,j] += shift

                self.images_rigid[:, :, j] = self.shift(img_slice, shift)
###########################
    # Calculate cross correlation: functions from skimage modified to set maximum shift!
    def _upsampled_dft(self, data, upsampled_region_size,
                       upsample_factor=1, axis_offsets=None):
        """
        Upsampled DFT by matrix multiplication.
        This code is intended to provide the same result as if the following
        operations were performed:
            - Embed the array "data" in an array that is ``upsample_factor`` times
              larger in each dimension.  ifftshift to bring the center of the
              image to (1,1).
            - Take the FFT of the larger array.
            - Extract an ``[upsampled_region_size]`` region of the result, starting
              with the ``[axis_offsets+1]`` element.
        It achieves this result by computing the DFT in the output array without
        the need to zeropad. Much faster and memory efficient than the zero-padded
        FFT approach if ``upsampled_region_size`` is much smaller than
        ``data.size * upsample_factor``.
        Parameters
        ----------
        data : array
            The input data array (DFT of original data) to upsample.
        upsampled_region_size : integer or tuple of integers, optional
            The size of the region to be sampled.  If one integer is provided, it
            is duplicated up to the dimensionality of ``data``.
        upsample_factor : integer, optional
            The upsampling factor.  Defaults to 1.
        axis_offsets : tuple of integers, optional
            The offsets of the region to be sampled.  Defaults to None (uses
            image center)
        Returns
        -------
        output : ndarray
                The upsampled DFT of the specified region.
        """
        # if people pass in an integer, expand it to a list of equal-sized sections
        if not hasattr(upsampled_region_size, "__iter__"):
            upsampled_region_size = [upsampled_region_size, ] * data.ndim
        else:
            if len(upsampled_region_size) != data.ndim:
                raise ValueError("shape of upsampled region sizes must be equal "
                                 "to input data's number of dimensions.")
    
        if axis_offsets is None:
            axis_offsets = [0, ] * data.ndim
        else:
            if len(axis_offsets) != data.ndim:
                raise ValueError("number of axis offsets must be equal to input "
                                 "data's number of dimensions.")
    
        im2pi = 1j * 2 * np.pi
    
        dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))
    
        for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
            kernel = ((np.arange(ups_size) - ax_offset)[:, None]
                      * fftfreq(n_items, upsample_factor))
            kernel = np.exp(-im2pi * kernel)
            # use kernel with same precision as the data
            kernel = kernel.astype(data.dtype, copy=False)
    
            # Equivalent to:
            #   data[i, j, k] = kernel[i, :] @ data[j, k].T
            data = np.tensordot(kernel, data, axes=(1, -1))
        return data
    
    
    def phase_cross_correlation(self, reference_image, moving_image, *,
                                upsample_factor=1, space="real",
                                return_error=True,
                                normalization="phase",
                                max_shift = None):
        """Efficient subpixel image translation registration by cross-correlation.
        This code gives the same precision as the FFT upsampled cross-correlation
        in a fraction of the computation time and with reduced memory requirements.
        It obtains an initial estimate of the cross-correlation peak by an FFT and
        then refines the shift estimation by upsampling the DFT only in a small
        neighborhood of that estimate by means of a matrix-multiply DFT [1]_.
        Parameters
        ----------
        reference_image : array
            Reference image.
        moving_image : array
            Image to register. Must be same dimensionality as
            ``reference_image``.
        upsample_factor : int, optional
            Upsampling factor. Images will be registered to within
            ``1 / upsample_factor`` of a pixel. For example
            ``upsample_factor == 20`` means the images will be registered
            within 1/20th of a pixel. Default is 1 (no upsampling).
            Not used if any of ``reference_mask`` or ``moving_mask`` is not None.
        space : string, one of "real" or "fourier", optional
            Defines how the algorithm interprets input data. "real" means
            data will be FFT'd to compute the correlation, while "fourier"
            data will bypass FFT of input data. Case insensitive. Not
            used if any of ``reference_mask`` or ``moving_mask`` is not
            None.
        return_error : bool, optional
            Returns error and phase difference if on, otherwise only
            shifts are returned. Has noeffect if any of ``reference_mask`` or
            ``moving_mask`` is not None. In this case only shifts is returned.
        reference_mask : ndarray
            Boolean mask for ``reference_image``. The mask should evaluate
            to ``True`` (or 1) on valid pixels. ``reference_mask`` should
            have the same shape as ``reference_image``.
        moving_mask : ndarray or None, optional
            Boolean mask for ``moving_image``. The mask should evaluate to ``True``
            (or 1) on valid pixels. ``moving_mask`` should have the same shape
            as ``moving_image``. If ``None``, ``reference_mask`` will be used.
        overlap_ratio : float, optional
            Minimum allowed overlap ratio between images. The correlation for
            translations corresponding with an overlap ratio lower than this
            threshold will be ignored. A lower `overlap_ratio` leads to smaller
            maximum translation, while a higher `overlap_ratio` leads to greater
            robustness against spurious matches due to small overlap between
            masked images. Used only if one of ``reference_mask`` or
            ``moving_mask`` is None.
        normalization : {"phase", None}
            The type of normalization to apply to the cross-correlation. This
            parameter is unused when masks (`reference_mask` and `moving_mask`) are
            supplied.
        Returns
        -------
        shifts : ndarray
            Shift vector (in pixels) required to register ``moving_image``
            with ``reference_image``. Axis ordering is consistent with
            numpy (e.g. Z, Y, X)
        error : float
            Translation invariant normalized RMS error between
            ``reference_image`` and ``moving_image``.
        phasediff : float
            Global phase difference between the two images (should be
            zero if images are non-negative).
        Notes
        -----
        The use of cross-correlation to estimate image translation has a long
        history dating back to at least [2]_. The "phase correlation"
        method (selected by ``normalization="phase"``) was first proposed in [3]_.
        Publications [1]_ and [2]_ use an unnormalized cross-correlation
        (``normalization=None``). Which form of normalization is better is
        application-dependent. For example, the phase correlation method works
        well in registering images under different illumination, but is not very
        robust to noise. In a high noise scenario, the unnormalized method may be
        preferable.
        When masks are provided, a masked normalized cross-correlation algorithm is
        used [5]_, [6]_.
        References
        ----------
        .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
               "Efficient subpixel image registration algorithms,"
               Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
        .. [2] P. Anuta, Spatial registration of multispectral and multitemporal
               digital imagery using fast Fourier transform techniques, IEEE Trans.
               Geosci. Electron., vol. 8, no. 4, pp. 353–368, Oct. 1970.
               :DOI:`10.1109/TGE.1970.271435`.
        .. [3] C. D. Kuglin D. C. Hines. The phase correlation image alignment
               method, Proceeding of IEEE International Conference on Cybernetics
               and Society, pp. 163-165, New York, NY, USA, 1975, pp. 163–165.
        .. [4] James R. Fienup, "Invariant error metrics for image reconstruction"
               Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
        .. [5] Dirk Padfield. Masked Object Registration in the Fourier Domain.
               IEEE Transactions on Image Processing, vol. 21(5),
               pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
        .. [6] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
               Pattern Recognition, pp. 2918-2925 (2010).
               :DOI:`10.1109/CVPR.2010.5540032`
        """
    
        # images must be the same shape
        if reference_image.shape != moving_image.shape:
            raise ValueError("images must be same shape")
    
        if max_shift == None:
            max_shift = reference_image.shape
    
        # assume complex data is already in Fourier space
        if space.lower() == 'fourier':
            src_freq = reference_image
            target_freq = moving_image
        # real data needs to be fft'd.
        elif space.lower() == 'real':
            src_freq = fftn(reference_image)
            target_freq = fftn(moving_image)
        else:
            raise ValueError('space argument must be "real" of "fourier"')
    
        # Whole-pixel shift - Compute cross-correlation by an IFFT
        shape = src_freq.shape
        image_product = src_freq * target_freq.conj()
        if normalization == "phase":
            eps = np.finfo(image_product.real.dtype).eps
            image_product /= np.maximum(np.abs(image_product), 100 * eps)
        elif normalization is not None:
            raise ValueError("normalization must be either phase or None")
        cross_correlation = ifftn(image_product)
    
        # Locate maximum 
        # Check if shift is in maximum shift
        k = 0
        x_vec = np.abs(cross_correlation)
        L = np.argsort(-x_vec.flatten(order='C'))
    
        while True:
            # k-largest value
            maxima = np.unravel_index(L[k], cross_correlation.shape, order='C')
    
            # Check if in maximum
            if (maxima[0] <= max_shift[0]) and  (maxima[1] <= max_shift[1]):
                break
            
            k += 1
        
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])
    
        float_dtype = image_product.real.dtype
    
        shifts = np.stack(maxima).astype(float_dtype, copy=False)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
    
        if upsample_factor == 1:
            if return_error:
                src_amp = np.sum(np.real(src_freq * src_freq.conj()))
                src_amp /= src_freq.size
                target_amp = np.sum(np.real(target_freq * target_freq.conj()))
                target_amp /= target_freq.size
                CCmax = cross_correlation[maxima]
        # If upsampling > 1, then refine estimate with matrix multiply DFT
        else:
            # Initial shift estimate in upsampled grid
            upsample_factor = np.array(upsample_factor, dtype=float_dtype)
            shifts = np.round(shifts * upsample_factor) / upsample_factor
            upsampled_region_size = np.ceil(upsample_factor * 1.5)
            # Center of output array at dftshift + 1
            dftshift = np.fix(upsampled_region_size / 2.0)
            # Matrix multiply DFT around the current shift estimate
            sample_region_offset = dftshift - shifts*upsample_factor
            cross_correlation = self._upsampled_dft(image_product.conj(),
                                               upsampled_region_size,
                                               upsample_factor,
                                               sample_region_offset).conj()
            # Locate maximum and map back to original pixel grid

            # Locate maximum and map back to original pixel grid
            maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                                      cross_correlation.shape)
            CCmax = cross_correlation[maxima]
    
            maxima = np.stack(maxima).astype(float_dtype, copy=False)
            maxima -= dftshift
    
            shifts += maxima / upsample_factor
    
            if return_error:
                src_amp = np.sum(np.real(src_freq * src_freq.conj()))
                target_amp = np.sum(np.real(target_freq * target_freq.conj()))
    
        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        for dim in range(src_freq.ndim):
            if shape[dim] == 1:
                shifts[dim] = 0
    

        return shifts


###############################
    def shift(self, image, vector):
        shifted = ndimage.shift(image, vector, mode = 'nearest')
        
        return shifted.astype(image.dtype)

    def reference(self, images_ref, option='avg'):
        if option == 'avg':
            reference_img = np.mean(images_ref, axis=2)
            
            #reference_img = (reference_img - np.amin(reference_img))/(np.amax(reference_img)- np.amin(reference_img))

        return reference_img

    def gradient_img(self, images_grad):
        #img_grad = np.zeros(images.shape)
        #T = (img_grad, img_grad)
        #for i in range(0, images.shape[-1]):
        # Get x-gradient in "sx"
        sx = ndimage.sobel(images_grad, axis=0, mode='nearest')
        # Get y-gradient in "sy"
        sy = ndimage.sobel(images_grad, axis=1, mode='nearest')
        # Get square root of sum of squares
        #img_grad[:, :, i] = np.hypot(sx, sy)
        norm_x = np.linalg.norm(sx)
        norm_y = np.linalg.norm(sy)
        if norm_x == 0:
            norm_x = 1
        if norm_y == 0:
            norm_y = 1
            
        T_x = sx / norm_x
        T_y = sy / norm_y

        return (T_x, T_y)
    
    # Function is written by D.Kroon University of Twente (February 2009)
    def movepixels_2d_double(self, Iin,Tx,Ty,mode = 1):
    
        height = Iin.shape[0]
        width = Iin.shape[1]
        
        if len(Iin.shape) == 2:
            depth = 1
            Iin = Iin[:,:,np.newaxis]
            out_red = True
        else:
            depth = Iin.shape[2]
            out_red = False
            
        # Make all x,y indices
        x, y = np.meshgrid(np.arange(0, height),np.arange(0, width), indexing='ij')
    
        # Calculate the Transformed coordinates
        Tlocalx = x + Tx
        Tlocaly = y + Ty
    
        # All the neighborh pixels involved in linear interpolation.
        xBas0 = np.floor(Tlocalx)
        yBas0 = np.floor(Tlocaly)
        xBas1 = xBas0 + 1     
        yBas1 = yBas0 + 1
    
        # Linear interpolation constants (percentages)
        xCom = Tlocalx - xBas0
        yCom = Tlocaly - yBas0
        perc0 = (1 - xCom) * (1 - yCom)
        perc1 = (1 - xCom) * yCom
        perc2 = xCom * (1 - yCom)
        perc3 = xCom * yCom
    
        # limit indexes to boundaries
        check_xBas0 = (xBas0 < 0) | (xBas0 > (Iin.shape[0]-1))
        check_yBas0 = (yBas0 < 0) | (yBas0 > (Iin.shape[1]-1))
        xBas0[check_xBas0] = 0 
        yBas0[check_yBas0] = 0 
        check_xBas1 = (xBas1 < 0) | (xBas1 > (Iin.shape[0]-1))
        check_yBas1 = (yBas1 < 0) | (yBas1 > (Iin.shape[1]-1))
        xBas1[check_xBas1] = 0 
        yBas1[check_yBas1] = 0 
    
        Iout = np.zeros(Iin.shape)
        for i in range(0,depth):    
            Iin_one = Iin[:,:,i]
            # Get the intensities
            intensity_xyz0 = np.reshape(Iin_one.flatten(order="F")[(xBas0 + yBas0 * Iin.shape[0]).astype(np.int32)],(height, width))
            intensity_xyz1 = np.reshape(Iin_one.flatten(order="F")[(xBas0 + yBas1 * Iin.shape[0]).astype(np.int32)],(height, width))
            intensity_xyz2 = np.reshape(Iin_one.flatten(order="F")[(xBas1 + yBas0 * Iin.shape[0]).astype(np.int32)],(height, width))
            intensity_xyz3 = np.reshape(Iin_one.flatten(order="F")[(xBas1 + yBas1 * Iin.shape[0]).astype(np.int32)],(height, width))
            
            # Make pixels before outside Ibuffer mode
            if mode==1 and mode==3:
                intensity_xyz0[check_xBas0 | check_yBas0] = 0
                intensity_xyz1[check_xBas0 | check_yBas1] = 0
                intensity_xyz2[check_xBas1 | check_yBas0] = 0
                intensity_xyz3[check_xBas1 | check_yBas1] = 0
            
            # Calculate new intensities and reshape to original data
            Iout_one = intensity_xyz0 * perc0 + intensity_xyz1 * perc1 + intensity_xyz2 * perc2 + intensity_xyz3 * perc3 
            Iout[:,:,i] = np.reshape(Iout_one, (Iin.shape[0],Iin.shape[1]), order="F")

        if out_red:
            Iout = Iout[:,:,0]

        return Iout

    def norm(self, imgs):
        img_norm = imgs
        for i in range(0,imgs.shape[-1]):
            img_norm[..., i] = (imgs[..., i] - np.amin(imgs[..., i]))/(np.amax(imgs[..., i]) - np.amin(imgs[..., i]))
            
        return img_norm

    def non_rigid_align(self, i_non_rigid_it = 2, i_non_rigid_max = 1000, row = None):
        from scipy.signal import convolve2d
        
        # Normalize images
        img_norm = self.norm(np.copy(self.images_rigid))
        
        # Get reference image
        img_ref = self.reference(img_norm)
        
        # Save whole transformation
        self.transf_field_x_non = np.zeros(img_norm.shape)
        self.transf_field_y_non = np.zeros(img_norm.shape)
        
        self.image_align = np.copy(self.images_rigid)
        
        # Set smoothing kernel --> other kernel? 2 pixel smoothing kernel according to paper?
        #k_smooth = np.array([[1,1,1],
        #                     [1,2,1],
        #                     [1,1,1]])
        
        # Run non-rigid registration
        for k in range(0,i_non_rigid_it):
        
            # Calculate gradient
            img_ref_grad = self.gradient_img(img_ref)  
            print(str(k + 1) + ' of ' + str(i_non_rigid_it) + ' non-rigid registration')
            # Run each image in the stack
            for i in range(0, img_norm.shape[-1]):
                #image_align = self.images_rigid[:, :, i]
                # X for linear fit
                x_fast = np.arange(0, img_norm.shape[1])
                x_slow = np.arange(0, img_norm.shape[0])

                # Run until convergence or maximum iteration
                for j in range(0, i_non_rigid_max):
                    # Calculate gradient
                    img_align_grad = self.gradient_img(img_norm[:,:,i])
                    
                    # Difference gradient
                    grad_x = - (img_norm[:,:,i] - img_ref) * (img_align_grad[0] + img_ref_grad[0])
                    grad_y = - (img_norm[:,:,i] - img_ref) * (img_align_grad[1] + img_ref_grad[1])
                   
                    # Constrain Transformation!!!
                    
                    # Displacement map smoothing
                    #Correct smoothing!!!!
                    #grad_x_smooth = convolve2d(grad_x, k_smooth, mode='same', boundary = 'fill', fillvalue = 0)
                    #grad_y_smooth = convolve2d(grad_y, k_smooth, mode='same', boundary = 'fill', fillvalue = 0)
                    grad_x_smooth = ndimage.gaussian_filter(grad_x, sigma = 1, mode = 'nearest')
                    grad_y_smooth = ndimage.gaussian_filter(grad_y, sigma = 1, mode = 'nearest')
                    #self.x_grad = grad_x_smooth
                    #self.y_grad = grad_y_smooth                    
                    # Linear fit for each scan direction (fast and slow)
                    # Run linear fit for each line
                    grad_x_smooth_fit = np.copy(grad_x_smooth)
                    grad_y_smooth_fit = np.copy(grad_y_smooth)
                    
                    #if row == None:
                        # Raw gradients
                    if row == 'locked':
                        for k_line in range(0, img_norm.shape[0]):
                            grad_x_smooth_fit[k_line,:] = np.mean(grad_x_smooth[k_line, :])
                        
                        for k_line in range(0, img_norm.shape[1]):
                            grad_y_smooth_fit[:,k_line] = np.mean(grad_y_smooth[:, k_line])                      
                    elif row == 'fitted':
                        # Linear fit                      
                        for k_line in range(0, img_norm.shape[0]):
                            grad_x_smooth_fit[k_line,:] = np.polynomial.polynomial.polyval(x_fast, np.polyfit(x_fast, grad_x_smooth[k_line, :], deg = 1))
                        
                        for k_line in range(0, img_norm.shape[1]):
                            grad_y_smooth_fit[:,k_line] = np.polynomial.polynomial.polyval(x_slow, np.polyfit(x_slow, grad_y_smooth[:, k_line], deg = 1))
    
                    #grad_y_smooth_fit = np.zeros(grad_x_smooth_fit.shape)
    
                    if j == 0:
                        Tx = grad_x_smooth_fit
                        Ty = grad_y_smooth_fit
                    else:
                        Tx = 7*grad_x_smooth_fit
                        Ty = 7*grad_y_smooth_fit
                    
                    self.transf_field_x_non[:,:,i] += Tx
                    self.transf_field_y_non[:,:,i] += Ty

                    # Check tolerance --> if true break loop (j)
                    if np.abs(np.mean(grad_x_smooth_fit)) < 0.001 and np.abs(np.mean(grad_y_smooth_fit)) < 0.001:
                        print('Frame ' + str(i+1) + ' converged after ' + str(j) + ' iterations.')
                        break
                        
                    # Apply transformation
                    self.image_align[:,:,i] = self.movepixels_2d_double(self.image_align[:,:,i],Tx,Ty)
                    img_norm[:,:,i] = self.movepixels_2d_double(img_norm[:,:,i],Tx,Ty)

            # Get new reference image
            img_ref = self.reference(img_norm)
            
    def align_second(self, second_signal, index_cells, slice_L2_excluded):
        
        width = self.image_align.shape[0]
        height = self.image_align.shape[1]
        
        EELS_sum = np.zeros((width, height, second_signal.shape[-1]))
        
        for i, cell_index in enumerate(tqdm_notebook(index_cells)):
            # If slice in the excluded list skip iteration
            if i in slice_L2_excluded:
                continue
            
            # Crop EELS data
            EELS_slice = second_signal[cell_index[0]:cell_index[0]+width,cell_index[1]:cell_index[1]+height,:]
        
            # Rigid align
            EELS_slice_rigid = self.shift(EELS_slice, np.append(self.transf_field_rigid[:,i],0)) 
            
            # Non-rigid align
            Tx = self.transf_field_x_non[:,:,i]
            Ty = self.transf_field_y_non[:,:,i]
            
            EELS_slice_non_rigid = self.movepixels_2d_double(EELS_slice_rigid,Tx,Ty)
        
            # Applying transformation for every energy channel
            EELS_sum += EELS_slice_non_rigid
            
        return EELS_sum
            
    def plot_aligned(self):
        self.fig, ((ax1, ax2), (self.ax3, self.ax4)) = plt.subplots(2,2, sharex=True, sharey=True)
        
        ax1.imshow(np.sum(self.images, axis = 2))
        ax1.set_title('Raw images')

        ax2.imshow(np.sum(self.image_align, axis = 2))
        ax2.set_title('Aligned images')
        
        self.plot_anim3 = self.ax3.imshow(self.images[:,:,0])
        self.plot_anim3.set_clim(vmin=np.amin(self.images), vmax=np.amax(self.images))
        self.ax3.set_title(f'Raw: Cell {0}')

        self.plot_anim4 = self.ax4.imshow(self.image_align[:,:,0])
        self.plot_anim4.set_clim(vmin=np.amin(self.image_align), vmax=np.amax(self.image_align))
        self.ax4.set_title(f'Aligned: Cell {0}')
        
        self.ani = animation.FuncAnimation(self.fig, self.animate, frames=self.images.shape[2], fargs=(self.images, self.image_align), interval=500)
        plt.show()
        
    def animate(self, i, raw, aligned):
        im_raw = raw[:,:,i]
        im_aligned = aligned[:,:,i]

        self.plot_anim3.set_data(im_raw)
        self.plot_anim4.set_data(im_aligned)

        self.ax3.set_title(f'Raw: Cell {i}')
        self.ax4.set_title(f'Aligned: Cell {i}')
        
        return self.plot_anim3, self.plot_anim4
    
    
# Plotting and saving results    
class Selector_pca():
    def __init__(self,analyse, d_neighbour, background_removal_pca, n_back):
        # Initialize variables
        self.s_averaged = analyse.s_averaged_orig
        self.roi_1 = analyse.roi_background
        self.roi_2 = analyse.roi_signal
        self.darkfield_aligned_averaged_norm = (analyse.darkfield_plot - np.amin(analyse.darkfield_plot))/(np.amax(analyse.darkfield_plot) - np.amin(analyse.darkfield_plot))
        self.path_EELS = analyse.path_EELS
        self.para_init = analyse.result_fit.params
        self.gmodel = analyse.gmodel
        self.poisson_noise = analyse.poisson_noise
        
        self.background = background_removal_pca
        self.d = d_neighbour
        self.n_back = n_back
        
        self.s_eels = None

   
        # Location of saving results
        self.path = os.path.join(self.path_EELS, 'Post_Processing')

         # If folder is already present, ask to overwrite files
        if not os.path.exists(self.path):          
            os.mkdir(self.path)
            # Save dark field image
            Image.fromarray(self.darkfield_aligned_averaged_norm ).save(os.path.join(self.path, 'Darkfield_Averaged.tiff'))

        # Save dark field image
        Image.fromarray(self.darkfield_aligned_averaged_norm ).save(os.path.join(self.path, 'Darkfield_Averaged.tiff'))

        # Initial index
        self.idx = 1
        # Differentiate between click and drag for zooming
        self.MAX_CLICK_LENGTH = 0.2 
        
        # Poisson distribution does not allow negative values
        if self.poisson_noise:
            self.s_averaged.data[self.s_averaged.data < 0] = 0
        
        ## Initialize plots
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2)
        
        # Share axes (for zooming)
        self.ax2.get_shared_x_axes().join(self.ax2, self.ax3)
        self.ax2.get_shared_x_axes().join(self.ax2, self.ax4)       
        self.ax2.get_shared_y_axes().join(self.ax2, self.ax3)
        self.ax2.get_shared_y_axes().join(self.ax2, self.ax4)    
        
        # Set ticks off
        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])
        self.ax3.set_xticklabels([])
        self.ax3.set_yticklabels([])
        self.ax4.set_xticklabels([])
        self.ax4.set_yticklabels([])      
        
        ## Plot ax2
        
        # Plot darkfield image
        self.ax2.imshow(self.darkfield_aligned_averaged_norm)
        
        
        ## Plot ax3
        
        # Plot EELS map from averaged EELS signal
        self.s_avg_residual = self.background_subtraction(self.gmodel, self.s_averaged, self.roi_1, self.para_init, self.d)
        self.s_avg_residual_integrated = self.s_avg_residual.isig[self.roi_2].integrate1D(-1)
        self.s_avg_residual_integrated_norm =self.normalize(self.s_avg_residual_integrated)
        
        # Save
        Image.fromarray(self.s_avg_residual_integrated_norm).save(self.path + '\\Signal_Averaged_Integrate_(' + str(np.around(self.roi_2.left,decimals = 2)) + '-' + str(np.around(self.roi_2.right,decimals = 2)) + ')_Background_(' + str(np.around(self.roi_1.left,decimals = 2)) + '-' + str(np.around(self.roi_1.right,decimals = 2)) + ')_Background_' + str(self.background) + '.tiff')    

        # Plot averaged EELS map
        self.ax3.imshow(self.s_avg_residual_integrated_norm) 
       
        
        ## Plot ax4
        
        # If background true --> denoise spectra --> remove background from raw data --> denoise spectra --> integrate        
        if self.background:
            self.s_averaged.decomposition(normalize_poissonian_noise = self.poisson_noise)
            # Denoise spectra
            self.s_pca_background = self.s_averaged.get_decomposition_model(int(self.n_back))
            # Remove PCA-denoised background from raw data
            self.s_pca_residual = self.background_subtraction(self.gmodel, self.s_pca_background, self.roi_1, self.para_init, self.d, self.s_averaged)
            
            if self.poisson_noise:
                self.s_pca_residual.data[self.s_pca_residual.data < 0] = 0            
            
            # Decompose residual
            self.s_pca_residual.decomposition(normalize_poissonian_noise = self.poisson_noise)   
            
        # If background false --> denoise spectra --> remove background from denoised spectra --> integrate        
        else:
            # Decompose raw data
            self.s_pca_residual = self.s_averaged
            self.s_pca_residual.decomposition(normalize_poissonian_noise = self.poisson_noise)   

        self.s_pca_s_arr = self.pca_denoise(self.s_pca_residual, self.background)
        
        self.p_pca_s_arr = self.ax4.imshow(self.s_pca_s_arr)
        
        
        ## Plot ax1
        
        # Get PCA components variance
        self.variance_pca = self.s_pca_residual.get_explained_variance_ratio().data[:100]
        self.x_scree = np.arange(1,len(self.variance_pca)+1)
        
        # Plot scree plot
        self.lineplot = self.ax1.plot(self.x_scree[1:],self.variance_pca[1:], marker='o', color='b', zorder=0)
        self.scatter_2, = self.ax1.plot(self.x_scree[self.idx], self.variance_pca[self.idx] , marker='o', color='r')
        self.ax1.set_yscale('log')
     
    
        # Set titles
        textstr =  'Click to change components. \n Press "w" or "e" for plus/minus.'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        text = self.ax1.text(0.5, 1.02, textstr, fontsize=8, horizontalalignment='center', verticalalignment='bottom', bbox=props, transform=self.ax1.transAxes)
        
        self.ax2.set_title('Darkfield image')
        self.ax3.set_title('EELS map avg')
        self.ax4.set_title('Number of components:' + str(self.idx))
        textstr =  'Press "a" to save image'
        text = self.ax4.text(0.5, -0.02, textstr, fontsize=8, horizontalalignment='center', verticalalignment='top', bbox=props, transform=self.ax4.transAxes)  
    
        
        # mpl_connect (press & release to distinguish between clicking and dragging)
        self.cid1 = self.ax1.figure.canvas.mpl_connect("button_press_event",self.select_change_press)
        self.cid2 = self.ax1.figure.canvas.mpl_connect("button_release_event",self.select_change_release)
        self.cid3 = self.ax1.figure.canvas.mpl_connect("key_press_event",self.select_change_key)
        
      
        
    def select_change_press(self, event):
        # Start timer
        self.time_onclick = time.time()
        
        return

    def select_change_release(self, event):  
        # If mouse is released before MAX_CLICK_LENGTH duration --> click, otherwise dragging
        if (time.time() - self.time_onclick) < self.MAX_CLICK_LENGTH:
            # Get position of mouse
            x = event.xdata

            # Find nearest point
            array = np.asarray(self.x_scree)
            self.idx = (np.abs(array - x)).argmin()

            # Start plot function
            self.plot()
        else:
            pass
        
        return
    
    def select_change_key(self, event):   
        # Detect key
        if event.key == "w":
            self.idx -= 1
        elif event.key == "e":
            self.idx += 1
        elif event.key == "a":
            # Save results PCA single background
            Image.fromarray(self.signal_arr).save(self.path + '\\Signal_PCA_denoised_n_' + str(int(self.idx)) + '_Integrate_(' + str(np.around(self.roi_2.left,decimals = 2)) + '-' + str(np.around(self.roi_2.right,decimals = 2)) + ')_Background_(' + str(np.around(self.roi_1.left,decimals = 2)) + '-' + str(np.around(self.roi_1.right,decimals = 2)) + ')_Background_' + str(self.background) + '.tiff')
            return              

        
        # Reset idx,if out of boundary
        if self.idx < 0:
            self.idx = 0
        elif self.idx > self.x_scree[-1]-1:
            self.idx = self.x_scree[-1]-1

        # Plot
        self.plot()
        
        return
    
    def plot(self):
        
        self.signal_arr = self.pca_denoise(self.s_averaged, self.background)
        
        self.p_pca_s_arr.set_data(self.signal_arr)
        
        # Set new point
        self.scatter_2.set_data(self.x_scree[self.idx],self.variance_pca[self.idx])
        
        # Set new title
        self.ax4.set_title('Number of components:' + str(self.idx))

        # Update figure
        self.fig.canvas.draw_idle()
        
        return
    
    def normalize(self, s):
        s_arr = s.data
        return (s_arr-np.amin(s_arr))/(np.amax(s_arr)-np.amin(s_arr))
    
    # Function for fitting the background with local background averaging (and linear combination of powerlaws) --> returns the signal
    def background_subtraction(self, gmodel, s_hs, roi_background, para_init, d = 0, s_raw = None):
        s_shape = s_hs.data.shape

        # x-vector for background fitting
        x_axes_b = np.arange(roi_background.left, roi_background.right, s_hs.axes_manager[-1].scale)
        x_axes = np.arange(s_hs.axes_manager[-1].offset, s_hs.axes_manager[-1].offset + s_hs.axes_manager[-1].scale*s_hs.axes_manager[-1].size, s_hs.axes_manager[-1].scale)

        # Crop signal for background fitting
        s_fit = s_hs.isig[roi_background].data

        # Flatten spectrum image for easier indexing
        s_flatten = s_fit.reshape(-1, s_fit.shape[-1])

        # Array, which saves signal
        s_residual = np.zeros(s_shape)

        for i in tqdm_notebook(range(0,s_shape[0])):
            for j in range(0,s_shape[1]):
                # Get average EELS spectrum from d-neighbors
                s_summed = self.cell_neighbors(s_flatten, s_shape, i, j, d=d)
                # Fit background for averaged EELS spectrum
                results = gmodel.fit(s_summed, params = para_init, x=x_axes_b)
                # Subtract background from the signal
                background_fitted = gmodel.eval(results.params, x=x_axes)
                
                if s_raw == None:
                    s_residual[i,j,:] = s_hs.data[i,j,:] - background_fitted
                else:
                    s_residual[i,j,:] = s_raw.data[i,j,:] - background_fitted
                    
        s_residuals = copy.deepcopy(s_hs)
        s_residuals.data = s_residual

        return s_residuals

    def pca_denoise(self, s_raw, background):
        
        if background:
            # Denoise residual
            s_pca_residual_denoised = s_raw.get_decomposition_model(int(self.idx))
            self.s_eels = s_pca_residual_denoised.deepcopy()
            
        else:
            
            # Denoise spectra
            s_pca_background = s_raw.get_decomposition_model(int(self.idx))
            self.s_eels = s_pca_background.deepcopy()
            # Remove PCA-denoised background from denoised data
            s_pca_residual_denoised = self.background_subtraction(self.gmodel, s_pca_background, self.roi_1, self.para_init, self.d)

        s_pca_signal =  s_pca_residual_denoised.isig[self.roi_2].integrate1D(-1)   
        s_pca_s_arr = self.normalize(s_pca_signal)
        
        return s_pca_s_arr

        
    
    # Following two function are used for local background averaging 
    def sliding_window(self, arr, window_size):
        """ Construct a sliding window view of the array"""
        arr = np.asarray(arr)
        window_size = int(window_size)
        if arr.ndim != 2:
            raise ValueError("need 2-D input")
        if not (window_size > 0):
            raise ValueError("need a positive window size")
        shape = (arr.shape[0] - window_size + 1,
                 arr.shape[1] - window_size + 1,
                 window_size, window_size)
        if shape[0] <= 0:
            shape = (1, shape[1], arr.shape[0], shape[3])
        if shape[1] <= 0:
            shape = (shape[0], 1, shape[2], arr.shape[1])
        strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
                   arr.shape[1]*arr.itemsize, arr.itemsize)
        return as_strided(arr, shape=shape, strides=strides)

    def cell_neighbors(self, s_flatten,s_shape, i, j, d):
        """Return d-th neighbors of cell (i, j)"""
        arr = np.arange(s_shape[0]*s_shape[1]).reshape(s_shape[0], s_shape[1])
        w = self.sliding_window(arr, 2*d+1)

        ix = np.clip(i - d, 0, w.shape[0]-1)
        jx = np.clip(j - d, 0, w.shape[1]-1)

        i0 = max(0, i - d - ix)
        j0 = max(0, j - d - jx)
        i1 = w.shape[2] - max(0, d - i + ix)
        j1 = w.shape[3] - max(0, d - j + jx)

        elements = w[ix, jx][i0:i1,j0:j1].ravel()

        s_sum = np.sum(s_flatten[elements,:],axis=0)/(len(elements))

        return s_sum
    
    def save_eels(self):
        self.s_eels.data = self.s_eels.data.astype(np.float32)
        self.s_eels.save(self.path + '\\EELS_denoised_n_' + str(self.idx) + '.rpl', encoding = 'utf8')
        
        

def saving_notebook(path, NOTEBOOK_FULL_PATH, name_notebook = '\\Post_processing.ipynb'):
    # Activate conda with the current environment
    path_env = os.path.dirname(os.path.abspath(sys.executable))    
    cmd_env = 'conda activate ' + path_env
    subprocess.call(cmd_env, shell=True)

    # Create pdf-file from the notebook

    # Path and name of notebook
    ipynb_path = os.path.dirname(os.path.realpath("__file__")) + name_notebook 

    # Execute command in windows cmd
    cmd = 'jupyter-nbconvert --to PDFviaHTML ' + ipynb_path
    subprocess.call(cmd, shell=True)

    # Shift the pdf-file to the results folder
    source = NOTEBOOK_FULL_PATH[:-5] + 'pdf'
    destination = path + '\\Post_Processing\\Documentation_PostProcessing.pdf'
    os.rename(source, destination)