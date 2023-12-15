#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Containing the NMFobject class and its features
Created by Ana Luisa Costa
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk


## DEFINE THE NMF OBJECT CLASS -------------------------------------------------#
class NMFobject:
    """
    Class of the NMF object resulting from a single NMF run with multiple ranks
    Includes functions for visualisation and summaries of the results:
        - calc_rank_selection_metrics - calculates the rank selection metrics
        - plot_rank_selection_metrics - plots the rank selection metrics
        - plot_convergence - plots the progress of the NMF convergence
        - plot_H - heatmap of H matrix exposures
        - plot_W - heatmap of W matrix exposures
    """
    
    def __init__(self, 
                 NMF_type = 'basic', # possible values would be basic, regularised, integrative, joint, or temporal
                 k = None, 
                 H = None, 
                 W = None, 
                 W_eval = None, 
                 final_iterations = None, 
                 frobenius = None):
        self.k = k
        self.H = H
        self.W = W
        self.W_eval = W_eval
        self.final_iterations = final_iterations
        self.frobenius = frobenius
    
    @ property
    def define_signatures(self, 
                          k = 'all', 
                          exposure_threshold = 0.8):
        """
        Defines the k signatures from W matrix exposures
        Input:
            - k: selected rank (defaults as 'all')
            - exposure_threshold: threshold for exposure values (from 0 to 1)
            
        Exposure threshold: from which normalised exposure value should
        the observation be considered part of a signature at that rank.
        (default is 0.8, lower values may result in observations being assigned 
        to multiple signatures)
        """
        
    
    @ property
    def normalise_W(self):
        """
        Normalises the W matrix exposures for all ranks (sum of each column = 1)
        If the W matrix was already normalised, returns the W matrix
        """
        # Initialise the normalised W matrix
        W_norm = []
        
        # Verifies if the W matrix was already calculated for all ranks 
        W_norm_sumtest = []
        for i in range(len(self.W)):
            W_norm_sumtest.append(np.sum(self.W[i], axis = 0))
            
        # If the W matrix was already normalised, return the W matrix
        # else normalise the W matrix
        if np.all(np.isclose(W_norm_sumtest, 1)) == True:
            print('W matrix was already normalised')
            return self.W
        
        else:
            # Normalise the W matrix exposures by row
            for i in range(len(self.W)):
                W_norm.append(self.W[i] / np.sum(self.W[i], axis = 0))
            
            # Update the W matrix
            self.W = W_norm
        
    
    
    @ property
    def calc_rank_selection_metrics(self, 
                                    metric = 'all'):
        
        """
        Calculates the rank selection metrics for each k. 
        Metrics include:
            - Frobenius norm (lower is better)
            - Amari Distance (lower is better)
            - Silhouette Width (higher is better)
            - Cophenetic Correlation Coefficient (higher is better)
        """
        # Initialise the metrics 
        frobenius = []
        amari = []
        silhouette = []
        cophenetic = [] 
            
        # Calculate the metrics for each k
        for i in range(len(self.W_eval)):
            ### Calculate the frobenius norm
            frobenius.append(np.linalg.norm(self.W_eval[i] - self.H[i]))
            
            ### Calculate the amari distance
            dotp = np.abs(np.dot(self.W_eval[i], np.linalg.inv(self.H[i])))
            # Compute the sum over each row and each column
            sum_rows = np.sum(dotp, axis=0)
            sum_cols = np.sum(dotp, axis=1)
            # Compute the maximum value over each row and each column
            max_rows = np.max(dotp / sum_rows, axis=0)
            max_cols = np.max(dotp / sum_cols, axis=1)
            # Compute the distance            
            amari_distance =  (np.sum(max_rows) + np.sum(max_cols)) / (2.0 * dotp.shape[0])
            amari.append(amari_distance)
            
            # Calculate the silhouette width
            #silhouette.append(silhouette_width(self.W_eval[i], self.H[i]))
            
            # Calculate the cophenetic correlation coefficient
            #cophenetic.append(cophenetic_correlation(self.W_eval[i], self.H[i]))

    
        
    
    # @ property
    # def plot_rank_selection_metrics(self, 
    #                                 metric = 'all', 
    #                                 save = False, 
    #                                 save_path = None):
    #     """
    #     Plot the rank selection metrics (frobenius) for each k
    #     """
 
    #     # Define the x axis
    #     x = np.arange(1, len(self.frobenius) + 1)
        
    #     # Plot the metrics
    #     plt.plot(x, self.frobenius, label = 'Frobenius')
        
    #     # Add labels
    #     plt.xlabel('Rank')
    #     plt.ylabel('Metric')
        
    #     # Add legend
    #     plt.legend()
        
    #     # Save the plot
    #     if save == True:
    #         plt.savefig(save_path)
        
    #     # Show the plot
    #     plt.show()
        
        
        
        


