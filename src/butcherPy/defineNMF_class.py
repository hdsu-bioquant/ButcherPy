#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Containing the NMFobject class and its features
Created by Ana Luisa Costa
"""

import matplotlib.pyplot as plt
import numpy as np

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
    def calc_rank_selection_metrics(self, 
                                    metric = 'all'):
        """
        Calculates the rank selection metrics for each k. 
        Metrics include:
            - Frobenius norm (smaller is better)
            - Amari Distance (smaller is better)
            - Silhouette Width (larger is better)
            - Cophenetic Correlation Coefficient (larger is better)
        """
        
        
    
    @ property
    def plot_rank_selection_metrics(self, 
                                    metric = 'all', 
                                    save = False, 
                                    save_path = None):
        """
        Plot the rank selection metrics (frobenius) for each k
        """
 
        # Define the x axis
        x = np.arange(1, len(self.frobenius) + 1)
        
        # Plot the metrics
        plt.plot(x, self.frobenius, label = 'Frobenius')
        
        # Add labels
        plt.xlabel('Rank')
        plt.ylabel('Metric')
        
        # Add legend
        plt.legend()
        
        # Save the plot
        if save == True:
            plt.savefig(save_path)
        
        # Show the plot
        plt.show()
        
        
        
        


