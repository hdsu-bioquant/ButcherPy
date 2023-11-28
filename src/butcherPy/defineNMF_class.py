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
        - plot_rank_selection_metrics
        - plot_convergence
        - plot_H
        - plot_W
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
    def plot_rank_selection_metrics(self, 
                                    metric = 'frobenius', 
                                    save = False, 
                                    save_path = None):
        """
        Plot the rank selection metrics (frobenius)
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
        
        
        
        


