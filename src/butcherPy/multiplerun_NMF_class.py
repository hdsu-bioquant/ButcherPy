#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Containing the NMFobject class and its features
Created by Ana Luisa Costa and Leonie Boland
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk


## DEFINE THE NMF OBJECT CLASS -------------------------------------------------#
class multipleNMFobject:
    """
    Class of the NMF object resulting multiple NMF runs with possibily different ranks
    Includes functions for visualisation and summaries of the results:
    """
    
    def __init__(self, 
                 input_matrix,
                 NMF_run_settings,
                 ranks,
                 WMatrix,
                 HMatrix,
                 frobenius,
                 NMF_type = 'basic', # possible values would be basic, regularised, integrative, joint, or temporal
                 #k = None, 
                 #H = None, 
                 #Hviews = None, # only for iNMF
                 #W = None, 
                 #W_eval = None, 
                 #final_iterations = None, 
                 #frobenius = None,
                 #timestamp = None
                 ):
        self.input_matrix = input_matrix
        self.NMF_run_settings = NMF_run_settings
        self.ranks = ranks
        self.WMatrix = WMatrix
        self.HMatrix = HMatrix
        self.NMF_type = NMF_type
        self.frobenius = frobenius
        #self.k = k
        #self.H = H
        #self.Hviews = Hviews
        #self.W = W
        #self.W_eval = W_eval
        #self.final_iterations = final_iterations
        #self.frobenius = frobenius
        #self.timestamp = timestamp

    
    #------------------------------------------------------------------------------#
    #                               NMF Normalization                              #
    #------------------------------------------------------------------------------#
        
    def normalise_W(self, ranks="all"):
        """
        Normalises the W matrix signatures for the indicated ranks (sum of each column = 1)
        If the W matrix was already normalised, isn't done again
            ranks: Determines the W matrix to normalize by the rank
        """       

        if ranks=="all":
            index = list(range(len(self.ranks)))
        else:
            # If ranks are specified (as a list) search the corresponding indices of the runs
            index = []
            for i, settings in enumerate(self.NMF_run_settings):
                if settings["rank"] in ranks:
                    index.append(i)

        if len(index)==0:
            print("There was no run conducted with the indicated rank(s)", ranks)
        else:
            # Do the normalisation for the previously identified indices
            for i in index:

                column_sums = self.WMatrix[i].sum(axis=0) 
                # If the W matrix was already normalised, don't normalise again
                if np.all(np.isclose(column_sums, 1)) == True:
                    print('W matrix was already normalised for W matrix with index', i)
                            
                else:
                    W_norm = self.WMatrix[i]/column_sums
                    # Transform the H matrix in a way that the multiplication of the
                    # normalised W and the new H is the same as before
                    H_new = self.HMatrix[i]*column_sums[:, np.newaxis]
                    
                    self.WMatrix[i] = W_norm
                    self.HMatrix[i] = H_new

    def normalise_H(self, ranks="all"):
        """
        Normalises the H matrix exposures for the given ranks (sum of each column = 1)
        If the W matrix was already normalised, isn't done again
            ranks: Determines the W matrix to normalize by the rank
        """       

        if ranks=="all":
            index = list(range(len(self.ranks)))
        else:
            # If ranks are specified (as a list) search the corresponding indices of the runs
            index = []
            for i, settings in enumerate(self.NMF_run_settings):
                if settings["rank"] in ranks:
                    index.append(i)

        if len(index)==0:
            print("There was no run conducted with the indicated rank(s)", ranks)
        else:
            # Do the normalisation for the previously identified indices
            for i in index:

                row_sums = self.HMatrix[i].sum(axis=1)
                
                # If the H matrix was already normalised, don't normalise again
                if np.all(np.isclose(row_sums, 1)) == True:
                    print('H matrix was already normalised for H matrix with index', i)
                            
                else:
                    H_norm = self.HMatrix[i]/row_sums[:, np.newaxis]
                    # Transform the W matrix in a way that the multiplication of the
                    # normalised H and the new W is the same as before
                    W_new = self.WMatrix[i]*row_sums
                    
                    self.HMatrix[i] = H_norm
                    self.WMatrix[i] = W_new

    #@ property
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
        
        # Initialise the signatures
        signatures = []
        
        # Verifies if the W matrix was already normalised
        # if it was not, normalise the W matrix
        self.normalise_W#(self.W)

        #### IF STATEMENT ADDED BY ME
        # Select the k signatures
        if k != "all":
            W_sign = [[row[i] for i in k] for row in self.W]
            for i in range(len(W_sign)):
                if np.max(W_sign[i], axis = 0) >= exposure_threshold:
                    signatures.append(np.argmax(W_sign[i], axis = 0))
                else:
                    signatures.append(None)
        else:
            #### THIS WAS ORIGINAL CODE
            # Assign the observations to the signatures
            for i in range(len(self.W)):
                if np.max(W_sign[i], axis = 0) >= exposure_threshold:
                    signatures.append(np.argmax(self.W[i], axis = 0))
                else:
                    signatures.append(None)
            
        # Update the signatures
        self.signatures = signatures
    
    #def compute_SignatureFeatures(self):



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
            #dotp = np.abs(np.dot(self.W_eval[i], np.linalg.inv(self.H[i])))
            # Compute the sum over each row and each column
            #sum_rows = np.sum(dotp, axis=0)
            #sum_cols = np.sum(dotp, axis=1)
            # Compute the maximum value over each row and each column
            #max_rows = np.max(dotp / sum_rows, axis=0)
            #max_cols = np.max(dotp / sum_cols, axis=1)
            # Compute the distance            
            #amari_distance =  (np.sum(max_rows) + np.sum(max_cols)) / (2.0 * dotp.shape[0])
            #amari.append(amari_distance)
            
            # Calculate the silhouette width between the signatures
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
        
        
        
        


