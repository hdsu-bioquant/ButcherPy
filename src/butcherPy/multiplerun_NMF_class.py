#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Containing the NMFobject class and its features
Created by Ana Luisa Costa and Leonie Boland
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
from scipy.spatial.distance import pdist, squareform
from sklearn_extra.cluster import KMedoids
import sklearn.metrics as metrics
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.special import expit  # This is the sigmoid function
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd

## DEFINE THE NMF OBJECT CLASS -------------------------------------------------#
class multipleNMFobject:
    """
    Class of the NMF object resulting from multiple NMF runs with possibily different ranks.

    Parameters
    ----------
    input_matrix
        a dictionary containing properties of the input matrix, at least the matrix itself in the key "gene_expression"
    NMF_run_settings
        a dictionary containing all input parameters (names of parameters as keys) to the NMF function, so that the run could be replicated
    ranks
        a list of factorisation ranks for which the NMF algorithm has been performed
    WMatrix
        a list of best W matrices found by NMF with the corresponding rank
    HMatrix
        a list of best H matrices found by NMF with the corresponding rank
    frobenius
        a list of frobenius norms corresponding to the W and H matrices 
    W_eval
        a list of a list of all best W matrices from the different initializations for the different factorisation ranks
    NMF_type
        the type of NMF performed as a string (options: "basic")
            
    Includes functions for normalization/regularization of the W and H matrices, calculating optimal k statistics and signature extraction:
    """
    
    def __init__(self, 
                 input_matrix,
                 NMF_run_settings,
                 ranks,
                 WMatrix,
                 HMatrix,
                 frobenius,
                 W_eval,
                 NMF_type = 'basic', # possible values would be basic, regularised, integrative, joint, or temporal
                 #Hviews = None, # only for iNMF
                 ):
        self.input_matrix = input_matrix
        self.NMF_run_settings = NMF_run_settings
        self.ranks = ranks
        self.WMatrix = WMatrix
        self.HMatrix = HMatrix
        self.NMF_type = NMF_type
        self.frobenius = frobenius
        self.W_eval = W_eval
        self.OptKStats = []
        self.OptK = []
        self.feature_contributions = pd.DataFrame()

    #------------------------------------------------------------------------------#
    #                     ACCESS H AND W MATRIX BY RANK                            #
    #------------------------------------------------------------------------------#
    
    def get_W(self, ranks="all"):
        """
        Return the W matrices of the given ranks.

        Parameters
        ----------
        ranks
            list of integer(s) or "all"
        """

        index = []
        found_rank = []
        if ranks=="all":
            index = list(range(len(self.ranks)))
        else:
            # If ranks are specified (as a list) search the corresponding indices of the runs
            for i, settings in enumerate(self.NMF_run_settings):
                if settings["rank"] in ranks:
                    index.append(i)
                    found_rank.append(settings["rank"])
        
        if len(index)==0:
            print("No NMF runs have been performed with the indicated ranks, try one of {}".format(', '.join(map(str, self.ranks))))
        elif ranks != "all" and len(index)!=len(ranks):
            print("Not for all ranks there exist a W matrix, only ranks {} were found".format(', '.join(map(str, found_rank))))
        
        return [self.WMatrix[i] for i in index]


    def get_H(self, ranks="all"):
        """
        Return the H matrices of the given ranks.

        Parameters
        ----------
        ranks
            list of integer(s) or "all"
        """

        index = []
        found_rank = []
        if ranks=="all":
            index = list(range(len(self.ranks)))
        else:
            # If ranks are specified (as a list) search the corresponding indices of the runs
            for i, settings in enumerate(self.NMF_run_settings):
                if settings["rank"] in ranks:
                    index.append(i)
                    found_rank.append(settings["rank"])
        
        if len(index)==0:
            print("No NMF runs have been performed with the indicated ranks, try one of {}".format(', '.join(map(str, self.ranks))))
        elif ranks != "all" and len(index)!=len(ranks):
            print("Not for all ranks there exist a H matrix, only ranks {} were found".format(', '.join(map(str, found_rank))))
        
        return [self.HMatrix[i] for i in index]
    

    #------------------------------------------------------------------------------#
    #                               NMF Normalization                              #
    #------------------------------------------------------------------------------#
        
    def normalise_W(self, ranks="all"):
        """
        Normalises the W matrix signatures for the indicated ranks (sum of each column = 1).

        Parameters
        ----------
        ranks
            list of integer(s) or "all"
        """       

        index = []
        if ranks=="all":
            index = list(range(len(self.ranks)))
        else:
            # If ranks are specified (as a list) search the corresponding indices of the runs
            for i, settings in enumerate(self.NMF_run_settings):
                if settings["rank"] in ranks:
                    index.append(i)

        if len(index)==0:
            print("There was no run conducted with the indicated rank(s), try one of {}".format(', '.join(map(str, self.ranks))))
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
                
    def regularize_W(self, ranks="all"):
        """
        Regularize the W matrix for the indicated ranks (range of the values between 0 and 1).

        Parameters
        ----------
        ranks
            list of integer(s) or "all"
        """ 

        index = []
        if ranks=="all":
            index = list(range(len(self.ranks)))
        else:
            # If ranks are specified (as a list) search the corresponding indices of the runs
            for i, settings in enumerate(self.NMF_run_settings):
                if settings["rank"] in ranks:
                    index.append(i)

        if len(index)==0:
            print("There was no run conducted with the indicated rank(s), try one of {}".format(', '.join(map(str, self.ranks))))
        else:

            for i in index:
                tempW =self.WMatrix[i]
                tempH = self.HMatrix[i]
                
                # Maximal exposure value per row
                normFactor = np.max(tempW, axis=0, keepdims=True)

                self.WMatrix[i] = tempW/normFactor
                self.HMatrix[i] = tempH*normFactor.T


    def normalise_H(self, ranks="all"):     
        """
        Normalises the H matrix exposures for the indicated ranks (sum of each row = 1).

        Parameters
        ----------
        ranks
            list of integer(s) or "all"
        """ 

        index = []
        if ranks=="all":
            index = list(range(len(self.ranks)))
        else:
            # If ranks are specified (as a list) search the corresponding indices of the runs
            for i, settings in enumerate(self.NMF_run_settings):
                if settings["rank"] in ranks:
                    index.append(i)

        if len(index)==0:
            print("There was no run conducted with the indicated rank(s), try one of {}".format(', '.join(map(str, self.ranks))))
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

    
    def regularize_H(self, ranks="all"):
        """
        Regularize the H matrix exposures for the indicated ranks (range of the values between 0 and 1).

        Parameters
        ----------
        ranks
            list of integer(s) or "all"
        """ 

        index = []
        if ranks=="all":
            index = list(range(len(self.ranks)))
        else:
            # If ranks are specified (as a list) search the corresponding indices of the runs
            for i, settings in enumerate(self.NMF_run_settings):
                if settings["rank"] in ranks:
                    index.append(i)

        if len(index)==0:
            print("There was no run conducted with the indicated rank(s), try one of {}".format(', '.join(map(str, self.ranks))))
        else:

            for i in index:
                tempW =self.WMatrix[i]
                tempH = self.HMatrix[i]
                
                # Maximal exposure value per row
                normFactor = np.max(tempH, axis=1, keepdims=True)

                self.WMatrix[i] = tempW*normFactor.T
                self.HMatrix[i] = tempH/normFactor


    #------------------------------------------------------------------------------#
    #                         OPTIMAL FACTORIZATION RANK                           #
    #------------------------------------------------------------------------------#
       
    def compute_OptKStats_NMF(self, ranks = "all"):
        """
        Computes several metrics for all the best W matrices resulting from the different initializations for the indicated ranks.
        The statistics include several evaluations of the frobenius norm, the silhouette width, the cophenetic coefficient and the amari distance.

        Parameters
        ----------
        ranks
            list of integer(s) or "all"
        """ 

        # First check if the statistics have already been computed for the indicated ranks
        already_comp_OptK = [OptK["rank"] for OptK in self.OptKStats]

        if ranks=="all":
            ranks = [self.NMF_run_settings[i]["rank"] for i in range(len(self.NMF_run_settings))]

        absent_ranks = [k for k in ranks if k not in already_comp_OptK]
        if len(absent_ranks) == 0:
            print("All the ranks indicated have already been evaluated to save the optimal k statistics.")
        else:
            index = []
            for i, settings in enumerate(self.NMF_run_settings):
                if settings["rank"] in absent_ranks:
                    index.append(i)

            if len(index)==0:
                print("There was no run conducted with the indicated rank(s), try one of {}".format(', '.join(map(str, self.ranks))))
 
        
            for i in index:
                #----------------------------------------------------------------------------#
                #                            Frobenius error stats                           #
                #----------------------------------------------------------------------------#
                frob_errors = self.frobenius[i]
                frob_min = np.nanmin(frob_errors)
                frob_sd = np.std(frob_errors)
                frob_mean = np.mean(frob_errors)
                frob_cv = frob_sd/frob_mean
            
                #----------------------------------------------------------------------------#
                #     compute Silhouette Width, Cophenetic Coeff and Amari Distances         #
                #----------------------------------------------------------------------------#
                W_list = self.W_eval[i]
                B = len(W_list)
                concat_matrix = np.hstack(W_list)
                dist_matrix =  squareform(pdist(concat_matrix.T, metric='cosine'))
                dist_matrix = np.round(dist_matrix, decimals = 14)
                
                if len(W_list) > 1:
                    #------------------------------------------------------------------------#
                    #                         compute Silhouette Width                       #
                    #------------------------------------------------------------------------#
                    my_pam = KMedoids(n_clusters=W_list[0].shape[1], metric = "precomputed", random_state = 0)
                    my_pam.fit(dist_matrix)
                    silhouette_scores = metrics.silhouette_samples(dist_matrix, my_pam.labels_, metric="precomputed")
                    sum_sil_width = np.sum(silhouette_scores)
                    mean_sil_width = np.mean(silhouette_scores)
                    
                    #------------------------------------------------------------------------#
                    #                         compute Cophenetic Coeff                       #
                    #------------------------------------------------------------------------#
                    my_hclust = linkage(dist_matrix, method="average")
                    dist_cophenetic = squareform(cophenet(my_hclust))
                    
                    # Distance matrices without diagonal elements
                    np.fill_diagonal(dist_matrix, np.nan)
                    dist_matrix = dist_matrix[~np.isnan(dist_matrix)]
                    
                    np.fill_diagonal(dist_cophenetic, np.nan)
                    dist_cophenetic = dist_cophenetic[~np.isnan(dist_cophenetic)]

                    copheneticCoeff = np.corrcoef(np.column_stack((dist_cophenetic, dist_matrix)), rowvar=False)[0, 1]
                    
                    #------------------------------------------------------------------------#
                    #                         compute Amari Distances                        #
                    #------------------------------------------------------------------------#
                    distance_list = []

                    for b in range(B-1):
                        for b_hat in range(b+1, B):
                            dist = amariDistance(W_list[b], W_list[b_hat])
                            distance_list.append(dist)

                    meanAmariDist = np.mean(distance_list)
                    
                else:
                    sum_sil_width = None
                    mean_sil_width = None
                    copheneticCoeff = None
                    meanAmariDist = None
                
                stats = {"rank": self.NMF_run_settings[i]["rank"],
                        "FrobError_min": frob_min,
                        "FrobError_mean": frob_mean,
                        "FrobError_sd": frob_sd,
                        "FrobError_cv": frob_cv,
                        "sumSilWidth": sum_sil_width,
                        "meanSilWidth": mean_sil_width,
                        "copheneticCoeff": copheneticCoeff,
                        "meanAmariDist": meanAmariDist}
                
                self.OptKStats.append(stats)

        return self.OptKStats
    

    def compute_OptK(self):
        """
        From the optimal k statistic computation evaluate the best suited factorisation rank.
        """

        if len(self.OptKStats) > 1:
            for OptK in self.OptKStats:
                if not isinstance(OptK, dict):
                    print("The optimal K statistics for some rank seems to not have the correct type, please run the function compute_OptKStats_NMF() for all ranks that you want to compare.")
                else:
                    expected_keys = {"rank", "copheneticCoeff", "meanAmariDist"}
                    if not expected_keys.issubset(OptK.keys()):
                        print("The optimal K statistics for some rank seem to have missing some keys, please run the function compute_OptKStats_NMF() for all ranks that you want to compare.")
        else:
            print("At least two ranks need to be evaluated by compute_OptKStat_NMF(), so that a comparison can be made.")

        cophenetic_values = [OptK['copheneticCoeff'] for OptK in self.OptKStats]
        meanAmari_values = [OptK['meanAmariDist'] for OptK in self.OptKStats]
        
        # Getting the indices of the runs that have lower (respectively higher) cophenetic (respectively amari) values than its neighbours.
        max_cophenetic_index = np.logical_and(np.roll(cophenetic_values, 1) <= cophenetic_values, np.roll(cophenetic_values, -1) <= cophenetic_values)
        min_amari_index = np.logical_and(np.roll(meanAmari_values, 1) >= meanAmari_values, np.roll(meanAmari_values, -1) >= meanAmari_values)
        
        max_cophenetic_index = np.where(max_cophenetic_index)
        min_amari_index = np.where(min_amari_index)

        # Intersect the indices and get the ranks for those indices as the optimal ranks.
        intersection = np.intersect1d(max_cophenetic_index, min_amari_index)
        OptKs = [self.OptKStats[i] for i in intersection]
        OptKs = [OptK["rank"] for OptK in OptKs]

        self.OptK = OptKs

        if len(self.OptK) == 1:
            print("The optimal factorisation rank from the given ranks is ", self.OptK[0])
        elif len(OptKs) == 0:
            print("No optimal K could be determined from the Optimal K stat.")
        else:
            print("The optimal factorisation rank from the given ranks are ", self.OptK)
        
        return OptKs

    def optK_Frob(self):
        """
        Returns the frobenius errors of all differently initialized run with the optimal factorization rank.
        """

        if len(self.OptK) == 0:
            print("No optimal K found, please call compute_OptK first.")
        else:
            print("Frobenius error from all initializations for the optimal K:")
            optFrob = [self.frobenius[i] for i in range(len(self.frobenius)) if self.NMF_run_settings[i]["rank"] in self.OptK]
        
            return optFrob
        
     
    #------------------------------------------------------------------------------#
    #                   SIGNATURE EXTRACTION AND COMPARISON                        #
    #------------------------------------------------------------------------------#

    def WcomputeFeatureStats(self, ranks = "all"):
        """
        Compute a matrix indicating for each feature if it contributes to a signature by running
        a k means with 2 clusters and assign features to the clusters. The clusters can be interpreted
        as contributing or not contributing to the signature.

        Parameters
        ----------
        ranks
            list of integer(s) or "all"
        
        Returns
        -------
        feature_contributions
            a dataframe with genes/features in the rows and signatures in the columns, possible values are
            0 "not contributing" and 1 "contributing"
        """

        # Avoid to do computations for ranks that have already been computed
        existing_ranks = []
        if self.feature_contributions.empty == False:
            col_names = self.feature_contributions.columns
            existing_ranks = list({int(name.split('_')[0][3:]) for name in col_names})

        index = []
        if ranks=="all":
            if len(existing_ranks)==0:
                index = list(range(len(self.ranks)))
            else:
                for i, settings in enumerate(self.NMF_run_settings):
                    if settings["rank"] not in existing_ranks:
                        index.append(i)
        else:

            # If ranks are specified (as a list) search the corresponding indices of the runs
            for i, settings in enumerate(self.NMF_run_settings):
                if settings["rank"] in ranks and settings["rank"] not in existing_ranks:
                    index.append(i)

        every_sig_df = pd.DataFrame()

        if len(index)==0:
            print("There was no run conducted with the indicated rank(s), try one of {}".format(', '.join(map(str, self.ranks))))
 
        for i in index:
            
            rank = self.NMF_run_settings[i]['rank']
            print("Start feature contribution determination for rank", rank)

            W = self.WMatrix[i]
            idx = np.sum(W, axis=1) == 0
            # Only keep features that contribute towards one or more signatures
            Wf = W[~idx, :]

            #----------------------------------------------------------------------------#
            #      Run k means over all rows and assign features to the clusters         #
            #----------------------------------------------------------------------------#
            def feature_clustering(x):
                x = expit(x)
                kmeans = KMeans(n_clusters=2, random_state=0).fit(x.reshape(-1, 1))
                centers = kmeans.cluster_centers_
                max_idx = np.argmax(centers)
                cluster_labels = kmeans.labels_
                return np.array([1 if label == max_idx else 0 for label in cluster_labels])

            ssf = np.apply_along_axis(feature_clustering, 1, Wf)

            # Have rows consiting of zeros for the non-contributing features
            sig_features = np.zeros_like(W, dtype = int)
            sig_features[~idx, :] = ssf

            # Use a dataframe to assign column and row names
            df_sig = pd.DataFrame(sig_features)
            df_sig.columns = ["Sig" + str(rank) + "_" + str(i+1) for i in range(df_sig.shape[1])]
            df_sig.index = self.input_matrix["genes"]#.tolist()
            #df_sig.index = ["Gene" + str(i) for i in range(df_sig.shape[0])]

            every_sig_df = pd.concat([every_sig_df, df_sig], axis=1)
        
        self.feature_contributions = pd.concat([self.feature_contributions, every_sig_df], axis=1)

        # Sort the columns by rank and then by j
        sorted_columns = sorted(self.feature_contributions.columns, key=lambda x: (int(x.split('_')[0][3:]), int(x.split('_')[1])))
        self.feature_contributions = self.feature_contributions[sorted_columns]

        return self.feature_contributions

    def signature_heatmap(self, path):

        sig_df = self.feature_contributions

        if not sig_df.empty:
            # For each signature get the genes that contribute to it
            gene_sets = []
            for col in sig_df.columns:
                gene_sets.append(set(sig_df.index[sig_df[col] == 1]))

            n = len(gene_sets)
            distance_matrix = np.zeros((n, n))

            # Calculating the Jaccard distance in between each of the gene sets
            for i in range(n):
                for j in range(n):
                    if i == j:
                        distance_matrix[i, j] = float("NaN")
                    else:
                        distance_matrix[i, j] = jaccardDistance(gene_sets[i], gene_sets[j])

            heatmap = sns.heatmap(distance_matrix, annot=True, xticklabels=sig_df.keys(), yticklabels=sig_df.keys())
            plt.yticks(rotation=0)
            plt.title("Jaccard Distance Heatmap")
            # Add border around the heatmap
            ax = plt.gca()
            for _, spine in ax.spines.items():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(0.5)

            cbar = heatmap.collections[0].colorbar
            cbar.ax.text(1.1, 0.02, 'High Similarity', ha='left', va='center', transform=cbar.ax.transAxes)
            cbar.ax.text(1.1, 0.98, 'Low Similarity', ha='left', va='center', transform=cbar.ax.transAxes)
            
            plt.show()
            plt.savefig(path)
            plt.close()
        else:
            print("The feature contribution to the signatures haven't been computed yet, please use WcomputeFeatureStats() to do so.")
        
        
def amariDistance(matrixA, matrixB):
    """
    Calculates the Amari distance between two matrices.
    The Amari distance is a similarity measure between two matrices.

    Parameters
    ----------
    matrixA
        a 2d numpy array of shape (M, K)
    matrixB
        a 2d numpy array of shape (M, K)
    
    Returns
    -------
    amari_dist
        the Amari distance between the two matrices, a lower value indicates higher similarity.
    """

    K = matrixA.shape[1]
    C = np.corrcoef(matrixA, matrixB, rowvar=False)[:K, K:]

    row_max = np.max(C, axis = 1)
    col_max = np.max(C, axis = 0)

    amari_dist = 1-(np.sum(row_max)+np.sum(col_max))/(2*K)
    return amari_dist


def jaccardDistance(sigA, sigB):
    """
    Calculates the Jaccard distance between two sets.
    The Jaccard distance is a measure of dissimilarity between two sets.

    Parameters
    ----------
    sigA
        a set containing elements
    sigB
        a set containing elements

    Returns
    -------
        the Jaccard distance between the two sets, a value of 0 indicates identical sets, while 
        a value of 1 indicates completely disjoint sets.
    """

    intersection = len(sigA & sigB)
    union = len(sigA|sigB)
    return 1-(intersection/union)
