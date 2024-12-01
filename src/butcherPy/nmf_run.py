#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contains the main function (run_NMF) to run basic NMF in Pytorch

Created by Andres Quintero  
Reimplemented for Pytorch by Ana Luisa Costa and Leonie Boland
"""

# Dependencies
import os
import torch
import numpy as np
import calendar
import time
from src.butcherPy.multiplerun_NMF_class import multipleNMFobject
import anndata as ad
import pandas as pd
import warnings


def run_NMF(matrix, 
            rank, 
            n_initializations, 
            iterations, 
            seed, 
            stop_threshold=40, 
            nthreads=0, 
            **kwargs):
    
    """
    Iteratively runs NMF for one given rank with different initializations for input matrix.

    Parameters
    ----------
    matrix
        numpy array with two dimensions, initial matrix
    rank
        integer, factorisation rank (k)
    n_initializations
        integer, number of initializations to run NMF
    iterations
        integer, number of iterations to run NMF for each initialization
    seed
        integer, random seed selected
    stop_threshold
        integer, when to stop the iterations after convergence
    nthreads
        integer, number of threads, apply multi-threading if your system supports it

    Returns
    -------
    rank
        the factorisation rank
    H_num
       the exposure (H) matrix corresponding to the best NMF run
    W_num
        the signature (W) matrix corresponding to the best NMF run
    W_eval_num
        the signature (W) matrix corresponding to the best NMF run per initialization
    iter_to_conv
        the amount of iterations needed to get convergence for each initialization
    frobNorm
        the frobenius norm for each initialization at the end of the iterations (or after convergence)
    time_stamp
        start of NMF algorithm
    """
    # Set number of threads                
    torch.set_num_threads(nthreads)
    
    # Add timestamp of the run start
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    # Check type and shape of matrix
    if not type(matrix) == np.ndarray:
        raise TypeError(f"The matrix is supposed to be a numpy ndarray, not a {type(matrix)}.")
    else:
        if len(matrix.shape) != 2:
            raise ValueError(f"The matrix needs to be 2 dimensional, not {len(matrix.shape)} dimensional.")
        elif matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError(f"At least one of the 2 dimensions has no entry.")
        
    # Check the given rank
    if not type(rank) == int:
        raise TypeError(f"The rank has to be an integer, not {type(rank)}.")
    elif rank < 1:
        raise ValueError(f"The rank must at least be 1 or higher.")

    # NMF in tensorflow
    n = matrix.shape[0] # number of rows
    m = matrix.shape[1] # number of columns

    if n < rank and m < rank:
        warnings.warn("Be aware that the number of columns/samples and rows/genes are lower than the indicated ranks.")
    elif n < rank:
        warnings.warn("Be aware that the number of rows/genes is lower than the indicated ranks.")
    elif m < rank:
        warnings.warn("Be aware that the number of columns/samples is lower than the indicated ranks.")

    if matrix.shape[0] == 1 or matrix.shape[1] == 1:
        warnings.warn("Your input has only 1 entry in either of the dimensions. Be aware that the NMF algorithm might not be able to catch any patterns in the vector. Check if your matrix input has the shape you expected or if the NMF algorithm makes sense to use for your problem.")

    # Creates a constant tensor object from the matrix
    X = torch.tensor(matrix, dtype=torch.float32)
    #print(f"Initial matrix with {n} rows and {m} columns converted to tensor object.")
    
    # Initialization Metrics 
    frobNorm = []
    iter_to_conv = []
    W_eval = []
    
    ##-----------------------------------------------------------------------##
    ##                              N inits                                  ##
    ##-----------------------------------------------------------------------##
    # cycle through n initializations and choose best factorization
    if seed is not None:
      torch.manual_seed(seed)
    for init_n in range(n_initializations):
        
        ##-------------------------------------------------------------------##
        ##                  Initialize W and H matrices                      ##
        ##-------------------------------------------------------------------##
        # This is really where the magic happens: W and H are initialized
        # randomly, and then iteratively updated until convergence
        
        # Initialize uniform distribution (0 to 1)
        H = torch.empty(rank, m).uniform_(0, 1)
        W = torch.empty(n, rank).uniform_(0, 1)
        #print(H.data[0][1])
    
        ##-------------------------------------------------------------------##
        ##                     Initialize frob error                         ##
        ##-------------------------------------------------------------------##
        if init_n == 0 :            
            Best_frob = np.linalg.norm(X.numpy() - torch.matmul(W, H).numpy()) / np.linalg.norm(X.numpy())
            Best_H    = H
            Best_W    = W            
        # Calc. the relative Frobenius norm of the difference between a matrix X 
        # and the product of matrices W and H
        # This metric can be used as a sort of loss function
    
        ##-------------------------------------------------------------------##
        ##        Save initial max exposures in H matrices                   ##
        ##-------------------------------------------------------------------##
        oldExposures = torch.argmax(H, dim=0) # outputs indices of max values
        const = 0            
    
        ##-------------------------------------------------------------------##
        ##                          Run NMF                                  ##
        ##-------------------------------------------------------------------##
        for inner in range(iterations):
            ##---------------------------------------------------------------##
            ##                          Update H                             ##
            ##---------------------------------------------------------------##
            WTX  = torch.matmul(W.t(), X)
            WTW  = torch.matmul(W.t(), W)
            WTWH = torch.matmul(WTW.t(), H)
            newH = torch.div(torch.mul(H, WTX), WTWH)
            newH = torch.where(torch.isnan(newH), torch.zeros_like(newH), newH)
            H = newH # tensors are mutable, so no need to use assign in pytorch
    
            ##---------------------------------------------------------------##
            ##                          Update W                             ##
            ##---------------------------------------------------------------##
            XHT  = torch.matmul(X, H.t())
            WH   = torch.matmul(W, H)
            WHHT = torch.matmul(WH, H.t())
            newW = torch.div(torch.mul(W, XHT), WHHT)
            newW = torch.where(torch.isnan(newW), torch.zeros_like(newW), newW)
            W = newW
            
            ##---------------------------------------------------------------##
            ##                    Evaluate Convergence                       ##
            ##---------------------------------------------------------------##
            newExposures = torch.argmax(H, axis=0)
            if torch.all(torch.eq(oldExposures, newExposures)).__invert__():
                oldExposures = newExposures
                const = 0
            else:
                const += 1
                if const == stop_threshold:
                    #print(f"NMF converged after {inner} iterations")
                    break
        
        ##-------------------------------------------------------------------##
        ##         Evaluate if best factorization initialization             ##
        ##-------------------------------------------------------------------##
        frobInit = torch.linalg.norm(X - torch.matmul(W, H)) / torch.linalg.norm(X)
        
        # Append to list of initialization metrics
        frobNorm.append(frobInit)
        iter_to_conv.append(inner+1)
        W_eval.append(W)
        
        if frobInit < Best_frob :
            Best_frob = frobInit
            Best_H    = H
            Best_W    = W
    
    ##-----------------------------------------------------------------------##
    ##             Convert to numpy, transpose and return                    ##
    ##-----------------------------------------------------------------------##
    
    W_num  = Best_W.numpy()
    H_num  = Best_H.numpy()

    frobNorm    = [i.numpy() for i in frobNorm]
    W_eval_num  = [i.numpy() for i in W_eval]
        
    # Return the relevant values for the NMF run with the given rank
    # rank: the factorisation rank
    # H_num: the exposure (H) matrix corresponding to the best NMF run
    # W_num: the signature (W) matrix corresponding to the best NMF run
    # W_eval_num: a list of the signature (W) matrix corresponding to the results of each initialization
    # iter_to_conv: the amount of iterations needed to get convergence for each initialization
    # frobNorm: the frobenius norm for each initialization at the end of the iterations (or after convergence)
    # time_stamp: start of NMF algorithm
    return rank, H_num, W_num, W_eval_num, iter_to_conv, frobNorm, time_stamp


# Run the basic NMF algorithm for multiple factorisation ranks
def multiple_rank_NMF(matrixobj, 
            ranks, # list of ks
            n_initializations, 
            iterations, 
            seed, 
            stop_threshold=40, 
            nthreads=0, 
            **kwargs):
    
    """
    Iteratively runs NMF for multiple different factorisation ranks.

    Parameters
    ----------
    matrixobj
        initial matrix
            - numpy array with two dimensions
            - AnnData object
            - pandas DataFrame
    ranks
        list of integers, factorisation ranks (k)
    n_initializations
        integer, number of initializations to run NMF
    iterations
        integer, number of iterations to run NMF for each initialization
    seed
        integer, random seed selected
    stop_threshold
        integer, when to stop the iterations after convergence
    nthreads
        integer, number of threads, apply multi-threading if your system supports it
    """

    if type(matrixobj) == np.ndarray:
        matrix = matrixobj
        rows = [f"Gene_{(i+1):d}" for i in range(matrix.shape[0])]
        columns = [f"Sample_{(i+1):d}" for i in range(matrix.shape[1])]
        
    if type(matrixobj) == ad.AnnData:
        matrix = np.transpose(matrixobj.X)
        if type(matrix) != np.ndarray:
            print("The AnnData matrix is converted to a numpy array.")
            matrix = matrix.toarray()
        rows = matrixobj.var_names
        columns = matrixobj.obs_names

    if type(matrixobj) == pd.DataFrame:
        matrix = matrixobj.to_numpy()
        rows = matrixobj.index.tolist()
        columns = matrixobj.columns.tolist()

    if type(rows) != list:
        rows = rows.tolist()
    if type(columns) != list:
        columns = columns.tolist()
    # Save the input matrix and a few properties in a dictionary
    input_matrix = {"gene_expression": matrix, "genes": rows, "samples": columns, "dim": matrix.shape}

    if type(ranks) != list:
        raise TypeError(f"Expected parameter of type list, but got {type(ranks).__name__}")
    
    NMF_result = []
    run_settings = []

    # Call the run_NMF function for each rank and save the results
    for k in ranks:
        print("Factorization rank: ", k)
        run_settings.append({"rank": k,
                            "n_initializations": n_initializations,
                            "iterations": iterations,
                            "seed": seed,
                            "stop_threshold": stop_threshold,
                            "nthreads": nthreads,
                            "kwargs": kwargs})
        
        NMF_result.append(dict(zip(['rank', 'H', 'W', 'W_eval_num', 'final_iterations', 'frobenius', 'time_stamp'], run_NMF(matrix, k, n_initializations, iterations, seed, stop_threshold, nthreads, **kwargs))))
        
        iters = list(map(lambda res: res['final_iterations'], NMF_result))[-1]
        print("NMF converges after {} iterations for the different initializations.".format(', '.join(map(str, iters))))

    # Extract the different values from all results from the different factorisation rank NMF runs
    Ws = list(map(lambda res: res['W'], NMF_result))
    Hs = list(map(lambda res: res['H'], NMF_result))
    frob_errors = list(map(lambda res: res['frobenius'], NMF_result))
    W_evals = list(map(lambda res: res['W_eval_num'], NMF_result))

    # Save the runs as a NMF object
    NMF = multipleNMFobject(input_matrix,
                            run_settings,
                            ranks,
                            Ws,
                            Hs,
                            frob_errors,
                            W_evals,
                            'basic')
    
    return NMF


