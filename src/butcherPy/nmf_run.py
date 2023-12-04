#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contains the main function (run_NMF) to run basic NMF in Pytorch

Created by Andres Quintero  
Reimplemented for Pytorch by Ana Luisa Costa
"""

# Dependencies
import os
import torch
import numpy as np
from defineNMF_class import NMFobject


# Define the NMF_tensor_py function
def run_NMF(matrix, 
            rank, # k or common dim
            n_initializations, 
            iterations, 
            seed, 
            stop_threshold=40, 
            nthreads=0, 
            **kwargs):
    
    """
    Iteratively runs NMF to choose best factorisation rank for input matrix V
      matrix: initial V matrix
      rank: factorisation rank (k)
      n_initializations: number of initializations to run NMF
      iterations: number of iterations to run NMF for each initialization
      seed: random seed selected
      stop_threshold: when to stop the iterations after convergence
      nthreads: apply multi-threading if your system supports it
    """
    # Set number of threads                
    torch.set_num_threads(nthreads)
    
    # NMF in tensorflow
    n = matrix.shape[0] # number of rows
    m = matrix.shape[1] # number of columns    
    # Creates a constant tensor object from the matrix
    X = torch.tensor(matrix, dtype=torch.float32)
    print(f"Initial matrix with {n} rows and {m} columns converted to tensor object.")
    
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
        print(H.data[0][1])
    
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
                #print(f'new eval diff, {const}')
                if const == stop_threshold:
                    print(f"NMF converged after {inner} iterations")
                    break
        
        ##-------------------------------------------------------------------##
        ##         Evaluate if best factorization initialization             ##
        ##-------------------------------------------------------------------##
        frobInit = torch.linalg.norm(X - torch.matmul(W, H)) / torch.linalg.norm(X)
        
        # Append to list of initialization metrics
        print("Appending results to the list of metrics.")
        frobNorm.append(frobInit)
        iter_to_conv.append(inner+1)
        W_eval.append(W)
        
        if frobInit < Best_frob :
            #print('is less')
            Best_frob = frobInit
            Best_H    = H
            Best_W    = W
        #x = frobInit = tf.linalg.norm(X - tf.matmul(Best_W, Best_H)) / tf.linalg.norm(X)
        #print("Best frob:", x.numpy())
        #print("Current frob", frobInit.numpy())
        #fb = tf.reduce_sum(fb, 0)
    
    ##-----------------------------------------------------------------------##
    ##             Convert to numpy, transpose and return                    ##
    ##-----------------------------------------------------------------------##
    print("Preparing results to output.")
    W_num  = Best_W.numpy()
    H_num  = Best_H.numpy()

    frobNorm    = [i.numpy() for i in frobNorm]
    W_eval_num  = [i.numpy() for i in W_eval]
    
    # Compile the results into a single NMF object
    NMF_out_o = NMFobject(k = rank,
                          H = H_num, 
                          W = W_num, 
                          W_eval = W_eval_num,
                          final_iterations = iter_to_conv, 
                          frobenius = frobNorm 
                          )
    
    print("Completed NMF run and compiled results into an NMF object.")
    # Return the NMF object specific to this run
    # return W_num, H_num, iter_to_conv, frobNorm, W_eval_num
    return NMF_out_o


# TEST
np.random.seed(123)
test_mat = np.random.rand(1000,6)
test_rank = 3
tn_initializations = 10
titerations = 100
tseed = 123
tstop_threshold = 40
tnthreads = 1

nmf_test = run_NMF(test_mat, test_rank, tn_initializations, titerations, tseed, tstop_threshold, tnthreads)

import pickle
pickle.dump(nmf_test, open("nmf_teste2.p", "wb")) 


