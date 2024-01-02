#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import os
import torch
import calendar
import time
from defineNMF_class import NMFobject


"""
Created by Andres Quintero
Reimplemented for Pytorch by Ana Luisa Costa

Integrative NMF is an NMF algorithm used to integrate multiple views of the data. 
The method decomposes matrices V into multiple W matrices, 
multiple H matrices and a common H matrix. This approach allows to
identify common and specific patterns across multiple views of the data.


This script includes the Pytorch implementation of the integrative NMF with
functions to update W and H matrices while integrating multiple views. 
Functions are:
    - run_iNMF: main function to run the integrative NMF algorithm
    - iNMF_update_H: update H shared matrix
    - iNMF_update_Ws: update W matrices
    - iNMF_update_Hs: update H matrices for each view
    - iNMF_obj_eval: evaluate the objective function

"""


##---------------------------------------------------------------------------##
##                               Update H                                    ##
##---------------------------------------------------------------------------##
def iNMF_update_sharedH(Xs, Ws, H, Hvs, Nviews):
    num   = tf.reduce_sum([tf.matmul(Xs[i], Ws[i], transpose_b=True) for i in range(Nviews)], 0)
    den_K = []
    for i in range(Nviews):
        Ht = tf.reduce_sum([H, Hvs[i]], 0)
        WW = tf.matmul(Ws[i], Ws[i], transpose_b=True)
        den_K.append(tf.matmul(Ht, WW))
    den    = tf.reduce_sum(den_K, 0)
    H_new  = tf.multiply(H,  tf.divide(num, den))
    H_new  = tf.where(tf.math.is_nan(H_new), tf.zeros_like(H_new), H_new)
    return H_new

##---------------------------------------------------------------------------##
##                    Update view specficic Ws                               ##
##---------------------------------------------------------------------------##                        
def iNMF_update_Ws(Xs, Ws, H, Hvs, Nviews, lamb, Sp):
    for i in range(Nviews):
        Ht     = tf.reduce_sum([H, Hvs[i]], 0)
        HtHt   = tf.matmul(Ht, Ht, transpose_a=True)
    
        HvHv   = tf.matmul(Hvs[i], Hvs[i], transpose_a=True)
        HsHs   = tf.reduce_sum([HtHt, tf.multiply(lamb,  HvHv)], 0)
        den    = tf.matmul(HsHs, Ws[i]) + Sp
        HX_den = tf.divide(tf.matmul(Ht, Xs[i], transpose_a=True), den)
        W_new  = tf.multiply(Ws[i], HX_den)
        W_new  = tf.where(tf.math.is_nan(W_new), tf.zeros_like(W_new), W_new)
        Ws[i].assign(W_new)
    return Ws

##---------------------------------------------------------------------------##
##                    Update view specficic Hs                               ##
##---------------------------------------------------------------------------##                        
def iNMF_update_viewH(Xs, Ws, H, Hvs, Nviews, lamb, Sp):
    for i in range(Nviews):
        Ht     = tf.reduce_sum([H, tf.multiply((1 + lamb), Hvs[i])], 0)
        WW     = tf.matmul(Ws[i], Ws[i], transpose_b=True)
        den    = tf.matmul(Ht, WW)
        XW     = tf.matmul(Xs[i], Ws[i], transpose_b=True)
        XW_den = tf.divide(XW, den)
        Hv_new = tf.multiply(Hvs[i], XW_den)
        Hv_new = tf.where(tf.math.is_nan(Hv_new), tf.zeros_like(Hv_new), Hv_new)
        Hvs[i].assign(Hv_new)
    return Hvs

##---------------------------------------------------------------------------##
##                   define objective function                               ##
##---------------------------------------------------------------------------##
def inmf_obj_eval(Xs, Ws, H, Hvs, Nviews, Sp, lamb):
    # Frobenius term
    #    frob_c = []
    #    for i in range(Nviews):
    #        frob_c.append(tf.linalg.norm(Xs[i] - tf.matmul(H, Ws[i])) / tf.linalg.norm(Xs[i]))
    #    frob_c = tf.reduce_sum(frob_c)
    frob_c   = []
    pen_c    = []
    sparse_c = []
    for i in range(Nviews):
        Ht = tf.add(H, Hvs[i])
        #frob_c.append(tf.linalg.norm(Xs[i] - tf.matmul(Ht, Ws[i])) / tf.linalg.norm(Xs[i]))
        frob_ci = tf.linalg.norm(Xs[i] - tf.matmul(Ht, Ws[i]))
        frob_c.append(tf.math.square(frob_ci))
        
        pen_ci = tf.math.square(tf.linalg.norm(tf.matmul(Hvs[i], Ws[i])))
        pen_c.append(lamb * pen_ci)
        
        sparse_c.append(Sp *tf.reduce_sum(Ws[i]))
        
    frob_c   = tf.reduce_sum(frob_c)
    pen_c    = tf.reduce_sum(pen_c)
    sparse_c = tf.reduce_sum(sparse_c)
    #return frob_c 
    return frob_c + pen_c + sparse_c

##---------------------------------------------------------------------------##
##                   define objective function                               ##
##---------------------------------------------------------------------------##
def inmf_max_exp(H, Hvs):
    Hts = [tf.add(H, Hv) for Hv in Hvs]
    max_exposures = tf.concat([tf.math.argmax(Ht, axis=1) for Ht in Hts], 0)
    return max_exposures 

##---------------------------------------------------------------------------##
##                        define main function                               ##
##---------------------------------------------------------------------------##
def iNMF_tensor_py(matrix_list, # instead of X as matrix
                   rank, 
                   n_initializations, 
                   iterations, 
                   Sp, 
                   stop_threshold=40, 
                   lamb = 10, 
                   **kwargs):
    
    # Add timestamp of the run start
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    
    # Transpose matrices from the initial list
    matrix_list = [np.transpose(Xv) for Xv in matrix_list]

    # Get number of views and number of samples
    Nviews = len(matrix_list)
    N = matrix_list[0].shape[0]
    Ms = [Xv.shape[1] for Xv in matrix_list]
    
    # Initialization Metrics 
    frobNorm = []
    iter_to_conv = []
    W_eval = []
    
    # X matrices to constant from initial list
    Xs = [torch.tensor(matrix_list[i], 
                       name = ("X" + str(i)), 
                       dtype=torch.float32) for i in range(Nviews)]
    
    ##-----------------------------------------------------------------------##
    ##                              N inits                                  ##
    ##-----------------------------------------------------------------------##
    if seed is not None:
      torch.manual_seed(seed)
    
    # cycle through n initializations and choose best factorization
    for init_n in range(n_initializations):
        ##-------------------------------------------------------------------##
        ##                     Initialize W matrices                         ##
        ##-------------------------------------------------------------------##
        Ws = [torch.empty(rank, Ms[i], 
                          names = ("W" + str(i)),
                          dtype=torch.float32).uniform_(0, 2) 
              for i in range(Nviews)]
        ##-------------------------------------------------------------------##
        ##                  Initialize shared H matrix                       ##
        ##-------------------------------------------------------------------##    
        H = torch.empty(N, rank, dtype=torch.float32, 
                        names="H") 
        ##-------------------------------------------------------------------##
        ##               Initialize view specific H matrices                 ##
        ##-------------------------------------------------------------------##
        Hvs = [torch.empty(N, rank, dtype=torch.float32,
                           names = ("Hview" + str(i))) for i in range(Nviews)]    
        ##-------------------------------------------------------------------##
        ##        Save initial max exposures in H matrices                   ##
        ##-------------------------------------------------------------------##
        oldExposures = inmf_max_exp(H, Hvs)
        const = 0       

        ##-------------------------------------------------------------------##
        ##                   Start matrix factorization                      ##
        ##-------------------------------------------------------------------##
        for inner in range(iterations):
            ##---------------------------------------------------------------##
            ##                          Update H                             ##
            ##---------------------------------------------------------------##
            H = iNMF_update_sharedH(Xs, Ws, H, Hvs, Nviews)
            ##---------------------------------------------------------------##
            ##                   Update view specficic Ws                    ##
            ##---------------------------------------------------------------##   
            Ws = iNMF_update_Ws(Xs, Ws, H, Hvs, Nviews, lamb, Sp)
            ##---------------------------------------------------------------##
            ##                    Update view specficic Hs                   ##
            ##---------------------------------------------------------------##                        
            Hvs = iNMF_update_viewH(Xs, Ws, H, Hvs, Nviews, lamb, Sp)

    
            ##---------------------------------------------------------------##
            ##                    Evaluate Convergence                       ##
            ##---------------------------------------------------------------##        
            newExposures = inmf_max_exp(H, Hvs)
    
            if torch.all(torch.eq(oldExposures, newExposures)).__invert__():
                oldExposures = newExposures
                const = 0
            else:
                const += 1
                if const == stop_threshold:
                    print(f"iNMF converged after {inner} iterations")
                    break
            
            #print("Best frob:", inmf_obj_eval(Xs, Ws, H, Hvs, Nviews, Sp, lamb).numpy())
        
        ##-------------------------------------------------------------------##
        ##                     Initialize frob error                         ##
        ##-------------------------------------------------------------------##
        if init_n == 0 :
            Best_frob = inmf_obj_eval(Xs, Ws, H, Hvs, Nviews, Sp, lamb)
            Best_H    = H
            Best_Ws   = Ws
            Best_Hvs   = Hvs
            
        #Hvs = [tf.Variable(initializer(shape=[N, rank]),
        #                   name= ("Hview" + str(i))) for i in range(K)]
        
        ##-------------------------------------------------------------------##
        ##         Evaluate if best factorization initialization             ##
        ##-------------------------------------------------------------------##
        frobInit = inmf_obj_eval(Xs, Ws, H, Hvs, Nviews, Sp, lamb)
        #        frobNorm_init = []
        #        for i in range(Nviews):
        #            #Ht = tf.add(H, Hvs[i])
        #            fb = tf.linalg.norm(Xs[i] - tf.matmul(H, Ws[i])) / tf.linalg.norm(Xs[i])
        #            frobNorm_init.append(fb)
        #        frobNorm_init = tf.reduce_sum(frobNorm_init)
        frobNorm.append(frobInit)
        iter_to_conv.append(inner+1)
        W_eval.append(torch.concat(Ws, 1))
        
        if frobInit < Best_frob :
            #print('is less')
            Best_frob = frobInit
            Best_H    = H
            Best_Ws   = Ws
        x = inmf_obj_eval(Xs, Ws, H, Hvs, Nviews, Sp, lamb)
        #print("Best frob:", x.numpy())
        #print("Current frob", frobInit.numpy())
    
    
    ##-----------------------------------------------------------------------##
    ##             Convert to numpy, transpose and return                    ##
    ##-----------------------------------------------------------------------##
    Ws_num  = [Wi.numpy().T for Wi in Best_Ws]
    H_num   = Best_H.numpy().T
    Hvs_num = [Hvi.numpy().T for Hvi in Best_Hvs]


    frobNorm = [i.numpy() for i in frobNorm]
    W_eval_num  = [i.numpy().T for i in W_eval]
    
    # Compile the results into a single NMF object
    iNMF_outo = NMFobject(k = rank,
                          NMF_type = 'basic',
                          H = H_num, 
                          Hviews = Hvs_num,
                          W = Ws_num, 
                          W_eval = W_eval_num,
                          final_iterations = iter_to_conv, 
                          frobenius = frobNorm,
                          timestamp = time_stamp 
                          )

    return iNMF_outo
