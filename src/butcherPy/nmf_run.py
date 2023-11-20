#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contains the main functions needed to run NMF in TensorFlow

Created by Andres Quintero  
Reimplemented in Python by Ana Luisa Costa
"""

# Dependencies
import os
import tensorflow as tf

# Control tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

# Define the NMF_tensor_py function
def NMF_tensor_py(matrix, 
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
      rank: factorisation rank
      n_initializations: 
      iterations: 
      seed: random seed selected
      stop_threshold:
      nthreads: 
    """
    # Set number of threads                
    tf.config.threading.set_intra_op_parallelism_threads(nthreads)
    tf.config.threading.set_inter_op_parallelism_threads(nthreads)
    
    # NMF in tensorflow
    n = matrix.shape[0] # number of rows
    m = matrix.shape[1] # number of columns    
    # Creates a constant tensor object from the matrix
    X = tf.constant(matrix, name = ("X"), dtype=tf.float32)
    
    # Initialization Metrics 
    frobNorm = []
    iter_to_conv = []
    W_eval = []
    
    ##-----------------------------------------------------------------------##
    ##                              N inits                                  ##
    ##-----------------------------------------------------------------------##
    # cycle through n initializations and choose best factorization
    if seed is not None:
      tf.random.set_seed(seed)
    for init_n in range(n_initializations):
        
        ##-------------------------------------------------------------------##
        ##                  Initialize W and H matrices                      ##
        ##-------------------------------------------------------------------##
        initializer = tf.random_uniform_initializer(minval=0, maxval=1)
        
        H = tf.Variable(initializer(shape=[rank, m]), name="H")
        W = tf.Variable(initializer(shape=[n, rank]), name="W")
    
        ##-------------------------------------------------------------------##
        ##                     Initialize frob error                         ##
        ##-------------------------------------------------------------------##
        if init_n == 0 :            
            Best_frob = tf.linalg.norm(X - tf.matmul(W, H)) / tf.linalg.norm(X)
            Best_H    = H
            Best_W    = W            
    
        ##-------------------------------------------------------------------##
        ##        Save initial max exposures in H matrices                   ##
        ##-------------------------------------------------------------------##
        #Hts = [tf.add(H, Hv) for Hv in Hvs]
        #oldExposures = tf.concat([tf.math.argmax(Ht, axis=1) for Ht in Hts], 0)
        oldExposures = tf.math.argmax(H, axis=0)
        const = 0            
    
        ##-------------------------------------------------------------------##
        ##                          Run NMF                                  ##
        ##-------------------------------------------------------------------##
        for inner in range(iterations):
            ##---------------------------------------------------------------##
            ##                          Update H                             ##
            ##---------------------------------------------------------------##
            WTX  = tf.matmul(W, X, transpose_a=True)
            WTW  = tf.matmul(W, W, transpose_a=True)
            WTWH = tf.matmul(WTW, H, transpose_a=True)
            newH = tf.math.divide(tf.math.multiply(H, WTX), WTWH)
            newH = tf.where(tf.math.is_nan( newH ), tf.zeros_like( newH ), newH )
            update_H = H.assign(newH)
    
            ##---------------------------------------------------------------##
            ##                          Update W                             ##
            ##---------------------------------------------------------------##
            XHT  = tf.matmul(X, H, transpose_b=True)
            WH   = tf.matmul(W, H)
            WHHT = tf.matmul(WH, H, transpose_b=True)
            newW = tf.math.divide(tf.math.multiply(W, XHT), WHHT)
            newW = tf.where(tf.math.is_nan( newW ), tf.zeros_like( newW ), newW )
            update_W = W.assign(newW)
            
            ##---------------------------------------------------------------##
            ##                    Evaluate Convergence                       ##
            ##---------------------------------------------------------------##
            newExposures = tf.math.argmax(H, axis=0)
            if tf.reduce_all(tf.math.equal(oldExposures, newExposures)).__invert__():
                oldExposures = newExposures
                const = 0
            else:
                const += 1
                #print(f'new eval diff, {const}')
                if const == stop_threshold:
                    #print(f"NMF converged after {i} iterations")
                    break
        
        ##-------------------------------------------------------------------##
        ##         Evaluate if best factorization initialization             ##
        ##-------------------------------------------------------------------##
        frobInit = tf.linalg.norm(X - tf.matmul(W, H)) / tf.linalg.norm(X)
        # Append to list of initialization metrics
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
    W_num  = Best_W.numpy()
    H_num  = Best_H.numpy()

    frobNorm    = [i.numpy() for i in frobNorm]
    W_eval_num  = [i.numpy() for i in W_eval]
    
    return W_num, H_num, iter_to_conv, frobNorm, W_eval_num
    #frobNorm = tf.linalg.norm(X - tf.matmul(W, H)) / tf.linalg.norm(X)
    #return W.numpy(),H.numpy(), i, frobNorm.numpy()

