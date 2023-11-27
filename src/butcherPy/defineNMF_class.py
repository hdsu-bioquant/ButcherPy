#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Containing the NMFobject class and its features
Created by Ana Luisa Costa
"""

class NMFobject:
    """
    Class of the NMF object resulting from a single NMF run with multiple ranks
    Includes functions for visualisation and summaries of the results:
        - plot_rank_selection_metrics
        - plot_convergence
        - plot_H
        - plot_W
    """
    
    def __init__(self, k=None, n_initializations=None, iterations=None):
        self.k = k


