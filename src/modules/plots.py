import matplotlib.gridspec
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from scipy.cluster import hierarchy
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
import math
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import warnings
from scipy.optimize import nnls
import plotly.colors as pc
import re
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA

def H_heatmap(NMFobj, ranks, sample_annot = None, path_to_save = None):
    """
    Creates a heatmap of the H matrix and saves it at the given path.

    Parameters
    ----------
    NMFobj
        object from the multipleNMFobject class containing the H matrices
    ranks
        list of integers, the matrix resulting from the NMF run with these ranks will be shown in the heatmap
    sample_annot
        None, use the sample annotations that have been saved in the NMF object OR
        dictionary, with names of the annotations as keys and lists with the same length as the number of samples as the values
    path_to_save
        string, path to a directory, including the name of the file to save
    
    Returns
    -------
    annotation_colors
        dictionary of tuples with three values corresponding to the RGB values for the colors used for the sample annotations, mainly to use same colors in recovery plot
    rank_colors
        dictionary of tuples with three values corresponding to the RGB values for the colors used for the signature annotations, mainly to use same colors in recovery plot
    """

    param_dict = {"heatmap_shrink": [0.73, 0.55, 0.45, 0.37],
                  "heatmap_anchor_y": [0.8, 0.87, 0.92, 0.95],
                  "cmap_shrink": [0.8, 0.64, 0.513, 0.409],
                  "fig_top": [0.7, 0.7, 0.6, 0.6],
                  "fig_bottom": [-0.05, -0.2, -0.3, -0.6]}
    
    ########################################################################################################
    #               REGULARIZE H MATRICES FOR PROVIDED RANKS AND CONNECT THE MATRICES INTO ONE             #
    ########################################################################################################
    
    NMFobj.regularize_H(ranks)
    regH = NMFobj.get_H(ranks)

    ranks.sort()
    if len(ranks)>1:
        matrices = []
        for i in range(len(ranks)):
            matrices.append(regH[i])
        regH = [np.vstack(matrices)]

    # Extract the column annotations
    if sample_annot is None:
        column_annot = {"Samples": NMFobj.input_matrix["samples"]}
    else:
        column_annot = sample_annot

    annot_names = list(column_annot.keys())
    if len(annot_names) > 4:
        warnings.warn("This heatmap plot is only capable of handling up to four sample annotations. Thus, only the first four sample annotations of your provided sample_annot dictionary are used.")
        annot_names = annot_names[:4]

    column_annots = [column_annot[key] for key in annot_names]
    nr_annots = len(column_annots)-1

    # Reversing, so that the first dictionary entry is on top and the last on the bottom
    column_annots.reverse()
    annot_names.reverse()
    # Create the figure
    fig, ax = plt.subplots(figsize=(16, 8))

    ########################################################################################################
    #                           CALCULATE LINKAGE AND DEFINE DENDROGRAM                                    # 
    ########################################################################################################
 
    # Override the default linewidth
    matplotlib.rcParams['lines.linewidth'] = 0.8
    
    # Perform hierarchical clustering
    linkage = hierarchy.linkage(regH[0].transpose(), method='complete', metric='euclidean')
    dendro = hierarchy.dendrogram(linkage, no_plot=True)

    # Reorder data based on dendrogram leaves
    idx = dendro['leaves']
    data_ordered = regH[0][:, idx]
    column_annots = [[column_annot[i] for i in idx] for column_annot in column_annots]

    ########################################################################################################
    #                                         CREATE THE HEATMAP                                           # 
    ########################################################################################################
    
    heatmap = sns.heatmap(data_ordered, cmap='viridis', ax=ax, cbar=True, cbar_kws={"shrink": param_dict['heatmap_shrink'][nr_annots], "location": "right", "anchor": (-0.33, param_dict['heatmap_anchor_y'][nr_annots])})
    heatmap_pos = ax.get_position()
    plt.xticks([])
    yticks = ["Sig{}_{}".format(rank, i+1) for rank in ranks for i in range(rank)]
    plt.yticks(np.arange(data_ordered.shape[0]) + 0.5, yticks, rotation=0, fontsize=15)
    plt.xlabel('Samples', fontsize=23, labelpad=15)

    # Adapt the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_title('Exposure', pad=8, weight = 'bold', fontsize=15)

    cbar.set_ticks([0.2, 0.4, 0.6, 0.8])
    cbar.ax.tick_params(labelsize=15)

    cbar.ax.text(1.1, 0.02, 'Low', ha='left', va='center', transform=cbar.ax.transAxes, fontsize=15)
    cbar.ax.text(1.1, 0.98, 'High', ha='left', va='center', transform=cbar.ax.transAxes, fontsize=15)
    
    ########################################################################################################
    #  CREATE COLORBAR ON THE RIGHT OF HEATMAP WITH COLORS DEFINING THE RANK MEMBERSHIP OF THE SIGNATURES  # 
    ########################################################################################################
    
    rank_numbers = [rank for rank in ranks for _ in range(rank)]
    rank_labels = ["Rank "+str(rank) for rank in rank_numbers]
    
    # In case that only one rank is given, we want to create different colors for all signatures for better
    # recognition when creating the recovery plot
    if len(ranks)==1:

        palette2 = sns.color_palette("pastel")
        if len(rank_numbers) > len(palette2):
            palette2 += sns.color_palette("muted") 

        sigs = ["Sig{}_{}".format(rank, i+1) for rank in ranks for i in range(rank)]
        rank_colors = {}
        for k, sig in enumerate(sigs):
            rank_colors[str(sig)] = palette2[k]

        rank_mapped_colors = [rank_colors[sig] for sig in sigs]
        rank_mapped_colors.reverse()

    # In case that multiple ranks are given, define for each rank one color and for each signature belonging
    # to that rank a different shade of that color
    else:

        rank_labels2 = [f'Sig{rank}_{j+1}' for rank in ranks for j in range(rank)]

        unique_ranks = np.unique(rank_numbers)

        # Base palette with distinct colors for each unique rank
        palette2 = sns.color_palette("pastel")
        if len(np.unique(rank_labels)) > len(palette2):
            palette2 += sns.color_palette("muted") 

        # Dictionary to hold the color shades for each rank
        rank_colors = {}

        for i, rank in enumerate(unique_ranks):
            # Generate a gradient of shades for the current rank
            shades = sns.light_palette(palette2[i], n_colors=rank_numbers.count(rank)+1)
            shades = shades[1:]
            for j in range(rank_numbers.count(rank)):
                rank_colors[f'Sig{rank}_{j+1}'] = shades[j]

        # Define color for each rank annotation
        rank_mapped_colors = [rank_colors[rank] for rank in rank_labels2]
        rank_mapped_colors.reverse()

    cmap_rank = ListedColormap(rank_mapped_colors)
    cmap_rank = plt.colorbar(ScalarMappable(cmap=cmap_rank), ax=ax, orientation='vertical', location='right', anchor = (0, 1), shrink = param_dict['cmap_shrink'][nr_annots], ticks = [i+1 for i in range(len(ranks))]) # for four sample annotations
    cmap_rank.outline.set_visible(False)

    # Calculate tick positions, so that the label is in the middle of the corresponding color and add square brackets
    cbartop = 1
    sumrank = np.sum(ranks)
    cmap_rank.set_ticks([])
    ticks = np.unique(rank_labels).tolist()
    ticks.reverse()
    # Add square brackets
    cbartop = 1
    for rank in ranks:
        procent = rank / sumrank
        start = cbartop - procent
        mid = cbartop - (procent / 2)
        end = cbartop
        cbartop -= procent

        # Draw the square bracket
        cmap_rank.ax.plot([1.45, 1.45], [start, end], color='black', linewidth=1.5, transform=cmap_rank.ax.transAxes, clip_on=False)
        cmap_rank.ax.plot([1.05, 1.45], [end, end], color='black', linewidth=1.5, transform=cmap_rank.ax.transAxes, clip_on=False)
        cmap_rank.ax.plot([1.05, 1.45], [start, start], color='black', linewidth=1.5, transform=cmap_rank.ax.transAxes, clip_on=False)

        # Adjust tick label position
        cmap_rank.ax.text(1.65, mid, ticks.pop(), ha='left', va='center', transform=cmap_rank.ax.transAxes, fontsize=15)
    
    ########################################################################################################
    #                    CREATE COLORBAR ON TOP OF THE HEATMAP WITH SAMPLE ANNOTATIONS                     # 
    ########################################################################################################

    # Adjust layout to make space for colorbars
    fig.subplots_adjust(left=0.1, top=param_dict["fig_top"][nr_annots], right=0.85, bottom=param_dict["fig_bottom"][nr_annots]) # four sample annotations

    # Get the position of the heatmap axes
    heatmap_pos = ax.get_position()
    legends = []

    # Colorbars for column annotations
    for i, (annot, title) in enumerate(zip(column_annots, annot_names)):

        unique_annotations = np.unique(annot)

        palette = sns.color_palette("hls", n_colors=len(unique_annotations))
        if len(unique_annotations) > len(sns.color_palette("hls")):
            warnings.warn(f"There are more sample annotation groups in {title} than colors in the palette resulting in the usage of the same color for different groups.")
        annotation_colors = {label: color for label, color in zip(unique_annotations, palette)}
        mapped_colors = [annotation_colors[val] for val in annot]
        
        cmap = ListedColormap(mapped_colors)
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, orientation='horizontal', pad=0.05, aspect=40)
        cbar.set_ticks([])
        cbar.outline.set_visible(False)
        cbar.ax.text(1.007, 0.5, title, transform=cbar.ax.transAxes, ha='left', va='center', fontsize=12)
        
        cbar.ax.set_position([heatmap_pos.x0, heatmap_pos.y1 + 0.02 + i*0.04, heatmap_pos.width, 0.03])

        # Add legends for the two column annotations
        legends.append([mpatches.Patch(color=annotation_colors[label], label=label) for label in np.unique(annot)])
    
    # Needs reversing again for better visualization of the legend
    legends.reverse()
    annot_names.reverse()
    legend_len = 0 
    for i in range(len(legends)):
        legend = ax.legend(handles=legends[i], title=annot_names[i], loc='upper right', bbox_to_anchor=(1.68, 1.4-legend_len*0.3))
        legend_len+=legend.get_window_extent().height/100
        # Add legends to the figure
        ax.add_artist(legend)

    ########################################################################################################
    #                                          PLOT DENDROGRAM                                             # 
    ########################################################################################################

    ax_dendro = fig.add_axes([0.1, 0.85, 0.8, 0.1])
    # Adjust the position to be on top of the colormap corresponding to the sample annotations
    bbox_colorbar = cbar.ax.get_position()
    ax_dendro.set_position([bbox_colorbar.x0,  bbox_colorbar.y1, bbox_colorbar.width, 0.13])
    hierarchy.dendrogram(linkage, ax=ax_dendro, orientation='top', no_labels=True) #, link_color_func=lambda x: 'k') # removing the '#' would turn the lines of the dendrogram black

    ax_dendro.set_axis_off()

    ########################################################################################################
    #                                      FINAL PLOT WITH TITLE                                           # 
    ########################################################################################################

    plt.title('H Matrix', fontsize=35, weight = 'bold')
    if path_to_save != None:
        plt.savefig(path_to_save)
    plt.show()
    plt.close()

    return annotation_colors, rank_colors

def W_heatmap(NMFobj, ranks, sig_specific = True, sig_annot = None, path_to_save = None, sig_annot_recovery = False, sample_annot = None):
    """
    Creates a heatmap of the W matrix and saves it at the given path.

    Parameters
    ----------
    NMFobj
        object from the multipleNMFobject class containing the H matrices
    ranks
        list of integers, the matrix resulting from the NMF run with these ranks will be shown in the heatmap
    sig_specific
        boolean, whether only to show signature specific features or all of them
    sig_annot
        None, the signatures are numbered OR
        list of strings, the list must have the same length as number of columns and its elements are used as xtick labels
    path_to_save
        string, path to a directory, including the name of the file to save
    sig_annot_recovery
        boolean, whether the naming of the signatures shall be done with the help of the recovery plot or not
    sample_annot
        None, use the sample annotations that have been saved in the NMF object OR
        list, with the same length as the number of samples will be used as the sample annotations for the recovery plot and thus the naming of the signatures
    """    
    
    ########################################################################################################
    #               REGULARIZE W MATRICES FOR PROVIDED RANKS AND CONNECT THE MATRICES INTO ONE             #
    ########################################################################################################
    
    NMFobj.regularize_W(ranks)
    regW = NMFobj.get_W(ranks)

    matrices = []
    indices = []
    for i in range(len(ranks)):
        if sig_specific:
            # determine signature specific features and filter the W matrix for those genes
            sigspecifics = SignatureSpecificFeatures(NMFobj, ranks[i])
            genes = NMFobj.input_matrix['genes']
            indices.extend(genes.index(name) for name in sigspecifics)
        else:
            matrices.append(regW[i])

    indices = list(set(indices))
    for i in range(len(ranks)):
        matrices.append(regW[i][indices, :])

    regW = [np.hstack(matrices)]

    ########################################################################################################
    #                           CALCULATE LINKAGE AND DEFINE DENDROGRAM                                    # 
    ########################################################################################################
    
    # override the default linewidth
    matplotlib.rcParams['lines.linewidth'] = 0.8

    # perform hierarchical clustering on the rows
    row_clusters = hierarchy.linkage(regW[0], method = 'ward')

    # create a figure with 4 subplots (one for y-axis label, one for dendrogram, one for less space between dendrogram and heatmap, and one for heatmap)
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.1, 1, -0.22, 4])

    ########################################################################################################
    #                              CREATE AXIS FOR LABEL AND DENDROGRAM                                    # 
    ########################################################################################################

    # add the axis for the y-axis label
    ax_label = fig.add_subplot(gs[0])
    ax_label.axis('off')
    ax_label.text(0.5, 0.5, 'Features (e.g. genes)', rotation=90, va='center', ha='center', fontsize=23)

    # add the axis for the dendrogram
    ax_dendro = fig.add_subplot(gs[1])
    dendro = hierarchy.dendrogram(row_clusters, orientation='left', ax=ax_dendro, no_labels=True, link_color_func=lambda x: 'k')

    # remove the border around the dendrogram
    ax_dendro.spines['top'].set_visible(False)
    ax_dendro.spines['right'].set_visible(False)
    ax_dendro.spines['bottom'].set_visible(False)
    ax_dendro.spines['left'].set_visible(False)

    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])

    ########################################################################################################
    #                                           CREATE HEATMAP                                             # 
    ########################################################################################################

    # reorder the rows according to the clustering
    reordered_W = regW[0][dendro['leaves'], :]

    # create the heatmap axis
    ax_heatmap = fig.add_subplot(gs[3])

    # create the x ticks labels
    if sig_annot == None and sig_annot_recovery == False:
        # default ticks, just numbering the signatures
        xticks = ["Sig{}_{}".format(rank, i+1) for rank in ranks for i in range(rank)]
    elif sig_annot != None:
        # handle user defined signature annotation
        if sig_annot_recovery == True:
            warnings.warn("You have provided your own signature annotation and set sig_annot_recovery to True. The annotations you provided will be used, if you want to see the annotations infered from the recovery plot set sig_annot to None and sig_annot_recovery to True.")
        if type(sig_annot)==list and len(sig_annot) == sum(ranks):
            xticks = sig_annot
        else:
            warnings.warn("Your provided signature annotation is not of the correct form. It must be a list with its length being equal to the sum of the ranks.")
            xticks = ["Sig{}_{}".format(rank, i+1) for rank in ranks for i in range(rank)]
    else:
        # create ticks from the recovery plot
        annot_df = {}
        # for each rank the recovery plot and the p values must be computed independently
        for rank in ranks:
            if sample_annot == None:
                sample_annot = NMFobj.input_matrix["samples"]
            annot_groups = list(set(sample_annot))
            auc_dfs = recovery_plot(NMFobj, rank, sample_annot, path_to_save="recovery_for_Wannots")
            for i in range(len(auc_dfs[0])):
                # get the p values in one row for each signature
                pvals = [df["p"].iloc[i] for df in auc_dfs]
                annots = []
                for e, p in enumerate(pvals):
                    # add the sample annotation to the signatures annotation if the p value is small
                    if p < 0.05:
                        annots.append(annot_groups[e])
                annot_df[f"Sig{rank}_{i+1}"] = annots
        xticks = [annot_df[key] for key in list(annot_df.keys())]    
        xticks = ["\n".join(ticks) for ticks in xticks]

    heatmap = sns.heatmap(reordered_W, ax=ax_heatmap, cmap='inferno', cbar=True, xticklabels=xticks, yticklabels=False)
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), fontsize=14, rotation=90)
    
    # create top x ticks with the signature labels in case that they are not given as the "normal" x ticks
    top_xticks = ["Sig{}_{}".format(rank, i+1) for rank in ranks for i in range(rank)]
        
    if xticks != top_xticks:
        secax = ax_heatmap.secondary_xaxis('top')
        secax.set_xticks(np.arange(len(top_xticks)) + 0.5) 
        secax.set_xticklabels(top_xticks, fontsize=10)
        
    # adapt the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_title('Exposure', pad=12, weight = 'bold', fontsize=15, ha='left')

    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.ax.tick_params(labelsize=15)

    ax_heatmap.set_xlabel('Signatures', fontsize=23, labelpad=10)

    fig.suptitle('W Matrix', fontsize=35, weight = 'bold')

    plt.tight_layout()
    if path_to_save != None:
        plt.savefig(path_to_save)
    plt.show()
    plt.close()

def SignatureSpecificFeatures(NMFobj, rank):
    # Perform feature extraction
    NMFobj.WcomputeFeatureStats(ranks=[rank])
    gene_cont_df = NMFobj.feature_contributions.filter(regex=fr'^Sig{rank}')
    
    sig_feats = gene_cont_df.loc[gene_cont_df.sum(axis=1)==1]

    return list(sig_feats.index)
    #return sig_feats




def recovery_plot(NMFobj, rank, sample_annot = None, path_to_save = None):
    """
    Creates a recovery plot that shows the separation and identification of underlying gene expression patterns (signatures)
    associated with specific sample annotations.

    Parameters
    ----------
    NMFobj
        object from the multipleNMFobject class containing the H matrices
    rank
        integer, for this rank the recovery plot for the H matrix will be visualized
    sample_annot
        None, use the sample annotations that have been saved in the NMF object OR
        list, with the same length as the number of samples will be used as the sample annotations
    path_to_save
        string, path to a directory, including the name of the file to save   
    
    Returns
    -------
    auc_dfs
        list of dictionaries, containing the p values for each sample annotation group per signature, mainly for use in the W heatmap plot
    """

    # Extract the sample annotations
    if sample_annot is None:
        sample_annot = NMFobj.input_matrix["samples"]
        sample_annot_dict = {"Samples": NMFobj.input_matrix["samples"]}
    else:
        if type(sample_annot)==list:
            sample_annot_dict = {"Samples": sample_annot}
        elif type(sample_annot)==dict:
            sample_annot_dict = sample_annot
            sample_annot = sample_annot[list(sample_annot.keys())[0]]
            

    if rank not in NMFobj.ranks:
        raise ValueError("For the given rank no NMF run has been conducted.")
    
    # H heatmap for colors and combined visualization
    color_annot, rank_annot = H_heatmap(NMFobj, [rank], sample_annot=sample_annot_dict, path_to_save="H_recovery_plot")
    # Extract the H matrix
    matrix = NMFobj.get_H([rank])[0]

    ########################################################################################################
    #                               PREPARING THE RANKS FOR PLOTTING                                       # 
    ########################################################################################################

    # Get plots for all groups of sample annotations
    annot_groups = list(set(sample_annot))
    ranks_annot = []
    freq_annot = []
    auc_dfs = []

    matrix_shape = matrix.shape
    ranks = np.arange(matrix_shape[1])

    # Precompute frequency for each annotation group
    annot_group_freq = {annot_group: 1 / sample_annot.count(annot_group) for annot_group in annot_groups}

    for annot_group in annot_groups:

        # Compute ranks for all signatures and filter for samples that belong to a certain sample annotation
        # Create lists with the ranks of the samples (belonging to the annotation group)
        # Create lists with the height of the step for each rank, the frequency
        rank_lists = []
        freq_lists = []

        ranks_for_auc = []

        # Iterate over all rows (signatures) of the H matrix
        for j in range(matrix_shape[0]):
            
            # Extract the signature
            sig = matrix[j, :].tolist()

            # Enumerate the list to get pairs of (index, value)
            indexed_sig = list(enumerate(sig))
            # Sort the indexed list by the values
            sorted_indexed_sig = sorted(indexed_sig, key=lambda x: x[1])
            # Only keep the sorted indices to rearrange the annotations list
            sorted_indices = [index for index, _ in sorted_indexed_sig]
            
            # Rearrange the sample annotations accordingly
            sorted_annot = [sample_annot[i] for i in sorted_indices]
            # Create a mask filtering one particular sample annotation
            belonging_to_annot = [annot==annot_group for annot in sorted_annot]

            # Get the ranks, corresponding to a sample with the annotation we are looking for with the help of the mask
            relevant_sorted_ranks = [value for value, condition in zip(ranks, belonging_to_annot) if condition]
            ranks_for_auc.append(relevant_sorted_ranks)
            ranks_sig = [0] + relevant_sorted_ranks

            # Create the frequency list
            freq = annot_group_freq[annot_group]
            freq_list = np.arange(0, freq * (len(ranks_sig) - 1) + freq, freq).tolist()
            freq_list.append(freq_list[-1])

            ranks_sig.append(len(ranks)-1)

            rank_lists.append(ranks_sig)
            freq_lists.append(freq_list)

        ranks_annot.append(rank_lists)
        freq_annot.append(freq_lists)

        ########################################################################################################
        #                                   AUC AND PVALUE CALCULATION                                         # 
        ########################################################################################################
        means = []
        stds = []
        num_samples = len(sample_annot)
        for rks in ranks_for_auc:
            random_permutations = [np.random.choice(num_samples, size=len(rks), replace=False).tolist() for _ in range(500)]
            random_aucs = custom_auc(random_permutations, matrix.shape[1])
            
            means.append(np.mean(random_aucs))
            stds.append(np.std(random_aucs))
    
        auc_df = pd.DataFrame()
        auc_df["mean"] = means
        auc_df["std"] = stds
        auc_df["auc"] = custom_auc(ranks_for_auc, matrix_shape[1])
        auc_df["z"] = (auc_df["auc"]-auc_df["mean"])/auc_df["std"]
        auc_df["p"] = np.where(auc_df["z"] > 0, norm.sf(auc_df["z"]), norm.cdf(auc_df["z"]))
        auc_dfs.append(auc_df)

    ########################################################################################################
    #                                        CREATING THE PLOT                                             # 
    ########################################################################################################

    n_cells = len(annot_groups)
    n_cols = int(math.ceil(math.sqrt(n_cells)))
    n_rows = int(math.ceil(n_cells / n_cols))

    relation_rowcell = n_cols/n_rows

    # Create a figure and axis grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(int(relation_rowcell*15), 15), sharex=True, sharey=True)
    # Adjust the space between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.09) 
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < n_cells:

            colors = [rank_annot[str(sig)] for sig in list(rank_annot.keys())]
            aucdf = auc_dfs[idx]

            for sig_ranks, freqs, p, color in zip(ranks_annot[idx], freq_annot[idx], aucdf["p"], colors):
                if p < 0.05:
                    line_type = "solid"
                    line_width = 2
                else:
                    line_type = "dashed"
                    line_width = 1

                ax.step(sig_ranks, freqs, where='post', label='Stair Function', color=color, linewidth = line_width, linestyle = line_type)

            ax.grid(False)

            # Add diagonal line
            ys = np.linspace(0, 1, matrix.shape[1])
            ax.plot([i for i in range(len(ys))], ys, color='black')
            
            # Set titles with colorful backgrounds
            rect = patches.Rectangle((0, 1), 1, 0.05, transform = ax.transAxes, color = color_annot[annot_groups[idx]], clip_on=False)
            ax.add_patch(rect)

            ax.set_title(annot_groups[idx], fontsize=16-n_rows, color='black', pad=3)
            ax.title.set_position([.5, 1])
    
    # Adding common x-label and y-label to the whole figure
    fig.text(0.465, 0.04, 'Sorted Feature Position', ha='center', va='center', fontsize=18)
    fig.text(0.06, 0.5, 'Frequency', ha='center', va='center', rotation='vertical', fontsize=18)

    # Adding legend for significant p-values
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, linestyle='solid', label='Significant p-val<0.05: True'),
        Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Significant p-val<0.05: False')
    ]
    fig.legend(handles=legend_elements, fontsize = 16, loc = "lower right")#, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=14, ncol=2)
    
    # Generate legend elements for the colors
    color_legend_elements = [Line2D([0], [0], color=color, lw=2, label=name) for name, color in rank_annot.items()]

    # Adjust the layout to make room for the new legend
    fig.subplots_adjust(right=0.85)

    # Adding the new legend to the right of the plot
    fig.legend(handles=color_legend_elements, fontsize=16, loc='center right')

    # Remove not used subplots
    for idx in range(n_cells, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Recovery plots\nAssociation of metadata to every signature', fontsize=20, weight = "bold")
    plt.show()
    if path_to_save != None:
        plt.savefig(path_to_save)
    plt.close()

    return auc_dfs


def custom_auc(list_ranks, max=None):
    """
    Computes a custom rank-based Area Under the Curve (AUC) for a given list of ranks.

    Parameters
    ----------
    list_ranks: 
        list, were each element is a list of ranks
    max:
        int, the maximum rank value to consider OR
        None, the maximum rank value within each list will be used

    Returns
    -------
    all_auc
        list of computed AUC values for each list of ranks
    """
    all_auc = []
    current_max = max
    for rank in list_ranks:

        if max == None:
            current_max = np.max(rank)
        
        rank.sort()
        X = 0
        i = 0
        while (i < len(rank)) and (rank[i] <= current_max):
            X = X + (current_max-rank[i])
            i += 1

        rauc = X/(i-1)/current_max
        all_auc.append(rauc)

    return all_auc


def optK_plot(NMFobj, plot_metrics = ["FrobError", "FrobError_cv", "meanAmariDist", "sumSilWidth", "meanSilWidth", "copheneticCoeff"], path_to_save = None):
    """
    Creates a plot showing metrics to evluate the quality of the different factorization ranks.

    Parameters
    ----------
    NMFobj
        object from the multipleNMFobject class
    plot_metrics
        list of strings, a list containing the name of the metrics that shall be plotted
        all options are: "FrobError", "FrobError_min", "FrobError_mean", "FrobError_sd", "FrobError_cv", "meanAmariDist", "sumSilWidth", "meanSilWidth", "copheneticCoeff"
    path_to_save
        string, path to a directory, including the name of the file to save   
    """

    # titles for the plots
    custom_titles = {
        "FrobError": "Frobenius error",
        "FrobError_min": "Frobenius error minimum",
        "FrobError_mean": "mean Frobenius error",
        "FrobError_sd": "Frobenius error standard deviation",
        "FrobError_cv": "Coefficient of variation",
        "meanAmariDist": "Mean Amari distance",
        "sumSilWidth": "sum Silhouette width",
        "meanSilWidth": "mean Silhouette width",
        "copheneticCoeff": "Cophenetic coefficient"
    }

    valid_metrics = []
    for metric in plot_metrics:
        if metric not in list(custom_titles.keys()):
            print(f"{metric} is not supported, instead choose metrics from {list(custom_titles.keys())}")
        else:
            valid_metrics.append(metric)

    valid_metrics = list(set(valid_metrics))
        
    if not valid_metrics:
        warnings.warn("There is no valid metric in the list you provided! The default metrics are used for the plot instead.")
        valid_metrics = ["FrobError", "FrobError_cv", "meanAmariDist", "sumSilWidth", "meanSilWidth", "copheneticCoeff"]

    # create datafram with Frobenius error for all initializations
    all_frobs = NMFobj.frobenius
    ks = NMFobj.ranks
    frob_df = pd.DataFrame(all_frobs, columns=[f"Init_{i+1}" for i in range(len(all_frobs[0]))])
    frob_df["rank"] = ks

    frob_df = frob_df.melt(id_vars=["rank"], var_name="Initialization", value_name="Stat")
    frob_df["Metric"] = "FrobError"
    frob_df["Stat"] = frob_df["Stat"].apply(lambda x: float(x))

    
    # compute all other metric values
    optKstats = NMFobj.compute_OptKStats_NMF()

    # create dataframe with all metrics (except Frobenius error for all initializations)
    metric_dict = pd.DataFrame(optKstats)
    metric_dict = metric_dict.melt(id_vars=["rank"], var_name="Metric", value_name="Stat")

    # concatenate the two dataframes
    plot_data = pd.concat([frob_df, metric_dict])
    plot_data = plot_data[plot_data["Metric"].isin(valid_metrics)]

    # save in the dataframe if the metric is to be minimized or maximized
    metrics_to_minimize = ["FrobError", "FrobError_min", "FrobError_mean", "FrobError_sd", "FrobError_cv", "meanAmariDist"]
    metrics_to_maximize = ["sumSilWidth", "meanSilWidth", "copheneticCoeff"]

    # create a mapping dictionary
    metric_mapping = {metric: "minimized" for metric in metrics_to_minimize}
    metric_mapping.update({metric: "maximized" for metric in metrics_to_maximize})

    # add the new column based on the mapping dictionary
    plot_data['max_or_min'] = plot_data['Metric'].map(metric_mapping)

    # calculate the optimal factorization rank
    opt_k = NMFobj.compute_OptK()

    # START PLOTTING
    g = sns.FacetGrid(plot_data, col="Metric", col_wrap=2, sharex=False, sharey=False, hue="max_or_min")
    # show the optimal factorization rank by a line in all subplots
    for k in opt_k:
        g.map(plt.axvline, x=k, color='firebrick', linestyle='--')
    g.map(sns.scatterplot, 'rank', "Stat")
    g.add_legend(loc="center right", title = "Goal of metric is to be")

    # set the x-ticks to be the ranks
    for ax in g.axes.flat:
        ax.set_xticks(ks)
        ax.grid(True, which='both', alpha=0.4)

    # set custom titles
    g.set_titles(col_template="{col_name}")
    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(custom_titles[title], color = "black")

    # adding title and x-axis label
    cols = int(np.ceil(len(valid_metrics)/2))
    pos_dict = {1: [0.3, 0.78, 0.16, 0.92],
                2: [0.16, 0.88, 0.09, 0.96],
                3: [0.11, 0.9, 0.06, 0.96],
                4: [0.08, 0.93, 0.05, 0.97],
                5: [0.07, 0.95, 0.04, 0.98]}
    
    plt.subplots_adjust(bottom=pos_dict[cols][0], top=pos_dict[cols][1])
    g.figure.text(0.45, pos_dict[cols][2], "Factorization rank k", ha="center", va="center", fontsize = 14)
    g.figure.text(0.45, pos_dict[cols][3], "Optimal Factorization Rank", ha="center", va="center", fontsize = 16, fontweight="bold")
    
    legend_elements=[Line2D([0], [0], color='firebrick', linestyle='--', label='Best factorization rank(s) computed by compute_OptK()')]
    g.figure.legend(handles=legend_elements, loc = "lower center", bbox_to_anchor=(0.49, 0))

    g.set_ylabels("")
    g.set_xlabels("")

    plt.show()
    if path_to_save != None:
        plt.savefig(path_to_save)
    plt.close()


def river(NMFobj, path_to_save = None, ranks = None, useH = False, edges_cutoff = 0, sig_labels = True, sig_annot = None, sig_annot_recovery = False, sample_annot = None):
    """
    Creates a river plot with nodes presenting the signatures and the edges between signatures of different ranks
    depict the stability of the signature throughout the ranks.

    Parameters
    ----------
    NMFobj
        object from the multipleNMFobject class
    path_to_save
        string, path to a directory, including the name of the file to save  
    ranks
        list of at least two integers
        None, the optimal ranks are used, if there exist more than one
    useH
        boolean, if False W matrix is used, if True H matrix is used
    edges_cutoff
        float, the edge between nodes is only presented if the edge value is above the edges_cutoff parameter
    sig_labels
        boolean, if True the signatures labels will be shown at the nodes
    sig_annot
        list with length of maximal rank, these annotations are used for a legend giving the medical values the signatures at the end of the river plot (highest rank presented) are relevant for
    sig_annot_recovery
        boolean, if True the relevant medical annotations for the legend of the highest rank in the river plot are taken from the recovery plot
    sample_annot
        if recovery plot is used, these will be the sample annotations the relevance of the signatures is checked for 
    """
    
    ######################################################################################################
    #                                   RETRIEVE AND PREPARE DATA                                        #
    ######################################################################################################

    # Retrieve list of matrices
    if useH:
        W_list = [NMFobj.HMatrix[i].transpose() for i in range(len(NMFobj.HMatrix))]
    else:
        W_list = NMFobj.WMatrix

    # Get W matrices corresponding to ranks
    if ranks==None:
        # Show the best ranks if no ranks were given
        NMFobj.compute_OptKStats_NMF()
        NMFobj.compute_OptK()
        ranks = NMFobj.OptK
        # If less than 2 ranks are optimal show error
        if len(ranks) < 2:
            raise ValueError(f"There are to few optimal ranks to show a river plot. Provide a list of at least two ranks out of {NMFobj.ranks}.")
    else:
        index = []
        if type(ranks) != list:
            raise TypeError(f"The ranks parameters must be a list not a {type(ranks)}")
        if len(ranks) < 2:
            raise ValueError(f"Provide a list of at least two ranks out of {NMFobj.ranks}.")
        
        # If ranks are specified (as a list) search the corresponding indices of the runs
        for i, settings in enumerate(NMFobj.NMF_run_settings):
            if settings["rank"] in ranks:
                index.append(i)
        W_list = [W_list[i] for i in index]

    # Combine W (or H) matrices horizontally
    W_combined = np.hstack(W_list)  

    ######################################################################################################
    #                                        CREATE NODE COLORS                                          #
    ######################################################################################################

    pca = PCA(n_components=3)
    pca_transformed = pca.fit_transform(W_combined.T) 
    # Normalize PCA results between 0 and 1
    pca_normalized = np.apply_along_axis(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), 0, pca_transformed)
    
    # Creating list of colors based on PCA
    colors = [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.9)' for r, g, b in pca_normalized]
    
    ######################################################################################################
    #                                  CREATE VALUES FOR RIVER PLOT                                      #
    ######################################################################################################

    # Node names and index allocation
    sig_ids = [f"Sig{r}_{i+1}" for r in ranks for i in range(r)]
    sig_id_to_index = {sig_id: index for index, sig_id in enumerate(sig_ids)}

    # Color allocation to the nodes
    color_dict = dict(zip(sig_ids, colors))

    # Get source, target and value for the Sankey diagram
    edges_source = []
    edges_target = []
    edges_value = []

    for i in range(len(W_list)-1):
        W_k = W_list[i]
        W_kplus = W_list[i+1]

        # Non negative least square for width of edge between nodes
        nnls_matrix = nnls_sol(W_kplus, W_k)

        for m in range(nnls_matrix.shape[0]):
            for n in range(nnls_matrix.shape[1]):
                if nnls_matrix[m, n] > edges_cutoff:
                    edges_source.append(f"Sig{W_k.shape[1]}_{n+1}")
                    edges_target.append(f"Sig{W_kplus.shape[1]}_{m+1}")
                    edges_value.append(nnls_matrix[m, n])

    # Get edge colors, the mean color between the two nodes the edge is connecting
    edge_color = []
    for i in range(len(edges_source)):
        source_color = color_dict[edges_source[i]]
        target_color = color_dict[edges_target[i]]
        edge_color.append(pc.find_intermediate_color(rgba_to_rgb_tuple(source_color), rgba_to_rgb_tuple(target_color), 0.5))
   
    edge_color = [rgb_to_rgba_string(col) for col in edge_color]

    unique_targets = np.unique(edges_target)
    normalized_values = []
    # Normalize all values
    for target in unique_targets:
        mask = [edges_target[i] == target for i in range(len(edges_target))]
        edges_to_target = [edges_value[i] for i in range(len(edges_target)) if mask[i]]
        total_value = np.sum(edges_to_target)
        normalized_values.append(edges_to_target/total_value)

    edges_value_normalized = np.concatenate(normalized_values)
    
    edges_source = [sig_id_to_index[edge] for edge in edges_source]
    edges_target = [sig_id_to_index[edge] for edge in edges_target]

    # Prepare labels for nodes in river plot
    if sig_labels:
        # the whole labels overlap with too many ranks, so they are shorted to only provide the signature number as the rank is given by the x-axis label
        if len(ranks) > 4:
            label = [f"{i+1}" for r in ranks for i in range(r)]
        else:
            label = sig_ids
    else:
        label = ['']

    ######################################################################################################
    #                                        CREATE RIVER PLOT                                           #
    ######################################################################################################

    # Create Sankey Diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=8,
            line=dict(color="black", width=0),
            label=label,
            color=colors
        ),
        link=dict(
            source=edges_source,
            target=edges_target,
            value=edges_value_normalized,
            color=edge_color
        )
    ))

    ######################################################################################################
    #                                         LAST RANK LEGEND                                           #
    ######################################################################################################

    # Get the last ranks labels and indices of the nodes
    last_column_labels = [label for label in sig_ids if f"Sig{max(ranks)}" in label] 
    last_column_indices = [i for i, label in enumerate(sig_ids) if f"Sig{max(ranks)}" in label]
   
    # Add labels for the nodes representing the last rank
    max_rank = max(ranks)

    if sig_annot == None and sig_annot_recovery == False:
        # In this case no labeling for the last rank's nodes is shown
        last_node_labels = ['']*max_rank
    elif sig_annot != None:
        # In this case user defined labels for the last rank's nodes is shown
        if sig_annot_recovery == True:
            warnings.warn("You have provided your own signature annotation and set sig_annot_recovery to True. The annotations you provided will be used, if you want to see the annotations infered from the recovery plot set sig_annot to None and sig_annot_recovery to True.")
        if type(sig_annot)==list and len(sig_annot) == max_rank:
            last_node_labels = sig_annot
        else:
            warnings.warn("Your provided signature annotation is not of the correct form. It must be a list with its length being equal to the sum of the ranks.")
            last_node_labels = ['']*max_rank
    else:
        # In this case annotations from the recovery plot are used as labels for the last rank's nodes
        annot_df = {}
        if sample_annot == None:
            sample_annot = NMFobj.input_matrix["samples"]
        annot_groups = list(set(sample_annot))
        auc_dfs = recovery_plot(NMFobj, max_rank, sample_annot, path_to_save="recovery_for_Wannots")
        for i in range(len(auc_dfs[0])):
            # Get the p values in one row for each signature
            pvals = [df["p"].iloc[i] for df in auc_dfs]
            annots = []
            for e, p in enumerate(pvals):
                # Add the sample annotation to the signatures annotation if the p value is small
                if p < 0.05:
                    annots.append(annot_groups[e])
            annot_df[f"Sig{max_rank}_{i+1}"] = annots
        last_node_labels = [annot_df[key] for key in list(annot_df.keys())]    
        # Start next annotation label in next line
        last_node_labels = ["<br>".join(ticks) for ticks in last_node_labels]
        # Add the node label (form Sig(rank)_(sig))
        last_node_labels = [last_column_labels[i]+": "+last_node_labels[i] for i in range(len(last_node_labels))]
        last_node_labels.reverse()
    
    # Define the y positions for the labels
    y_pos = list(np.linspace(0, 1, num=len(last_node_labels)))
    # Save the colors of the last nodes to color the labels in the same way
    colors = list(fig.data[0]['node']['color'][min(last_column_indices):(max(last_column_indices)+1)])
    colors.reverse()
    
    annotations = []
    for i in range(len(last_column_indices)):
        annotations.append(dict(
            x=1.05,
            y=y_pos[i],
            xref='paper',
            yref='paper',
            text=last_node_labels[i],
            showarrow=False,
            font=dict(size=10, color=colors[i]),
            xanchor='left'
        ))
    
    # Create x axis labels
    rank_positions = list(np.linspace(0, 1, num=len(ranks)))
    rank_labels = [f"K{r}" for r in ranks]

    for i, rank in enumerate(rank_labels):
        annotations.append(
            dict(
                x=rank_positions[i],
                y=-0.1,  # Position below the plot
                xref="paper",
                yref="paper",
                text=rank,
                showarrow=False,
                font=dict(size=14),
                xanchor='center'
            )
        )

    # Add x-axis title as an annotation
    annotations.append(
        dict(
            x=0.5,  
            y=-0.2,  
            text="Factorization Rank", 
            showarrow=False,
            font=dict(size=20) 
        )
    )


    longest_label_length = max(len(line) for label in last_node_labels for line in label.split('<br>'))
    # Update layout with annotations
    fig.update_layout(
        title="Riverplot of Signature Stability",
        font_size=16,
        annotations=annotations,
        margin=dict(l=50, r=50+(6*longest_label_length), t=50, b=80),
        width = 150+len(ranks)*150
    )
    
    if path_to_save != None:
        # Save the figure
        pio.write_image(fig, f"{path_to_save}.png")


def rgba_to_rgb_tuple(rgba_str):
    # Use regex to extract the numbers
    match = re.search(r'rgba?\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', rgba_str)
    if match:
        return tuple(map(int, match.groups()[:3]))
    else:
        raise ValueError(f"Invalid rgba color string: {rgba_str}")

def rgb_to_rgba_string(rgb_tuple, alpha=0.9):
    # Ensure the RGB tuple contains exactly three elements
    if len(rgb_tuple) != 3:
        raise ValueError("RGB tuple must have exactly three elements.")
    
    # Format the RGB values with the alpha value
    return f'rgba({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]}, {alpha})'


def nnls_sol(B, A):
    X = np.zeros((B.shape[1], A.shape[1]))
    for i in range(B.shape[1]):
        X[i, :] = nnls(A, B[:, i])[0]

    return X
