import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import src.butcherPy.multiplerun_NMF_class as multiNMF
import matplotlib
from matplotlib.colors import ListedColormap
from scipy.cluster import hierarchy
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
import math
import matplotlib.patches as patches
from sklearn.metrics import auc
from matplotlib.lines import Line2D
import warnings

def H_heatmap_backup(NMFobj, ranks, sample_annot = None, path_to_save = None):
    """
    Creates a heatmap of H matrix and saves it at the given path.

    Parameters
    ----------
    NMFobj
        object from the multipleNMFobject class containing the H matrices
    ranks
        list of integers, the matrix resulting from the NMF run with this ranks will be shown in the heatmap
    sample_annot
        None, use the sample annotations that have been saved in the NMF object OR
        list, with the same length as the number of samples will be used as the sample annotations
    path_to_save
        string, path to a directory, including the name of the file to save
    
    Returns
    -------
    annotation_colors
        dictionary of tuples with three values corresponding to the RGB values for the colors used for the sample annotations, mainly to use same colors in recovery plot
    rank_colors
        dictionary of tuples with three values corresponding to the RGB values for the colors used for the signature annotations, mainly to use same colors in recovery plot
    """

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


    ########################################################################################################
    #                                         CREATE THE HEATMAP                                           # 
    ########################################################################################################

    # Plot the heatmap with reordered data
    fig, ax = plt.subplots(figsize=(22, 15))
    
    # Main heatmap with continuous colorbar on the right
    heatmap = sns.heatmap(data_ordered, cmap='viridis', ax=ax, cbar=True, cbar_kws={"shrink": 0.8, "location": "right", "anchor": (-0.5, 0)})
    plt.xticks([])
    yticks = ["Sig{}_{}".format(rank, i+1) for rank in ranks for i in range(rank)]
    plt.yticks(np.arange(data_ordered.shape[0]) + 0.5, yticks, rotation=0, fontsize=15)  # Adjust the yticks positions if needed
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
    
    # In case that only one rank is given, I want to create different colors for all signatures for better
    # recognition when creating the recovery plot
    if len(ranks)==1:

        palette2 = sns.color_palette("pastel")
        if len(rank_numbers) > len(palette2):
            palette2 += sns.color_palette("muted") 

        #sigs = ["Sig " + str(j+1) for j in range(ranks[0])]
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

    # Create the colormap according to the colors
    cmap_rank = ListedColormap(rank_mapped_colors)
    cmap_rank = plt.colorbar(ScalarMappable(cmap=cmap_rank), ax=ax, orientation='vertical', location='right', anchor = (0, 0), shrink = 0.8, ticks = [i+1 for i in range(len(ranks))])
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
        cmap_rank.ax.plot([1.25, 1.25], [start, end], color='black', linewidth=1.5, transform=cmap_rank.ax.transAxes, clip_on=False)
        cmap_rank.ax.plot([1.05, 1.25], [end, end], color='black', linewidth=1.5, transform=cmap_rank.ax.transAxes, clip_on=False)
        cmap_rank.ax.plot([1.05, 1.25], [start, start], color='black', linewidth=1.5, transform=cmap_rank.ax.transAxes, clip_on=False)

        # Adjust tick label position
        cmap_rank.ax.text(1.45, mid, ticks.pop(), ha='left', va='center', transform=cmap_rank.ax.transAxes, fontsize=15)

    
    ########################################################################################################
    #                    CREATE COLORBAR ON TOP OF THE HEATMAP WITH SAMPLE ANNOTATIONS                     # 
    ########################################################################################################

    # Extract the column annotations
    if sample_annot is None:
        annotations = NMFobj.input_matrix["samples"]
    else:
        annotations = sample_annot
    
    # Reorder the annotations based on dendrogram leaves (same way as columns of data)
    annotations = [annotations[i] for i in idx]

    unique_annotations = np.unique(annotations)
    
    # Define initial color palette "Paired" in reverse seems most appealing combination with the heatmap colors
    palette = sns.color_palette("Paired")
    palette.reverse()
    # If there are more unique annotations than colors in the palette, extend the palette
    if len(unique_annotations) > len(palette):
        palette += sns.color_palette("Set2")

    # ERROR IF PALETTE IS NOT LONG ENOUGH
    # Define color for each sample annotation
    annotation_colors = {annot: color for annot, color in zip(unique_annotations, palette)}
    # List with colors for all annotations
    mapped_colors = [annotation_colors[ann] for ann in annotations]
    
    # Create the colormap according to the colors
    cmap_samples = ListedColormap(mapped_colors)
    cmap_samples = plt.colorbar(ScalarMappable(cmap=cmap_samples), ax=ax, orientation='horizontal', location='top')
    cmap_samples.set_ticks([])
    cmap_samples.outline.set_visible(False)
    
    ########################################################################################################
    #                                          PLOT DENDROGRAM                                             # 
    ########################################################################################################

    ax_dendro = fig.add_axes([0.1, 0.85, 0.8, 0.1])
    # Adjust the position to be on top of the colormap corresponding to the sample annotations
    bbox_colorbar = cmap_samples.ax.get_position()
    ax_dendro.set_position([bbox_colorbar.x0,  bbox_colorbar.y1*0.912, bbox_colorbar.width, 0.13])
    hierarchy.dendrogram(linkage, ax=ax_dendro, orientation='top', no_labels=True)#, link_color_func=lambda x: 'k') # removing the '#' would turn the lines of the dendrogram black

    ax_dendro.set_axis_off()

    # Create legend patches for annotations
    legend_patches = [mpatches.Patch(color=color, label=annotation) for annotation, color in annotation_colors.items()]
    # Add legend for annotations
    plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.35, 1.3), title='Sample Annotations', title_fontsize = 20, fontsize=15)
    
    ########################################################################################################
    #                                      FINAL PLOT WITH TITLE                                           # 
    ########################################################################################################

    plt.title('H Matrix', fontsize=40, weight = 'bold')
    if path_to_save != None:
        plt.savefig(path_to_save)
    plt.show()
    plt.close()

    return annotation_colors, rank_colors

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
    
    heatmap = sns.heatmap(data_ordered, cmap='viridis', ax=ax, cbar=True, cbar_kws={"shrink": param_dict['heatmap_shrink'][nr_annots], "location": "right", "anchor": (-0.14, param_dict['heatmap_anchor_y'][nr_annots])})
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
        legend = ax.legend(handles=legends[i], title=annot_names[i], loc='upper right', bbox_to_anchor=(1.77, 1.4-legend_len*0.3))
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



def recovery_plot(NMFobj, rank, sample_annot = None, path_to_save = None):
    """
    Creates a recovery plot that shows the separation and identification of underlying gene expression patterns (signatures)
    associated with specific sample annotations.

    Parameters
    ----------
    NMFobj
        object from the multipleNMFobject class containing the H matrices
    ranks
        integer, for this ranks the recovery plot for the H matrix will be visualized
    sample_annot
        None, use the sample annotations that have been saved in the NMF object OR
        list, with the same length as the number of samples will be used as the sample annotations
    path_to_save
        string, path to a directory, including the name of the file to save   
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
    fig.text(0.465, 0.04, 'Rank', ha='center', va='center', fontsize=18)
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