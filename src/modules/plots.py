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

def H_heatmap(NMFobj, ranks, path_to_save):
    """
    Creates a heatmap of H matrix and saves it at the given path.

    Parameters
    ----------
    NMFobj
        object from the multipleNMFobject class containing the H matrices
    ranks
        list of integers, the matrix resulting from the NMF run with this ranks will be shown in the heatmap
    path_to_save
        string, path to a directory, including the name of the file to save and the correct file extension, e.g.png
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
    fig, ax = plt.subplots(figsize=(22, 12))
    
    # Main heatmap with continuous colorbar on the right
    heatmap = sns.heatmap(data_ordered, cmap='viridis', ax=ax, cbar=True, cbar_kws={"shrink": 0.8, "location": "right", "anchor": (-0.5, 0)})
    plt.xticks([])
    yticks = ["Sig{}_{}".format(rank, i+1) for rank in ranks for i in range(rank)]
    plt.yticks(np.arange(data_ordered.shape[0]) + 0.5, yticks, rotation=0, fontsize=15)  # Adjust the yticks positions if needed
    plt.xlabel('Samples', fontsize=23)

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

    rank_labels = ["Rank "+str(rank) for rank in ranks for _ in range(rank)]

    # Palette with muted colors as this is not the thing that is supposed to stand out
    palette2 = sns.color_palette("pastel")
    # If there are more ranks than colors in the palette, extend the palette
    if len(np.unique(rank_labels)) > len(palette2):
           palette2 += sns.color_palette("muted") 
    
    
    # Define color for each rank annotation
    rank_colors = {rank: color for rank, color in zip(np.unique(rank_labels), palette2)}
    # List with colors for all ranks
    rank_mapped_colors = [rank_colors[rank] for rank in rank_labels]
    rank_mapped_colors.reverse()
    # Create the colormap according to the colors
    cmap_rank = ListedColormap(rank_mapped_colors)
    cmap_rank = plt.colorbar(ScalarMappable(cmap=cmap_rank), ax=ax, orientation='vertical', location='right', anchor = (0, 0), shrink = 0.8, ticks = [i+1 for i in range(len(ranks))])
    cmap_rank.outline.set_visible(False)

    # Calculate tick positions, so that the label is in the middle of the corresponding color
    cbartop = 1
    sumrank = np.sum(ranks)
    pos = []
    for rank in ranks:
        procent = rank/sumrank
        pos.append(cbartop-(procent/2))
        cbartop -= procent

    pos.reverse()
    cmap_rank.set_ticks(pos)
    ticks = np.unique(rank_labels).tolist()
    ticks.reverse()
    cmap_rank.set_ticklabels(ticks, fontsize=15)


    ########################################################################################################
    #                    CREATE COLORBAR ON TOP OF THE HEATMAP WITH SAMPLE ANNOTATIONS                     # 
    ########################################################################################################

    # Extract the column annotations
    annotations = NMFobj.input_matrix["samples"]
    # Reorder the annotations based on dendrogram leaves (same way as columns of data)
    annotations = [annotations[i] for i in idx]

    unique_annotations = np.unique(annotations)

    # Define initial color palette "Paired" in reverse seems most appealing combination with the heatmap colors
    palette = sns.color_palette("Paired")
    palette.reverse()
    # If there are more unique annotations than colors in the palette, extend the palette
    if len(unique_annotations) > len(palette):
           palette += sns.color_palette("Set2")

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
    ax_dendro.set_position([bbox_colorbar.x0,  bbox_colorbar.y1*0.922, bbox_colorbar.width, 0.13])
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
    plt.savefig(path_to_save)
    plt.show()
    plt.close()


def heatmap_H(NMFobj, path, ranks, matH = True):
    """
    Creates a heatmap of either the H or the W matrix and saves it at the given path.

    Parameters
    ----------
    NMFobj
        object from the multipleNMFobject class containing the W and H matrices
    path
        path to a directory, including the name of the file to save, as a string
    ranks
        list of integers, the matrix resulting from the NMF run with this ranks will be shown in the heatmap
    matH
        a boolean, if True the H matrix is used, if False the W matrix is used
    """
    if matH:
        NMFobj.regularize_H(ranks)
        regH = NMFobj.get_H(ranks)

        if len(ranks)>1:
            matrices = []
            for i in range(len(ranks)):
                matrices.append(regH[i])
            regH = [np.vstack(matrices)]

        plt.figure(figsize=(15, 6))
        heatmap = sns.heatmap(regH[0], cmap='viridis', annot=False, cbar=True)
        plt.title('H Matrix', fontsize=20)
        plt.xlabel('Samples')
        plt.ylabel('Signatures')

        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_title('Exposure', pad=10)

        cbar.set_ticks([])

        cbar.ax.text(1.1, 0.02, 'Low', ha='left', va='center', transform=cbar.ax.transAxes)
        cbar.ax.text(1.1, 0.98, 'High', ha='left', va='center', transform=cbar.ax.transAxes)

        plt.xticks([])
        plt.yticks([])

        plt.show()
        plt.savefig(path)
        plt.close()
    else:
        NMFobj.regularize_W(ranks)
        regW = NMFobj.get_W(ranks)

        plt.figure(figsize=(6, 15))
        heatmap = sns.heatmap(regW[0], cmap='viridis', annot=False, cbar=True)
        plt.title('W Matrix', fontsize=20)
        plt.xlabel('Signatures')
        plt.ylabel('Genes')

        cbar = heatmap.collections[0].colorbar

        plt.xticks([])
        plt.yticks([])

        plt.show()
        plt.savefig(path)
        plt.close()


def recovery_plot(matrix, annot):
    # Ensure compatibility with numpy array and transform it to a Dataframe
    if isinstance(matrix, np.ndarray):
        matrix = pd.DataFrame(matrix)
    
    h = matrix.copy()

    # Add signature IDs if missing
    if h.index.isnull().all():
        h.index = [f'Sig {i+1}' for i in range(h.shape[0])]

    # Check if annot length matches the number of columns in the matrix
    if isinstance(annot, (pd.Series, pd.Categorical, np.ndarray, list)) and len(annot) == h.shape[1]:
        # TODO: not sure if the categories = h.columns makes sense, let's see
        annot_factor = pd.Categorical(annot)#, categories = h.columns)
    else:
        raise ValueError("Not a vaild annotation input")
    
    n_samples = h.shape[1]
    annot_series = pd.Series(annot)

    print("series: ", annot_series)
    print("factor: ", annot_factor)

    ## -------------------------------------------------------------------##
    ##                        Find ranks                                  ##
    ##--------------------------------------------------------------------##

    lIds = annot_factor.categories

    all_ranks = []
    
    for l in lIds:
        level_ranks = []
        for i in range(h.shape[0]):
            exp=h.iloc[i].sort_values(ascending=False)
            i_rank = exp.index[annot_series == l].tolist()
            level_ranks.append(sorted(i_rank))
        
        all_ranks.append(level_ranks)

    print(all_ranks)
    auc_singleannot = []
    for r in all_ranks:
        auc_rand = []
        for x in r:
            t = 1
            l = [resample(range(n_samples), n_samples=len(x), replace=False) for _ in range(500)]
            aux = auc(l, max_val=n_samples)
            auc_rand.append([np.mean(aux), np.std(aux)])
        auc_rand = np.array(auc_rand)

        #print(auc_rand)
        
        auc_vals = auc(r, max_val=n_samples)

        auc_df = pd.DataFrame(auc_rand, columns=["mean", "sd"])
        auc_df["val"] = auc_vals
        auc_df["z"] = (auc_df["val"]-auc_df["mean"]/auc_df["sd"])
        auc_df["p"] = auc_df["z"].apply(lambda z: norm.sf(z) if z > 0 else norm.cdf(z))
        print(auc_df)
        auc_singleannot.append(auc_df)
    
    auc_allannot = pd.concat(auc_singleannot, keys=lIds, names=["Annotation_level", "SignatureID"]).reset_index()

    all_ranks = [[np.concatenate(([0], xi, [n_samples])) for xi in x] for x in all_ranks]
    all_ranks_df = pd.concat(
        [pd.DataFrame({"Annotation_level": l, "SignatureID": i, "Rank": xi})
        for l, ranks in zip(lIds, all_ranks) for i, xi in enumerate(ranks)],
        ignore_index=True
        )

    #pd.melt(all_ranks_df, id_vars=["Annotation_level"], value_vars=["SignatureID"])
    all_ranks_df = pd.melt(all_ranks_df, id_vars=["Annotation_level", "SignatureID"], value_vars=["Rank"], value_name="rank")
    all_ranks_df = all_ranks_df.drop('variable', axis=1)
    all_ranks_df = all_ranks_df.merge(auc_allannot, on = ["Annotation_level", "SignatureID"], how="left")

    print(all_ranks_df)
    all_ranks_df = all_ranks_df.sort_values(by=['Annotation_level', 'SignatureID', 'rank'])
    all_ranks_df['Frequency'] = all_ranks_df.groupby(['Annotation_level', 'SignatureID']).cumcount() / all_ranks_df.groupby(['Annotation_level', 'SignatureID'])['rank'].transform('count')
    all_ranks_df['Frequency'] = all_ranks_df['Frequency'].replace(0, np.nan).ffill().fillna(0)
    all_ranks_df['issignif'] = all_ranks_df['p'] < 0.05

    ## Plotting
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    g = sns.FacetGrid(all_ranks_df, col="Annotation_level", col_wrap=4, sharex=False, sharey=False)

    def plot_func(data, color):
        sns.lineplot(data=data, x="rank", y="Frequency", hue="SignatureID", style="issignif", size="issignif", sizes=(0.5, 1.5), dashes=[(2, 2), (1, 0)], palette="tab10", legend=False)

    g.map_dataframe(plot_func)
    for ax in g.axes.flat:
        ax.plot([0, n_samples], [0, 1], 'k--', lw=0.5)

    g.add_legend()
    g.set_axis_labels("rank", "Frequency")

    plt.show()
    plt.savefig("recovery.png")
    plt.close()

    #return all_ranks_df



def auc(rank_list, max_val=None):
    auc = []
    for r in rank_list:
        if max_val is None:
            max_val = max(r)

        rank_sorted = sorted(r)
        X = 0
        i = 0
        #ngenes = len(r)

        while i < len(rank_sorted) and rank_sorted[i] <= max_val:
            X += max_val - rank_sorted[i]
            i += 1

        rauc = X/i/max_val if i > 0 else 0
        auc.append(rauc)
    
    return auc