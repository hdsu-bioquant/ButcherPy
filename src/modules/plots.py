import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap_H(matrix, path):    
    # TODO: Needs to be regularized before
    plt.figure(figsize=(15, 6))
    heatmap = sns.heatmap(matrix, cmap='viridis', annot=False, cbar=True)
    plt.title('H Matrix')
    plt.xlabel('Samples')
    plt.ylabel('Signatures')

    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_title('Exposure', pad=10)

    cbar.ax.text(1.1, 0, 'Low', ha='left', va='center', transform=cbar.ax.transAxes)
    cbar.ax.text(1.1, 1, 'High', ha='left', va='center', transform=cbar.ax.transAxes)


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