import rds2py
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
import warnings

def setup_rds(path):
    """
    Trys to convert a type of matrix or experiment stored in a RData file to a numpy array.
    
    Parameters
    ----------
    path
        string, a path to a RData file containing a matrix or experiment
    
    Returns
    -------
        numpy matrix extracted from the RData file
    """

    if not path.endswith('.rds'):
        # Raise a ValueError if the file doesn't end with .rds
        raise ValueError(f"Invalid file extension for {path}. Expected a .rds file.")
    
    r_obj = rds2py.read_rds(path)
    mat = None

    try:
        mat = rds2py.as_sparse_matrix(r_obj)
        print("The rds file stores a sparse matrix: dgCMatrix, dgRMatrix or dgTMatrix")
    except:
        pass

    try:
        mat = rds2py.as_dense_matrix(r_obj)
        print("The rds file stores a dense matrix")
    except:
        pass

    try:
        mat = rds2py.as_summarized_experiment(r_obj)
        print("The rds file strose a SingleCellExperiment or SummarizedExperiment")
    except:
        pass

    if mat is None:
        print("The setup of the gene expression matrix failed. The rds file seems not to save valid options, like sparse or dense matrics or experiments.")

    return mat

def rds_to_ann(path_to_rdsmatrix, path_to_rdsannot = None, path_to_rdsmeta = None, gene_index = 0, sample_index = 0, extra_metas = False, gene_annot = None, sample_annot = None, gene_columns = None):
    """
    Transforms RData to AnnData.

    Parameters
    ----------
    path_to_rdsmatrix
        string, the path to a RData file containing a gene expression matrix
    path_to_rdsannot
        string, the path to a RData file containing the gene annotations, the gene_annot gets priority if both are provided
    path_to_rdsmeta
        string, the path to a RData file containing the sample annotations, the sample_annot gets priority if both are provided
    gene_index
        integer, the rds data are read as a list of different attributes, the gene index defines which list element will be taken as gene annotation
    gene_index
        integer, the rds data are read as a list of different attributes, the sample index defines which list element will be taken as sample annotation
    extra_metas
        boolean, defining if further meta data stored in the path_to_rdsmeta file should be stored in the AnnData object
    gene_annot
        list of gene annotations 
    sample_annot
        list of sample annotations
    gene_columns
        a boolean stating if the gene expression matrix stores the genes in the columns (True) or if it stores the genes in the rows (False)
    
    Returns
    -------
    adata
        AnnData object containing the provided information
    """

    if not path_to_rdsmatrix.endswith('.rds'):
        # Raise a ValueError if the file doesn't end with .rds
        raise ValueError(f"Invalid file extension for {path_to_rdsmatrix}. Expected a .rds file.")
    
    #----------------------------------------------------------------------------------------------------------------#
    #                            Get the gene expression matrix for AnnData, saved in X                              #
    #----------------------------------------------------------------------------------------------------------------#
    
    # Get the gene expression matrix from the RData file
    npmat = setup_rds(path_to_rdsmatrix)

    # In AnnData the genes are stored in the columns, if that is not the case for the data at hand, the matrix is transposed
    if gene_columns == False:
        npmat = np.transpose(npmat)

    # Check if the matrix has same column and row dimension
    if npmat.shape[0] == npmat.shape[1] and gene_columns == None:
        raise warnings.warn("The column and row dimension are equal, please make sure that the gene_columns parameter indicates, if the genes are saved in the columns (True) or not (False).")
    
    # counts = csr_matrix(npmat, dtype=np.float32)
    adata = ad.AnnData(npmat)


    #----------------------------------------------------------------------------------------------------------------#
    #                Get the variable names (gene annotations) for AnnData saved in the columns                      #
    #----------------------------------------------------------------------------------------------------------------#

    if gene_annot != None:
        gene_annots = gene_annot
    
    elif path_to_rdsannot != None:
        if not path_to_rdsannot.endswith('.rds'):
            # Raise a ValueError if the file doesn't end with .rds
            raise ValueError(f"Invalid file extension for {path_to_rdsannot}. Expected a .rds file.")
    
        annot = rds2py.read_rds(path_to_rdsannot)
        print()
        if gene_index < 0 or gene_index >= len(annot['attributes']['names']['data']):
            gene_index = 0
            warnings.warn("The provided gene_index is out of range, by default it is set to 0.")

        # Gene annotations
        gene_annots = annot['data'][gene_index]['data']
        print(f"Your gene annotation file stores {', '.join(annot['attributes']['names']['data'])}. Due to the provided gene_index parameter of {gene_index}, {annot['attributes']['names']['data'][gene_index]} is used for the annotation. If this is not the desired gene annotation, you can change the index parameter, with these options:")
        print(', '.join([f"{i} (corresponding to {name})" for i, name in [tuple((i, name)) for i, name in enumerate(annot['attributes']['names']['data'])]]))

    else:
        warnings.warn("You have not provided an annotation for the genes, by default the columns are numerated.")
        gene_annots = [f"Gene_{(i+1):d}" for i in range(adata.n_vars)]

    if type(gene_annots) == list and len(gene_annots) != adata.n_vars:
        warnings.warn("The provided gene annotation does not have the same dimension as the number of columns in the gene expression matrix, by default the columns are numerated. Check your data, especially if the matrix must be transposed and indicate it by the gene_columns parameter.")
        gene_annots = [f"Gene_{(i+1):d}" for i in range(adata.n_vars)]
    
    adata.var_names = gene_annots


    #----------------------------------------------------------------------------------------------------------------#
    #               Get the observation names (sample annotations) for AnnData saved in the rows                     #
    #----------------------------------------------------------------------------------------------------------------#

    if sample_annot != None:
        sample_annots = sample_annot

    elif path_to_rdsmeta != None:
        if not path_to_rdsmeta.endswith('.rds'):
            # Raise a ValueError if the file doesn't end with .rds
            raise ValueError(f"Invalid file extension for {path_to_rdsmeta}. Expected a .rds file.")
        
        meta = rds2py.read_rds(path_to_rdsmeta)
        print()
        if sample_index < 0 or sample_index >= len(meta['attributes']['names']['data']):
            sample_index = 0
            warnings.warn("The provided sample_index is out of range, by default it is set to 0.")

        # Sample annotations
        sample_annots = meta['data'][sample_index]['data']
        print(f"Your sample annotation file stores {', '.join(meta['attributes']['names']['data'])}. Due to the provided sample_index parameter of {sample_index}, {meta['attributes']['names']['data'][sample_index]} is used for the annotation. If this is not the desired sample annotation, you can change the index parameter, with these options:")
        print(', '.join([f"{i} (corresponding to {name})" for i, name in [tuple((i, name)) for i, name in enumerate(meta['attributes']['names']['data'])]]))

    else:
        warnings.warn("You have not provided an annotation for the samples, by default the rows are numerated.")
        sample_annots = [f"Sample_{(i+1):d}" for i in range(adata.n_obs)]

    if type(sample_annots) == list and len(sample_annots) != adata.n_obs:
        warnings.warn("The provided sample annotation has not the same dimension as the number of rows in the gene expression matrix, by default the rows are numerated. Check your data, especially if the matrix must be transposed and indicate it by the gene_columns parameter.")
        sample_annots = [f"Sample_{(i+1):d}" for i in range(adata.n_obs)]
    else:
        # Only if the sample annotations given in the RData file match the row dimension of the matrix the meta data can be used successfully
        if extra_metas and path_to_rdsmeta != None:# and sample_annot == None:
            meta = rds2py.read_rds(path_to_rdsmeta)
            metas = meta['attributes']['names']['data']
            for i, m in enumerate(metas):
                adata.obs[m] = meta['data'][i]['data'] 

    adata.obs_names = sample_annots

    return adata