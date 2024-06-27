import rds2py

def setup_rds(path):
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

def rds_to_ann():
    print("RData object into anndata object")
