import unittest
import numpy as np
import pandas as pd
import anndata as ad
import warnings
import sys
# If running from inside ButcherPy
sys.path.append("../ButcherPy")
# If running from inside tests
sys.path.append("../../ButcherPy")
from src.butcherPy.nmf_run import run_NMF, multiple_rank_NMF
from src.butcherPy.multiplerun_NMF_class import multipleNMFobject
from src.modules.utils import rds_to_ann

class TestNMF(unittest.TestCase):

    def setUp(self):
        # create simple test matrix in all three datatypes that can be used
        #self.matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.matrix = np.random.rand(10, 4)
        cols = [f"Sample_{i+1}" for i in range(self.matrix.shape[1])]
        rows = [f"Gene_{i+1}" for i in range(self.matrix.shape[0])]
        self.df = pd.DataFrame(self.matrix, columns=cols, index=rows)
        self.adata = ad.AnnData(self.matrix.T)
        self.adata.var_names = rows
        self.adata.obs_names = cols

        self.rank = 2
        self.n_initializations = 5
        self.iterations = 50
        self.seed = 42
        self.stop_threshold = 10
        self.nthreads = 1

    def test_run_NMF_ndarray(self):
        rank, H_num, W_num, W_eval_num, iter_to_conv, frobNorm, time_stamp = run_NMF(
            self.matrix,
            self.rank,
            self.n_initializations,
            self.iterations,
            self.seed,
            self.stop_threshold,
            self.nthreads
        )

        # Check shape of H and W matrices
        self.assertEqual(H_num.shape, (self.rank, self.matrix.shape[1]))
        self.assertEqual(W_num.shape, (self.matrix.shape[0], self.rank))

        # Check number of saved iterations to convergence and frobenius norms
        self.assertEqual(len(iter_to_conv), self.n_initializations)
        self.assertEqual(len(frobNorm), self.n_initializations)

        # Check for the correct number and shape of W matrices in W_eval_num
        self.assertEqual(len(W_eval_num), self.n_initializations)
        for W in W_eval_num:
            self.assertEqual(W.shape, (self.matrix.shape[0], self.rank))


    def test_multiple_rank_NMF(self):
        ranks = [1,2]
        inputs = [self.matrix, self.df, self.adata]
        for input in inputs:
            NMF = multiple_rank_NMF(
                input,
                ranks,
                self.n_initializations,
                self.iterations,
                self.seed,
                self.stop_threshold,
                self.nthreads
            )

            # Check that the type of the output is a multipleNMFObject
            self.assertIsInstance(NMF, multipleNMFobject)
            self._test_multipleNMFobject(NMF, ranks)

    def _test_multipleNMFobject(self, NMFobject, ranks):
        # Check that results were produced for each rank
        self.assertEqual(len(NMFobject.WMatrix), len(ranks))
        self.assertEqual(len(NMFobject.HMatrix), len(ranks))

        # Verify shapes of matrices for each rank
        for i, rank in enumerate(ranks):
            self.assertEqual(NMFobject.WMatrix[i].shape[1], rank)
            self.assertEqual(NMFobject.HMatrix[i].shape[0], rank)

        # Check frobenius norm and iteration counts
        for frob_error in NMFobject.frobenius:
            self.assertEqual(len(frob_error), self.n_initializations)
        for i, iters in enumerate(NMFobject.NMF_run_settings):
            self.assertEqual(iters['rank'], ranks[i])

    def test_edge_cases(self):
        with self.assertWarns(UserWarning) as cm:
            NMF =  multiple_rank_NMF(
                self.matrix,
                [5],
                self.n_initializations,
                self.iterations,
                self.seed,
                self.stop_threshold,
                self.nthreads
            )
        # Check that the type of the output is a multipleNMFObject
        self.assertIsInstance(NMF, multipleNMFobject)
        self._test_multipleNMFobject(NMF, [5])
        self.assertEqual(str(cm.warning), "Be aware that the number of columns/samples is lower than the indicated ranks.")

        with self.assertWarns(UserWarning) as cm:
            NMF = multiple_rank_NMF(
                self.matrix,
                [12],
                self.n_initializations,
                self.iterations,
                self.seed,
                self.stop_threshold,
                self.nthreads
            )
        # Check that the type of the output is a multipleNMFObject
        self.assertIsInstance(NMF, multipleNMFobject)
        self._test_multipleNMFobject(NMF, [12])
        self.assertEqual(str(cm.warning), "Be aware that the number of columns/samples and rows/genes are lower than the indicated ranks.")

        oned_array = np.random.rand(4, 1)
        with self.assertWarns(UserWarning) as cm:
            NMF = multiple_rank_NMF(
                    oned_array,
                    [1],
                    self.n_initializations,
                    self.iterations,
                    self.seed,
                    self.stop_threshold,
                    self.nthreads
                )
        # Check that the type of the output is a multipleNMFObject
        self.assertIsInstance(NMF, multipleNMFobject)
        self._test_multipleNMFobject(NMF, [1])
        self.assertEqual(str(cm.warning), "Your input has only 1 entry in either of the dimensions. Be aware that the NMF algorithm might not be able to catch any patterns in the vector. Check if your matrix input has the shape you expected or if the NMF algorithm makes sense to use for your problem.")
        
        # CREATE ERROR FOR THE CASE THAT SOMETHING HAS THE WRONG DATATYPE; FOR EXAMPLE RANKS ONLY BEING AN INTEGER
        rank = 3
        with self.assertRaises(TypeError):
            multiple_rank_NMF(
                    self.matrix,
                    rank,
                    self.n_initializations,
                    self.iterations,
                    self.seed,
                    self.stop_threshold,
                    self.nthreads
                )


    def test_leukemiadata(self):
        # WRITE GOOD ERRORS: the rdata reading, should give reasonable error if it is not a rdata file
        path_to_rdata = "../data/GSE_bpd/GSE53987_2.rds"
        path_to_genes = "../data/GSE_bpd/GSE53987_2_annots.rds"
        path_to_samples = "../data/GSE_bpd/GSE53987_2_metadata.rds"

        gene_idx = 1
        sample_idx = 1

        adata = rds_to_ann(path_to_rdata, path_to_rdsannot=path_to_genes, path_to_rdsmeta=path_to_samples, gene_index=gene_idx, sample_index=sample_idx, extra_metas=True, gene_columns=False)
        NMF = multiple_rank_NMF(adata,
                            [2, 3, 4, 5, 6, 7], 
                            20, 
                            300, 
                            self.seed, 
                            self.stop_threshold, 
                            self.nthreads)

        self.assertIsInstance(NMF, multipleNMFobject)

        rec_matrices = []
        for i in range(len(NMF.ranks)):
            rec_matrices.append(NMF.WMatrix[i]@NMF.HMatrix[i])
        
        mean_errors = [np.mean(np.abs(NMF.input_matrix["gene_expression"]-rec_matrices[i])) for i in range(len(rec_matrices))]
        frob_errors = [np.min(NMF.frobenius[i]) for i in range(len(rec_matrices))]

        print(mean_errors)
        print(frob_errors)

        self.assertTrue(any(np.mean(error) < 0.2 for error in mean_errors), "For no rank the reconstructed matrix is close enough to the original matrix")
        self.assertTrue(all(error < 0.1 for error in frob_errors), "For no rank the reconstructed matrix has a small enough frobenius error")
    
        

    #def test_multiRankNMF_different_inputs(self):
    #    inputs = [self.matrix, self.df, self.adata]
    #    for input in inputs:
    #        with self.subTest(input=input):
    #            self.test_multiRankNMF_different_inputs(input)


if __name__ == '__main__':
    unittest.main()