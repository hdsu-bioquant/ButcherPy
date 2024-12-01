import unittest
import numpy as np
import pandas as pd
import anndata as ad
import sys
# If running from inside ButcherPy
sys.path.append("../ButcherPy")
# If running from inside tests
sys.path.append("../../ButcherPy")
from src.butcherPy.nmf_run import run_NMF, multiple_rank_NMF
from src.butcherPy.multiplerun_NMF_class import multipleNMFobject
from src.modules.utils import rds_to_ann
from unittest.mock import patch

# Assuming the NMFobject class is imported here

class TestNMFObject(unittest.TestCase):
   
    @classmethod
    def setUpClass(cls):
        # Create a realistic NMF object to test all NMF object functions only once
        path_to_rdata = "../data/GSE_bpd/GSE53987_2.rds"
        path_to_genes = "../data/GSE_bpd/GSE53987_2_annots.rds"
        path_to_samples = "../data/GSE_bpd/GSE53987_2_metadata.rds"

        gene_idx = 1
        sample_idx = 1

        adata = rds_to_ann(
            path_to_rdata, 
            path_to_rdsannot=path_to_genes, 
            path_to_rdsmeta=path_to_samples, 
            gene_index=gene_idx, 
            sample_index=sample_idx, 
            extra_metas=True, 
            gene_columns=False
        )
        
        cls.NMF = multiple_rank_NMF(adata, [2, 3, 4, 5, 6], 20, 300, 42, 40, 1)
        cls.rec_matrices = [cls.NMF.WMatrix[i] @ cls.NMF.HMatrix[i] for i in range(len(cls.NMF.ranks))]


    # Test retrieval of W matrix by rank
    def test_get_W(self):
        # Retrieve all W matrices
        all_W = self.NMF.get_W(ranks="all")
        self.assertEqual(len(all_W), len(self.NMF.ranks))
        
        # Retrieve W matrix for rank 2
        rank_2_W = self.NMF.get_W(ranks=[2])
        self.assertEqual(len(rank_2_W), 1)
        np.testing.assert_array_equal(rank_2_W[0], self.NMF.WMatrix[0])
        
        # Attempt to retrieve W matrix for non-existent rank
        with self.assertRaises(AssertionError):
            self.NMF.get_W(ranks=[99])

        # Attempt to retrieve H matrices for ranks, including a rank that does not exist
        with self.assertWarns(UserWarning) as warning:
            rank_2_W = self.NMF.get_W(ranks=[2, 99]) 
            self.assertEqual(len(rank_2_W), 1)
            np.testing.assert_array_equal(rank_2_W[0], self.NMF.WMatrix[0])

    # Test retrieval of H matrix by rank
    def test_get_H(self):
        # Retrieve all H matrices
        all_H = self.NMF.get_H(ranks="all")
        self.assertEqual(len(all_H), len(self.NMF.ranks))
        
        # Retrieve H matrix for rank 2
        rank_2_H = self.NMF.get_H(ranks=[2])
        self.assertEqual(len(rank_2_H), 1)
        np.testing.assert_array_equal(rank_2_H[0], self.NMF.HMatrix[0])
        
        # Attempt to retrieve H matrix for non-existent rank
        with self.assertRaises(AssertionError):
            self.NMF.get_H(ranks=[99])

        # Attempt to retrieve H matrices for ranks, including a rank that does not exist
        with self.assertWarns(UserWarning) as warning:
            rank_2_H = self.NMF.get_H(ranks=[2, 99]) 
            self.assertEqual(len(rank_2_H), 1)
            np.testing.assert_array_equal(rank_2_H[0], self.NMF.HMatrix[0])

    # Test normalization of W matrix
    def test_normalize_W(self):
        
        self.NMF.normalize_W(ranks=[2])
        normed_W = self.NMF.get_W(ranks=[2])[0]

        np.testing.assert_almost_equal(normed_W.sum(axis=0), 1, decimal=5)
        np.testing.assert_almost_equal(self.rec_matrices[0], self.NMF.WMatrix[0] @ self.NMF.HMatrix[0], decimal=5)

        # Check that normalization is not multiple times for one rank
        with patch('sys.stdout'):# as mocked_stdout:
            self.NMF.normalize_W(ranks=[2])  # Test for rank 2 (pre-normalized)

    # Test normalization of H matrix
    def test_normalize_H(self):
        self.NMF.normalize_H(ranks=[2])
        normed_H = self.NMF.get_H(ranks=[2])[0]
        
        np.testing.assert_almost_equal(normed_H.sum(axis=1), 1, decimal=5)
        np.testing.assert_almost_equal(self.rec_matrices[0], self.NMF.WMatrix[0] @ self.NMF.HMatrix[0], decimal=5)
        
         # Check that normalization is not multiple times for one rank
        with patch('sys.stdout'):# as mocked_stdout:
            self.NMF.normalize_H(ranks=[2])  # Test for rank 2 (pre-normalized)

    # Test regularization of W matrix
    def test_regularize_W(self):
        self.NMF.regularize_W(ranks=[2])
        regularized_W = self.NMF.get_W(ranks=[2])[0]

        self.assertTrue(np.all(regularized_W >= 0) and np.all(regularized_W <= 1))
        np.testing.assert_almost_equal(self.rec_matrices[0], self.NMF.WMatrix[0] @ self.NMF.HMatrix[0], decimal=5)

    # Test regularization of H matrix
    def test_regularize_H(self):
        self.NMF.regularize_H(ranks=[2])
        regularized_H = self.NMF.get_H(ranks=[2])[0]

        self.assertTrue(np.all(regularized_H >= 0) and np.all(regularized_H <= 1))
        np.testing.assert_almost_equal(self.rec_matrices[0], self.NMF.WMatrix[0] @ self.NMF.HMatrix[0], decimal=5)
      
    # Test normalization of W matrix with all ranks
    def test_normalize_W_all(self):
        self.NMF.normalize_W(ranks="all")
        for normed_W in self.NMF.get_W(ranks="all"):
            np.testing.assert_almost_equal(normed_W.sum(axis=0), 1, decimal=5)

    # Test normalization of H matrix with all ranks
    def test_normalize_H_all(self):
        self.NMF.normalize_H(ranks="all")
        for normed_H in self.NMF.get_H(ranks="all"):
            np.testing.assert_almost_equal(normed_H.sum(axis=1), 1, decimal=5)

    # Test regularization of W matrix with all ranks
    def test_regularize_W_all(self):
        self.NMF.regularize_W(ranks="all")
        for regularized_W in self.NMF.get_W(ranks="all"):
            self.assertTrue(np.all(regularized_W >= 0) and np.all(regularized_W <= 1))

    # Test regularization of H matrix with all ranks
    def test_regularize_H_all(self):
        self.NMF.regularize_H(ranks="all")
        for regularized_H in self.NMF.get_H(ranks="all"):
            self.assertTrue(np.all(regularized_H >= 0) and np.all(regularized_H <= 1))

    # Edge case for non-existent rank normalization
    def test_normalize_non_existent_rank(self):
        with self.assertRaises(AssertionError):
            self.NMF.normalize_W(ranks=[99])
        with self.assertRaises(AssertionError):
            self.NMF.normalize_H(ranks=[99])
        with self.assertRaises(AssertionError):
            self.NMF.regularize_W(ranks=[99])
        with self.assertRaises(AssertionError):
            self.NMF.regularize_H(ranks=[99])

    # Test for compute_OptKStats_NMF
    def test_compute_OptKStats_NMF(self):
        # Compute statistics for all ranks
        OptKStats_all = self.NMF.compute_OptKStats_NMF(ranks="all")
        self.assertEqual(len(OptKStats_all), len(self.NMF.ranks))
        for stats in OptKStats_all:
            self.assertIn("rank", stats)
            self.assertIn("FrobError_min", stats)
            self.assertIn("FrobError_mean", stats)
            self.assertIn("FrobError_sd", stats)
            self.assertIn("FrobError_cv", stats)
            self.assertIn("sumSilWidth", stats)
            self.assertIn("meanSilWidth", stats)
            self.assertIn("copheneticCoeff", stats)
            self.assertIn("meanAmariDist", stats)

        # Compute statistics for specific rank
        rank_2_stats = self.NMF.compute_OptKStats_NMF(ranks=[2])
        self.assertEqual(len(rank_2_stats), 1)
        self.assertEqual(rank_2_stats[0]["rank"], 2)
        
        # Validate Frobenius statistics are not None
        self.assertIsNotNone(rank_2_stats[0]["FrobError_min"])
        self.assertIsNotNone(rank_2_stats[0]["FrobError_mean"])

        # Attempt to compute statistics for an already computed rank
        initial_OptKStats_count = len(self.NMF.OptKStats)
        OptKStats_recomputed = self.NMF.compute_OptKStats_NMF(ranks=[2])
        self.assertEqual(len(self.NMF.OptKStats), initial_OptKStats_count)  # No new statistics should be added

        # Attempt to compute statistics for a non-existent rank
        with self.assertRaises(AssertionError):
            self.NMF.compute_OptKStats_NMF(ranks=[99])
            
        # Validate mixed case: one valid rank and one non-existent rank
        with self.assertWarns(UserWarning):
            partial_stats = self.NMF.compute_OptKStats_NMF(ranks=[2, 99])
            self.assertEqual(len(partial_stats), 1)
            self.assertEqual(partial_stats[0]["rank"], 2)

    # Test for compute_OptK
    def test_compute_OptK(self):
        # Case 1: Ensure no OptK when OptKStats is empty
        self.NMF.OptKStats = []  # Clear any existing statistics
        OptKs_empty = self.NMF.compute_OptK()
        self.assertEqual(OptKs_empty, None)

        # Case 2: Ensure no OptK when only one rank is available
        self.NMF.OptKStats = [
            {"rank": 2, "copheneticCoeff": 0.8, "meanAmariDist": 0.2}
        ]
        OptKs_single = self.NMF.compute_OptK()
        self.assertEqual(OptKs_single, None)

        # Case 3: Multiple ranks with one clear optimal rank
        self.NMF.OptKStats = [
            {"rank": 2, "copheneticCoeff": 0.75, "meanAmariDist": 0.3},
            {"rank": 3, "copheneticCoeff": 0.85, "meanAmariDist": 0.25},  # Optimal
            {"rank": 4, "copheneticCoeff": 0.7, "meanAmariDist": 0.35}
        ]
        OptKs_optimal = self.NMF.compute_OptK()
        self.assertEqual(self.NMF.OptK, [3])

        # Case 4: Multiple ranks with multiple optimal ranks
        self.NMF.OptKStats = [
            {"rank": 2, "copheneticCoeff": 0.9, "meanAmariDist": 0.2},  # Optimal
            {"rank": 3, "copheneticCoeff": 0.85, "meanAmariDist": 0.25},
            {"rank": 4, "copheneticCoeff": 0.9, "meanAmariDist": 0.2}   # Optimal
        ]
        OptKs_multiple = self.NMF.compute_OptK()
        self.assertEqual(self.NMF.OptK, [2, 4])

        # Case 5: Ranks with no clear intersection (no optimal K)
        self.NMF.OptKStats = [
            {"rank": 2, "copheneticCoeff": 0.75, "meanAmariDist": 0.35},
            {"rank": 3, "copheneticCoeff": 0.7, "meanAmariDist": 0.3},
            {"rank": 4, "copheneticCoeff": 0.65, "meanAmariDist": 0.25}
        ]
        OptKs_none = self.NMF.compute_OptK()
        self.assertEqual(self.NMF.OptK, [])

        # Case 6: Missing or malformed OptKStats keys
        self.NMF.OptKStats = [
            {"rank": 2, "copheneticCoeff": 0.85},  # Missing meanAmariDist
            {"rank": 3, "meanAmariDist": 0.3}  # Missing copheneticCoeff
        ]
        with self.assertRaises(KeyError):
            self.NMF.compute_OptK()

        self.NMF.OptKStats = [] # Setting back statistics

    def test_optK_Frob(self):
        # Set a single optimal rank
        self.NMF.OptK = [4]

        result = self.NMF.optK_Frob()
        opt_frob = self.NMF.frobenius[2] # index for rank 4

        self.assertEqual(result[0], opt_frob)

    # testing function for WcomputeFeatureStats
    def test_compute_all_ranks(self):
        """
        Test computation of feature contributions for all ranks when none exist initially.
        """
        result = self.NMF.WcomputeFeatureStats(ranks="all")
        
        # Check the resulting DataFrame dimensions and content
        self.assertEqual(result.shape[0], self.NMF.WMatrix[0].shape[0])  # 10 genes/features
        self.assertEqual(result.shape[1], sum(self.NMF.ranks))  # Sum of rank dimensions (2 + 3 + 4)
        self.assertTrue(all(col.startswith("Sig") for col in result.columns))  # Check column naming
    
    # testing function for WcomputeFeatureStats
    def test_compute_all_ranks(self):
        """
        Test computation of invalid rank.
        """
        result = self.NMF.WcomputeFeatureStats(ranks=[99])
        
        # Check the resulting DataFrame dimensions and content
        self.assertTrue(result.empty)

    # testing function for WcomputeFeatureStats
    def test_compute_all_ranks(self):
        """
        Test computation of invalid and valid ranks.
        """
        result = self.NMF.WcomputeFeatureStats(ranks=[3, 99])
        
        # Check the resulting DataFrame dimensions and content
        self.assertEqual(result.shape[0], self.NMF.WMatrix[0].shape[0])  # 10 genes/features
        self.assertEqual(result.shape[1], 3)  # Sum of rank dimensions (2 + 3 + 4)
        self.assertTrue(all(col.startswith("Sig") for col in result.columns))  # Check column naming



if __name__ == '__main__':
    unittest.main()
