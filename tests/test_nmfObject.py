import unittest
import numpy as np

# Assuming the NMFobject class is imported here

class TestNMFObject(unittest.TestCase):
    def setUp(self):
        # Create an instance of NMFobject for testing
        # Example initialization values
        self.nmf = NMFobject(
            k=5,
            H=np.random.rand(5, 10),
            W=[np.random.rand(10, 20) for _ in range(5)]
        )

    def test_normalise_W_already_normalized(self):
        # Test if W matrix is already normalized
        # Setting W matrix to already normalized values
        normalized_W = [np.random.rand(10, 20) for _ in range(5)]
        self.nmf.W = normalized_W
        self.assertEqual(self.nmf.normalise_W, normalized_W,
                         "W matrix should remain unchanged if already normalized")

    def test_normalise_W(self):
        # Test the normalization of W matrix
        # Create random W matrix
        W = [np.random.rand(10, 20) for _ in range(5)]
        self.nmf.W = W
        normalized_W = self.nmf.normalise_W

        # Check if the sum of each column in each W matrix is approximately 1
        for w_matrix in normalized_W:
            self.assertTrue(np.allclose(np.sum(w_matrix, axis=0), 1),
                            "Columns in W matrix should sum up to 1 after normalization")

if __name__ == '__main__':
    unittest.main()
