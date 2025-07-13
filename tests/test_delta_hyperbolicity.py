import pytest
import torch

from manify.curvature_estimation.delta_hyperbolicity import sampled_delta_hyperbolicity, vectorized_delta_hyperbolicity, delta_hyperbolicity
from manify.manifolds import ProductManifold


def gromov_product(i, j, k, D):
    """(j,k)_i = 0.5 (d(i,j) + d(i,k) - d(j,k))"""
    return float(0.5 * (D[i, j] + D[i, k] - D[j, k]))


class TestSampledDeltaHyperbolicity:
    """Test the sampling-based delta hyperbolicity computation."""
    
    def test_basic_functionality(self):
        """Test basic functionality with hyperbolic manifold data."""
        torch.manual_seed(42)
        pm = ProductManifold(signature=[(-1.0, 2)])
        X, _ = pm.sample(z_mean=torch.vstack([pm.mu0] * 10))
        dists = pm.pdist(X)
        
        sampled_deltas, indices = sampled_delta_hyperbolicity(dists, n_samples=50, relative=True)
        
        assert sampled_deltas.shape == (50,), "Should return correct number of samples"
        assert indices.shape == (50, 3), "Should return triplet indices"
        assert (sampled_deltas <= 1).all(), "Relative deltas should be <= 1"
        assert (sampled_deltas >= -2).all(), "Relative deltas should be >= -2"
        assert torch.all(indices >= 0), "All indices should be non-negative"
        assert torch.all(indices < 10), "All indices should be within range"
    
    def test_different_sample_sizes(self):
        """Test with different sample sizes."""
        torch.manual_seed(42)
        D = torch.rand(8, 8)
        D = D + D.T  # Make symmetric
        D.fill_diagonal_(0)
        
        for n_samples in [1, 10, 100]:
            deltas, indices = sampled_delta_hyperbolicity(D, n_samples=n_samples)
            assert deltas.shape == (n_samples,), f"Wrong shape for {n_samples} samples"
            assert indices.shape == (n_samples, 3), f"Wrong indices shape for {n_samples} samples"
    
    def test_relative_vs_absolute(self):
        """Test relative vs absolute normalization."""
        torch.manual_seed(42)
        D = torch.rand(6, 6) * 10  # Large distances
        D = D + D.T
        D.fill_diagonal_(0)
        
        deltas_rel, _ = sampled_delta_hyperbolicity(D, n_samples=50, relative=True)
        deltas_abs, _ = sampled_delta_hyperbolicity(D, n_samples=50, relative=False)
        
        # Relative should be normalized by max distance
        assert torch.max(torch.abs(deltas_rel)) <= torch.max(torch.abs(deltas_abs))
    
    def test_reference_point_parameter(self):
        """Test different reference points."""
        torch.manual_seed(42)
        D = torch.rand(8, 8)
        D = D + D.T
        D.fill_diagonal_(0)
        
        deltas1, _ = sampled_delta_hyperbolicity(D, n_samples=50, reference_idx=0)
        deltas2, _ = sampled_delta_hyperbolicity(D, n_samples=50, reference_idx=3)
        
        # Results may differ with different reference points
        assert deltas1.shape == deltas2.shape
        assert torch.all(torch.isfinite(deltas1))
        assert torch.all(torch.isfinite(deltas2))


class TestVectorizedDeltaHyperbolicity:
    """Test the vectorized delta hyperbolicity computation."""
    
    def test_basic_functionality(self):
        """Test basic vectorized computation."""
        torch.manual_seed(42)
        pm = ProductManifold(signature=[(-1.0, 2)])
        X, _ = pm.sample(z_mean=torch.vstack([pm.mu0] * 8))
        dists = pm.pdist(X)
        
        # Test full computation
        full_deltas = vectorized_delta_hyperbolicity(dists, full=True, relative=True)
        assert full_deltas.shape == (8, 8, 8), "Full computation should return 3D tensor"
        assert (full_deltas <= 1).all(), "Relative deltas should be <= 1"
        assert (full_deltas >= -2).all(), "Relative deltas should be >= -2"
        
        # Test global maximum
        global_delta = vectorized_delta_hyperbolicity(dists, full=False, relative=True)
        assert isinstance(global_delta, float), "Global method should return float"
        assert global_delta == torch.max(full_deltas).item(), "Global should equal max of full"
    
    def test_consistency_with_sampled(self):
        """Test that vectorized and sampled methods are consistent."""
        torch.manual_seed(42)
        D = torch.rand(6, 6)
        D = D + D.T
        D.fill_diagonal_(0)
        
        # Get full computation
        full_deltas = vectorized_delta_hyperbolicity(D, full=True, relative=True)
        
        # Get sampled computation with same seed
        torch.manual_seed(42)
        sampled_deltas, indices = sampled_delta_hyperbolicity(D, n_samples=20, relative=True)
        
        # Sampled values should match corresponding full values
        expected_sampled = full_deltas[indices[:, 0], indices[:, 1], indices[:, 2]]
        assert torch.allclose(sampled_deltas, expected_sampled, atol=1e-6)
    
    def test_tree_metric_properties(self):
        """Test with tree metric (should have low delta hyperbolicity)."""
        # Create a simple tree metric (star graph)
        n = 5
        D = torch.full((n, n), 2.0)  # All pairs distance 2 (through center)
        D[0, :] = 1.0  # Center to all nodes distance 1
        D[:, 0] = 1.0
        D.fill_diagonal_(0)
        
        global_delta = vectorized_delta_hyperbolicity(D, full=False, relative=False)
        # Tree metrics should have delta-hyperbolicity = 0
        assert abs(global_delta) < 1e-6, "Tree metric should have near-zero delta hyperbolicity"
    
    def test_euclidean_triangle_inequality(self):
        """Test with Euclidean distances from geometric points."""
        # Create points in Euclidean space
        coords = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=torch.float32)
        D = torch.cdist(coords, coords)
        
        global_delta = vectorized_delta_hyperbolicity(D, full=False, relative=False)
        # Euclidean spaces should have small delta hyperbolicity
        assert global_delta >= 0, "Delta hyperbolicity should be non-negative"
        assert global_delta < 0.5, "Euclidean metric should have small delta hyperbolicity"


class TestDeltaHyperbolicityInterface:
    """Test the main delta_hyperbolicity function interface."""
    
    def test_method_parameter(self):
        """Test different method parameters."""
        torch.manual_seed(42)
        D = torch.rand(6, 6)
        D = D + D.T
        D.fill_diagonal_(0)
        
        # Test sampled method
        sampled_result = delta_hyperbolicity(D, method="sampled", n_samples=20)
        assert sampled_result.shape == (20,), "Sampled method should return samples"
        
        # Test global method
        global_result = delta_hyperbolicity(D, method="global")
        assert isinstance(global_result, float), "Global method should return float"
        
        # Test full method
        full_result = delta_hyperbolicity(D, method="full")
        assert full_result.shape == (6, 6, 6), "Full method should return 3D tensor"
    
    def test_invalid_method(self):
        """Test error handling for invalid methods."""
        D = torch.eye(4)
        
        with pytest.raises(ValueError, match="Unknown method"):
            delta_hyperbolicity(D, method="invalid")
    
    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        with pytest.raises(TypeError, match="distance_matrix must be a torch.Tensor"):
            delta_hyperbolicity("invalid", method="global")
    
    def test_parameter_passing(self):
        """Test that additional parameters are passed correctly."""
        torch.manual_seed(42)
        D = torch.rand(5, 5)
        D = D + D.T
        D.fill_diagonal_(0)
        
        # Test with custom parameters
        result1 = delta_hyperbolicity(D, method="sampled", n_samples=30, reference_idx=2)
        result2 = delta_hyperbolicity(D, method="global", reference_idx=2, relative=False)
        
        assert result1.shape == (30,), "Parameters should be passed to sampled method"
        assert isinstance(result2, float), "Parameters should be passed to global method"


class TestGromovProduct:
    """Test the Gromov product computation helper."""
    
    def test_gromov_product_properties(self):
        """Test mathematical properties of Gromov products."""
        # Create simple distance matrix
        D = torch.tensor([
            [0., 3., 4., 5.],
            [3., 0., 5., 6.],
            [4., 5., 0., 3.],
            [5., 6., 3., 0.]
        ], dtype=torch.float32)
        
        # Test symmetry: (j,k)_i = (k,j)_i
        assert abs(gromov_product(0, 1, 2, D) - gromov_product(0, 2, 1, D)) < 1e-6
        
        # Test non-negativity for valid metric
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    gp = gromov_product(i, j, k, D)
                    assert gp >= -1e-6, f"Gromov product should be non-negative, got {gp}"
    
    def test_gromov_product_degenerate_cases(self):
        """Test Gromov product with degenerate cases."""
        D = torch.eye(4) * 0  # All distances zero
        
        # All Gromov products should be zero when all distances are zero
        assert gromov_product(0, 1, 2, D) == 0.0
        assert gromov_product(1, 0, 3, D) == 0.0


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_small_matrices(self):
        """Test with very small matrices."""
        # Single point
        D1 = torch.zeros(1, 1)
        result = delta_hyperbolicity(D1, method="global")
        assert result == 0.0, "Single point should have zero hyperbolicity"
        
        # Two points
        D2 = torch.tensor([[0., 1.], [1., 0.]])
        result = delta_hyperbolicity(D2, method="global")
        assert result == 0.0, "Two points should have zero hyperbolicity"
    
    def test_large_distances(self):
        """Test with very large distances."""
        D = torch.rand(5, 5) * 1e6
        D = D + D.T
        D.fill_diagonal_(0)
        
        # Should not produce NaN or inf with relative normalization
        result = delta_hyperbolicity(D, method="global", relative=True)
        assert torch.isfinite(torch.tensor(result)), "Should handle large distances"
    
    def test_zero_distances(self):
        """Test with zero distance matrix."""
        D = torch.zeros(4, 4)
        
        result = delta_hyperbolicity(D, method="global")
        assert result == 0.0, "Zero distances should give zero hyperbolicity"
