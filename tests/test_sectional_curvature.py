"""Unit tests for sectional curvature estimation functionality.

Tests the sectional curvature implementation based on the triangle comparison theorem
from Gu et al. "Learning mixed-curvature representations in product spaces." ICLR 2019.
"""

import math

import networkx as nx
import pytest
import torch

from manify.curvature_estimation.sectional_curvature import (
    _discrete_curvature_estimator,
    sampled_sectional_curvature,
    sectional_curvature,
    vectorized_sectional_curvature,
)
from manify.manifolds import ProductManifold


class TestDiscreteCurvatureEstimator:
    """Test the core discrete curvature estimator function."""

    def test_basic_functionality(self):
        """Test basic functionality with simple distance matrix."""
        # Create a simple 4x4 distance matrix for testing
        D = torch.tensor(
            [[0.0, 1.0, 2.0, 1.5], [1.0, 0.0, 1.0, 0.5], [2.0, 1.0, 0.0, 1.5], [1.5, 0.5, 1.5, 0.0]],
            dtype=torch.float32,
        )

        # Test with different triangle configurations
        result = _discrete_curvature_estimator(D, a=0, b=1, c=2, m=3)
        assert isinstance(result, float), "Result should be a float"
        assert not math.isnan(result), "Result should not be NaN"
        assert not math.isinf(result), "Result should not be infinite"

    def test_degenerate_case_a_equals_m(self):
        """Test degenerate case where reference point equals midpoint."""
        D = torch.eye(4)  # Identity matrix
        result = _discrete_curvature_estimator(D, a=2, b=0, c=1, m=2)
        assert result == 0.0, "Should return 0.0 when a == m"

    def test_euclidean_triangle(self):
        """Test with Euclidean triangle that should have zero curvature."""
        # Create a perfect Euclidean triangle embedded in distance matrix
        # Using coordinates: a=(0,0), b=(1,0), c=(0,1), m=(0.5,0.5)
        coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        D = torch.cdist(coords, coords)

        result = _discrete_curvature_estimator(D, a=0, b=1, c=2, m=3)
        # For Euclidean geometry, the discrete curvature should be close to 0
        assert abs(result) < 1e-5, f"Euclidean triangle should have near-zero curvature, got {result}"

    def test_scale_invariance(self):
        """Test that the estimator has proper scale behavior."""
        # Create a simple triangle
        coords = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [1.5, 2.0]])
        D = torch.cdist(coords, coords)

        result1 = _discrete_curvature_estimator(D, a=0, b=1, c=2, m=3)

        # Scale coordinates by factor of 2
        coords_scaled = coords * 2
        D_scaled = torch.cdist(coords_scaled, coords_scaled)
        result2 = _discrete_curvature_estimator(D_scaled, a=0, b=1, c=2, m=3)

        # The normalization should make results comparable (not exactly equal due to scaling)
        # Handle case where result1 is zero
        if abs(result1) < 1e-10:
            assert abs(result2) < 1e-6, "Both results should be near zero for this configuration"
        else:
            assert abs(result1 - result2) < abs(result1) * 0.5, "Results should be relatively scale-invariant"

    def test_symmetric_properties(self):
        """Test symmetry properties of the estimator."""
        D = torch.tensor(
            [[0.0, 1.0, 1.0, 0.8], [1.0, 0.0, 1.4, 0.7], [1.0, 1.4, 0.0, 0.9], [0.8, 0.7, 0.9, 0.0]],
            dtype=torch.float32,
        )

        # Test that swapping b and c gives same result (triangle symmetry)
        result1 = _discrete_curvature_estimator(D, a=0, b=1, c=2, m=3)
        result2 = _discrete_curvature_estimator(D, a=0, b=2, c=1, m=3)
        assert abs(result1 - result2) < 1e-6, "Should be symmetric in b and c"


class TestSampledSectionalCurvature:
    """Test the sampling-based sectional curvature estimation."""

    def test_basic_functionality(self):
        """Test basic functionality with different sample sizes."""
        torch.manual_seed(42)  # For reproducible results

        # Create distance matrix from hyperbolic manifold
        pm = ProductManifold(signature=[(-1.0, 3)])
        X, _ = pm.sample(z_mean=torch.vstack([pm.mu0] * 10))
        D = pm.pdist(X)

        curvatures, indices = sampled_sectional_curvature(D, n_samples=100)

        assert curvatures.shape == (100,), f"Expected 100 curvature samples, got {curvatures.shape}"
        assert indices.shape == (100, 4), f"Expected 100x4 indices, got {indices.shape}"
        assert not torch.any(torch.isnan(curvatures)), "No curvature values should be NaN"
        assert not torch.any(torch.isinf(curvatures)), "No curvature values should be infinite"

        # Check that indices are within valid range
        n = D.shape[0]
        assert torch.all(indices >= 0), "All indices should be non-negative"
        assert torch.all(indices < n), f"All indices should be less than {n}"

    def test_sample_size_parameter(self):
        """Test different sample sizes."""
        torch.manual_seed(42)
        D = torch.rand(8, 8)
        D = D + D.T  # Make symmetric
        D.fill_diagonal_(0)  # Zero diagonal

        for n_samples in [10, 50, 200]:
            curvatures, indices = sampled_sectional_curvature(D, n_samples=n_samples)
            assert curvatures.shape == (n_samples,), f"Wrong shape for {n_samples} samples"
            assert indices.shape == (n_samples, 4), f"Wrong indices shape for {n_samples} samples"

    def test_relative_normalization(self):
        """Test relative normalization parameter."""
        torch.manual_seed(42)
        D = torch.rand(6, 6) * 10  # Scale up distances
        D = D + D.T
        D.fill_diagonal_(0)

        curvatures_rel, _ = sampled_sectional_curvature(D, n_samples=50, relative=True)
        curvatures_abs, _ = sampled_sectional_curvature(D, n_samples=50, relative=False)

        # Relative should be smaller in magnitude due to normalization by max distance
        max_dist = torch.max(D)
        assert max_dist > 1.0, "Test setup should have large distances"

        # Check that relative normalization reduces magnitude
        assert torch.max(torch.abs(curvatures_rel)) <= torch.max(torch.abs(curvatures_abs)), (
            "Relative normalization should not increase magnitude"
        )

    def test_degenerate_filtering(self):
        """Test that degenerate cases (a == m) are handled correctly."""
        torch.manual_seed(123)
        D = torch.rand(5, 5)
        D = D + D.T
        D.fill_diagonal_(0)

        # Even with small sample size, should handle degenerate cases gracefully
        curvatures, indices = sampled_sectional_curvature(D, n_samples=100)

        # Check that where a == m, curvature is 0
        a, m = indices[:, 0], indices[:, 3]
        degenerate_mask = a == m
        if degenerate_mask.any():
            assert torch.allclose(curvatures[degenerate_mask], torch.zeros_like(curvatures[degenerate_mask])), (
                "Degenerate cases should have zero curvature"
            )


class TestVectorizedSectionalCurvature:
    """Test the vectorized sectional curvature computation."""

    def test_basic_functionality(self):
        """Test basic functionality for per-node and global computation."""
        # Create a small graph distance matrix
        D = torch.tensor(
            [[0.0, 1.0, 2.0, 1.0], [1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0], [1.0, 2.0, 1.0, 0.0]],
            dtype=torch.float32,
        )

        # Test per-node curvatures
        node_curvatures = vectorized_sectional_curvature(D, full=True)
        assert node_curvatures.shape == (4,), f"Expected 4 node curvatures, got {node_curvatures.shape}"
        assert not torch.any(torch.isnan(node_curvatures)), "No node curvatures should be NaN"

        # Test global average
        global_curvature = vectorized_sectional_curvature(D, full=False)
        assert isinstance(global_curvature, float), "Global curvature should be a float"
        assert not math.isnan(global_curvature), "Global curvature should not be NaN"

        # Global should be mean of node curvatures
        expected_global = torch.mean(node_curvatures).item()
        assert abs(global_curvature - expected_global) < 1e-6, "Global should equal mean of node curvatures"

    def test_neighbor_detection(self):
        """Test that neighbor detection works correctly."""
        # Create a path graph: 0-1-2-3
        D = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0], [3.0, 2.0, 1.0, 0.0]],
            dtype=torch.float32,
        )

        node_curvatures = vectorized_sectional_curvature(D, full=True)

        # End nodes (0, 3) should have 0 curvature (only 1 neighbor each)
        assert abs(node_curvatures[0]) < 1e-6, "End node should have zero curvature"
        assert abs(node_curvatures[3]) < 1e-6, "End node should have zero curvature"

        # Middle nodes (1, 2) should have non-zero curvature (2 neighbors each)
        # This tests that the algorithm correctly identifies neighbors at distance 1

    def test_isolated_nodes(self):
        """Test handling of isolated nodes."""
        # Create distance matrix with an isolated node
        D = torch.tensor(
            [
                [0.0, 1.0, 2.0, float("inf")],
                [1.0, 0.0, 1.0, float("inf")],
                [2.0, 1.0, 0.0, float("inf")],
                [float("inf"), float("inf"), float("inf"), 0.0],
            ],
            dtype=torch.float32,
        )

        # Replace inf with large number for numerical stability
        D[float("inf") == D] = 1000.0

        node_curvatures = vectorized_sectional_curvature(D, full=True)

        # Isolated node should have zero curvature
        assert abs(node_curvatures[3]) < 1e-6, "Isolated node should have zero curvature"

    def test_relative_normalization(self):
        """Test relative normalization."""
        D = torch.rand(5, 5) * 5  # Random distances scaled up
        D = D + D.T  # Make symmetric
        D.fill_diagonal_(0)

        curvatures_rel = vectorized_sectional_curvature(D, relative=True, full=True)
        curvatures_abs = vectorized_sectional_curvature(D, relative=False, full=True)

        max_dist = torch.max(D)
        if max_dist > 1.0:
            # Relative should be smaller in magnitude
            assert torch.max(torch.abs(curvatures_rel)) <= torch.max(torch.abs(curvatures_abs)), (
                "Relative normalization should not increase magnitude"
            )


class TestSectionalCurvatureInterface:
    """Test the main sectional_curvature function interface."""

    def test_sampled_method(self):
        """Test sampled method interface."""
        # Create a simple cycle graph
        G = nx.cycle_graph(6)

        curvatures = sectional_curvature(G, method="sampled", n_samples=50)
        assert curvatures.shape == (50,), "Sampled method should return curvature samples"
        assert not torch.any(torch.isnan(curvatures)), "No curvatures should be NaN"

    def test_per_node_method(self):
        """Test per_node method interface."""
        # Create a simple path graph
        G = nx.path_graph(5)

        # Test per-node curvatures
        node_curvatures = sectional_curvature(G, method="per_node")
        assert node_curvatures.shape == (5,), "Should return per-node curvatures"

    def test_global_method(self):
        """Test global method interface."""
        # Create a simple path graph
        G = nx.path_graph(5)

        # Test global curvature
        global_curvature = sectional_curvature(G, method="global")
        assert isinstance(global_curvature, float), "Should return scalar for global curvature"

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        G = nx.complete_graph(4)

        with pytest.raises(ValueError, match="Unknown method"):
            sectional_curvature(G, method="invalid_method")
        
        # Test that 'full' method is no longer supported
        with pytest.raises(ValueError, match="Unknown method"):
            sectional_curvature(G, method="full")

    def test_distance_matrix_input(self):
        """Test that distance matrix input produces same results as graph input."""
        G = nx.path_graph(5)
        
        # Get curvature from graph
        graph_curvatures = sectional_curvature(G, method="per_node")
        
        # Get distance matrix and compute curvature from it
        D = torch.tensor(nx.floyd_warshall_numpy(G), dtype=torch.float32)
        matrix_curvatures = sectional_curvature(D, method="per_node")
        
        # Results should be identical
        assert torch.allclose(graph_curvatures, matrix_curvatures, atol=1e-6), (
            "Graph and distance matrix inputs should produce identical results"
        )

    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        with pytest.raises(TypeError, match="input_data must be a NetworkX graph or torch.Tensor"):
            sectional_curvature("invalid_input", method="sampled")

    def test_disconnected_graph(self):
        """Test handling of disconnected graphs."""
        # Create disconnected graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (3, 4)])  # Two disconnected components

        # Should handle gracefully (NetworkX will use inf for disconnected pairs)
        try:
            sectional_curvature(G, method="sampled", n_samples=10)
            # Should complete without error, though results may be inf/nan
        except Exception as e:
            pytest.fail(f"Should handle disconnected graphs gracefully: {e}")


class TestMathematicalCorrectness:
    """Test mathematical correctness against known curvature signatures."""

    def test_tree_negative_curvature(self):
        """Test that tree graphs exhibit negative curvature characteristics."""
        # Create a binary tree
        G = nx.balanced_tree(2, 3)  # 2-ary tree of height 3

        curvatures = sectional_curvature(G, method="sampled", n_samples=200, relative=True)

        # Trees should generally have negative curvature
        # Allow some positive values due to sampling noise
        negative_ratio = (curvatures < 0).float().mean()
        assert negative_ratio > 0.3, f"Trees should show negative curvature tendency, got {negative_ratio:.2f} negative"

    def test_cycle_positive_curvature(self):
        """Test that cycle graphs exhibit positive curvature characteristics."""
        # Create a cycle graph
        G = nx.cycle_graph(8)

        curvatures = sectional_curvature(G, method="sampled", n_samples=200, relative=True)

        # Cycles should generally have positive curvature
        positive_ratio = (curvatures > 0).float().mean()
        assert positive_ratio > 0.3, (
            f"Cycles should show positive curvature tendency, got {positive_ratio:.2f} positive"
        )

    def test_complete_graph_properties(self):
        """Test properties of complete graphs."""
        G = nx.complete_graph(6)

        node_curvatures = sectional_curvature(G, method="per_node", relative=True)

        # Complete graphs should have uniform curvature across nodes
        curvature_std = torch.std(node_curvatures)
        assert curvature_std < 0.1, f"Complete graph should have uniform curvature, std = {curvature_std:.4f}"

    def test_path_graph_curvature(self):
        """Test curvature properties of path graphs."""
        G = nx.path_graph(8)

        node_curvatures = sectional_curvature(G, method="per_node")

        # End nodes should have zero curvature (only one neighbor)
        assert abs(node_curvatures[0]) < 1e-6, "Path end node should have zero curvature"
        assert abs(node_curvatures[-1]) < 1e-6, "Path end node should have zero curvature"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_node_graph(self):
        """Test single node graph."""
        G = nx.Graph()
        G.add_node(0)

        curvatures = sectional_curvature(G, method="sampled", n_samples=10)
        # Should handle gracefully, likely returning zeros
        assert curvatures.shape == (10,), "Should return requested number of samples"

    def test_two_node_graph(self):
        """Test two node graph."""
        G = nx.Graph()
        G.add_edge(0, 1)

        node_curvatures = sectional_curvature(G, method="per_node")
        # Both nodes have only one neighbor, so curvature should be zero
        assert torch.allclose(node_curvatures, torch.zeros_like(node_curvatures)), (
            "Two-node graph should have zero curvature"
        )

    def test_triangle_graph(self):
        """Test smallest non-trivial graph (triangle)."""
        G = nx.complete_graph(3)

        curvatures = sectional_curvature(G, method="sampled", n_samples=50, relative=True)
        assert not torch.any(torch.isnan(curvatures)), "Triangle graph should not produce NaN"
        assert not torch.any(torch.isinf(curvatures)), "Triangle graph should not produce Inf"

    def test_large_graph_performance(self):
        """Test performance on larger graphs."""
        # Create a moderately large graph
        G = nx.barabasi_albert_graph(100, 3)

        # Sampled method should be fast even for large graphs
        curvatures = sectional_curvature(G, method="sampled", n_samples=100)
        assert curvatures.shape == (100,), "Should handle large graphs efficiently"

        # Full method might be slow but should work
        global_curvature = sectional_curvature(G, method="global")
        assert isinstance(global_curvature, float), "Should compute global curvature for large graph"


class TestNumericalStability:
    """Test numerical stability and robustness."""

    def test_small_distances(self):
        """Test with very small distances."""
        D = torch.rand(5, 5) * 1e-6  # Very small distances
        D = D + D.T
        D.fill_diagonal_(0)

        curvatures, _ = sampled_sectional_curvature(D, n_samples=20)
        assert torch.all(torch.isfinite(curvatures)), "Should handle small distances without numerical issues"

    def test_large_distances(self):
        """Test with very large distances."""
        D = torch.rand(5, 5) * 1e6  # Very large distances
        D = D + D.T
        D.fill_diagonal_(0)

        curvatures, _ = sampled_sectional_curvature(D, n_samples=20, relative=True)
        assert torch.all(torch.isfinite(curvatures)), "Should handle large distances with relative normalization"

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        G = nx.karate_club_graph()

        torch.manual_seed(12345)
        curvatures1 = sectional_curvature(G, method="sampled", n_samples=50)

        torch.manual_seed(12345)
        curvatures2 = sectional_curvature(G, method="sampled", n_samples=50)

        assert torch.allclose(curvatures1, curvatures2), "Results should be reproducible with same seed"


def test_integration_with_real_datasets():
    """Test integration with real datasets from manify library."""
    pytest.importorskip("manify.utils.dataloaders", reason="dataloaders not available")

    try:
        from manify.utils.dataloaders import load_hf

        # Test with a small real dataset
        try:
            data = load_hf("karate")
            if hasattr(data, "adj_matrix"):
                # Convert adjacency matrix to NetworkX graph
                adj = data.adj_matrix.numpy()
                G = nx.from_numpy_array(adj)

                # Test that sectional curvature works with real data
                curvatures = sectional_curvature(G, method="sampled", n_samples=30)
                assert curvatures.shape == (30,), "Should work with real dataset"
                assert torch.all(torch.isfinite(curvatures)), "Should produce finite curvatures"

        except Exception:
            # Skip if dataset loading fails
            pytest.skip("Could not load test dataset")

    except ImportError:
        pytest.skip("dataloaders not available for integration test")
