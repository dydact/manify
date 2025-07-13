"""Unit tests for sectional curvature estimation functionality."""

import math
import pytest
import torch

from manify.curvature_estimation.sectional_curvature import (
    _discrete_curvature_estimator,
    sampled_sectional_curvature,
    sectional_curvature,
    vectorized_sectional_curvature,
)


class TestDiscreteCurvatureEstimator:
    """Test the core discrete curvature estimator function."""

    def test_basic_functionality(self):
        """Test basic functionality with simple distance matrix."""
        D = torch.tensor(
            [[0.0, 1.0, 2.0, 1.5], [1.0, 0.0, 1.0, 0.5], [2.0, 1.0, 0.0, 1.5], [1.5, 0.5, 1.5, 0.0]],
            dtype=torch.float32,
        )

        result = _discrete_curvature_estimator(D, a=0, b=1, c=2, m=3)
        assert isinstance(result, float)
        assert not math.isnan(result)
        assert not math.isinf(result)

    def test_degenerate_case_a_equals_m(self):
        """Test degenerate case where reference point equals midpoint."""
        D = torch.eye(4)
        result = _discrete_curvature_estimator(D, a=2, b=0, c=1, m=2)
        assert result == 0.0


class TestSampledSectionalCurvature:
    """Test the sampling-based sectional curvature estimation."""

    def test_basic_functionality(self):
        """Test basic functionality with different sample sizes."""
        torch.manual_seed(42)
        D = torch.rand(6, 6)
        D = D + D.T
        D.fill_diagonal_(0)

        curvatures, indices = sampled_sectional_curvature(D, n_samples=50)

        assert curvatures.shape == (50,)
        assert indices.shape == (50, 4)
        assert not torch.any(torch.isnan(curvatures))
        assert not torch.any(torch.isinf(curvatures))
        assert torch.all(indices >= 0)
        assert torch.all(indices < 6)


class TestVectorizedSectionalCurvature:
    """Test the vectorized sectional curvature computation."""

    def test_basic_functionality(self):
        """Test basic functionality for per-node and global computation."""
        D = torch.tensor(
            [[0.0, 1.0, 2.0, 1.0], [1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, 1.0], [1.0, 2.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        A = (D == 1.0).float()
        
        # Test per-node curvatures
        node_curvatures = vectorized_sectional_curvature(A, D, full=True)
        assert node_curvatures.shape == (4,)
        assert not torch.any(torch.isnan(node_curvatures))

        # Test global average
        global_curvature = vectorized_sectional_curvature(A, D, full=False)
        assert isinstance(global_curvature, float)
        assert not math.isnan(global_curvature)

    def test_isolated_nodes(self):
        """Test handling of isolated nodes."""
        D = torch.tensor(
            [
                [0.0, 1.0, 2.0, 1000.0],
                [1.0, 0.0, 1.0, 1000.0],
                [2.0, 1.0, 0.0, 1000.0],
                [1000.0, 1000.0, 1000.0, 0.0],
            ],
            dtype=torch.float32,
        )
        A = (D == 1.0).float()
        
        node_curvatures = vectorized_sectional_curvature(A, D, full=True)
        # Isolated node should have zero curvature
        assert abs(node_curvatures[3]) < 1e-6


class TestSectionalCurvatureInterface:
    """Test the main sectional_curvature function interface."""

    def test_sampled_method(self):
        """Test sampled method interface."""
        A = torch.zeros(5, 5)
        for i in range(4):
            A[i, i + 1] = A[i + 1, i] = 1
        
        D = torch.zeros(5, 5)
        for i in range(5):
            for j in range(5):
                if i != j:
                    D[i, j] = abs(i - j)

        curvatures = sectional_curvature(A, D, method="sampled", n_samples=20)
        assert curvatures.shape == (20,)
        assert not torch.any(torch.isnan(curvatures))

    def test_per_node_method(self):
        """Test per_node method interface."""
        A = torch.zeros(5, 5)
        for i in range(4):
            A[i, i + 1] = A[i + 1, i] = 1
        
        D = torch.zeros(5, 5)
        for i in range(5):
            for j in range(5):
                if i != j:
                    D[i, j] = abs(i - j)

        node_curvatures = sectional_curvature(A, D, method="per_node")
        assert node_curvatures.shape == (5,)

    def test_global_method(self):
        """Test global method interface."""
        A = torch.zeros(5, 5)
        for i in range(4):
            A[i, i + 1] = A[i + 1, i] = 1
        
        D = torch.zeros(5, 5)
        for i in range(5):
            for j in range(5):
                if i != j:
                    D[i, j] = abs(i - j)

        global_curvature = sectional_curvature(A, D, method="global")
        assert isinstance(global_curvature, float)

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        A = torch.ones(4, 4) - torch.eye(4)
        D = torch.ones(4, 4)
        D.fill_diagonal_(0)

        with pytest.raises(ValueError, match="Unknown method"):
            sectional_curvature(A, D, method="invalid_method")

    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="Adjacency matrix and distance matrix must have the same shape"):
            sectional_curvature(torch.eye(3), torch.eye(4), method="sampled")

        with pytest.raises(TypeError, match="Both adjacency_matrix and distance_matrix must be torch.Tensors"):
            sectional_curvature("invalid", torch.eye(3), method="sampled")


class TestMathematicalCorrectness:
    """Test mathematical correctness against known curvature signatures."""

    def test_tree_negative_curvature(self):
        """Test that tree graphs exhibit negative curvature characteristics."""
        # Simple binary tree: 0 -> {1,2}, 1 -> {3,4}
        A = torch.zeros(5, 5)
        edges = [(0,1), (0,2), (1,3), (1,4)]
        for i, j in edges:
            A[i, j] = A[j, i] = 1
        
        # Tree distances
        D = torch.full((5, 5), float('inf'))
        D.fill_diagonal_(0)
        D[A == 1] = 1
        
        # Floyd-Warshall
        for k in range(5):
            for i in range(5):
                for j in range(5):
                    D[i, j] = min(D[i, j], D[i, k] + D[k, j])

        curvatures = sectional_curvature(A, D, method="sampled", n_samples=100, relative=True)
        
        # Trees should generally have negative curvature
        negative_ratio = (curvatures < 0).float().mean()
        assert negative_ratio > 0.1

    def test_cycle_positive_curvature(self):
        """Test that cycle graphs exhibit positive curvature characteristics."""
        # 6-node cycle
        A = torch.zeros(6, 6)
        for i in range(6):
            A[i, (i + 1) % 6] = A[i, (i - 1) % 6] = 1
        
        # Cycle distances
        D = torch.zeros(6, 6)
        for i in range(6):
            for j in range(6):
                if i != j:
                    clockwise = (j - i) % 6
                    counterclockwise = (i - j) % 6
                    D[i, j] = min(clockwise, counterclockwise)

        curvatures = sectional_curvature(A, D, method="sampled", n_samples=100, relative=True)
        
        # Cycles should generally have positive curvature
        positive_ratio = (curvatures > 0).float().mean()
        assert positive_ratio > 0.3


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_node_graph(self):
        """Test single node graph."""
        A = torch.zeros(1, 1)
        D = torch.zeros(1, 1)

        curvatures = sectional_curvature(A, D, method="sampled", n_samples=10)
        assert curvatures.shape == (10,)

    def test_two_node_graph(self):
        """Test two node graph."""
        A = torch.tensor([[0., 1.], [1., 0.]])
        D = torch.tensor([[0., 1.], [1., 0.]])

        node_curvatures = sectional_curvature(A, D, method="per_node")
        # Both nodes have only one neighbor, so curvature should be zero
        assert torch.allclose(node_curvatures, torch.zeros_like(node_curvatures))