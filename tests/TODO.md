# Tests to roll in from `notebooks`:
* `2_polblogs_benchmark.ipynb`:
    * Verify that ProductDT = Sklearn DT for Euclidean manifolds
    * Verify `dist_component_by_manifold` sums to 1
    * __NEW__: Check equivalence to Euclidean case for other models as well
* `3_verify_shapes.ipynb`:
    * Assert `Manifold.pdist()` is close to sqrt of `Manifold.pdist2()`
    * __NEW__: Check this for all `dist2` functions
    * Assert pdists is sum of squared dists:
        ```python
        assert torch.allclose(pdists[j, k], M.dist2(x_embed[j, dims], x_embed[k, dims]).sum(), atol=1e-6)
        ```
* `5_sampling.ipynb`:
    * Check that distances to origin are the same for all wrapped normal distributions (except spherical for very high curvature)
    * Check that log-likelihoods are generally positive (Q(z) - P(z)): (repeat with ProductManifolds)
        ```python
        for K in [-2, -1.0, -0.5, 0, 0.5, 1.0, 2.0]:
            print(K)
            m = Manifold(K, 4)
            # Pick a random point to use as the center
            mu = m.sample(m.mu0)
            Sigma = torch.diag(torch.randn(m.dim)) ** 2
            samples = m.sample(z_mean=torch.cat([mu] * N_SAMPLES, dim=0), sigma=Sigma)
            log_probs_p = m.log_likelihood(z=samples)  # Default args
            log_probs_q = m.log_likelihood(z=samples, mu=mu, sigma=Sigma)
            print(
                f"Shape: {log_probs_p.shape},\tP(z) = {log_probs_p.mean().item():.3f},\tQ(z) = {log_probs_q.mean().item():.3f},\tQ(z) - P(z) = {log_probs_q.mean().item() - log_probs_p.mean().item():.3f}"
            )
            print()
        ```
    * Check that KL divergence is equal to this difference
* `10_torchified_hyperdt.ipynb`:
    * Implement legacy (iterative) version of MCDT class
    * Assert info gains are the same
    * Assert splits are the same
    * __TODO__: revisit this more carefully. Probably notebook 13 supersedes all of this.
* `13_information_gain.ipynb`:
    * Verify comparisons tensor (`verification_tensor`)
    * Verify information gains (`ig_gains_nonan`)
    * Verify angles (`angles`)
* `14_covariance_scaling.ipynb`:
    * Verify dividing variance by dimension gives you ~same norm of spacelike dimensions
* `17_verify_new_mse.ipynb`:
    * Similar to notebook 13 - need to compare information gains in regression setting
* `47_stereographic_tests.ipynb`:
    * Does the ReLU decorator work correctly?
    * Stereographic projections are ~invertible
    * mu0 becomes 0 in stereographic projection
    * Projection matrix `man.projection_matrix` is identity matrix in stereographic manifolds
    * Check that our complicated left-multiplication is equivalent to this:
    ```python
        def left_multiply(self, A, X):
        out = torch.zeros_like(X)
        for i, (A_i, X_i) in enumerate(zip(A, X)):
            m_i = self.manifold.manifold.weighted_midpoint(xs=X_i, weights=A_i)
            out[i] = self.manifold.manifold.mobius_scalar_mul(r=A_i.sum(), x=m_i)
        return out
    ```
    * Verify stereographic logits are smooth at kappa=0 (how?)
