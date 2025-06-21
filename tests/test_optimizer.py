import torch
import geoopt
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam


from manify.optimizers.radan import RiemannianAdan
from manify.manifolds import ProductManifold


def get_product_manifold_and_target(device_str: str):
    """
    Construct a product manifold R^2 x R^2 and a target point.
    """
    signature = [(0.0, 2), (0.0, 2)]  # Two Euclidean spaces
    manify_pm_wrapper = ProductManifold(signature=signature, device=device_str, stereographic=False)
    product_manifold = manify_pm_wrapper.manifold
    target_point_tensor = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float32)
    return product_manifold, target_point_tensor


def objective_function(point, target_point, manifold):
    """
    Objective function: squared distance to the target point.
    """
    return manifold.dist(point, target_point) ** 2


def optimize_and_compare(
    manifold,
    target_point_tensor,
    optimizer_class,
    optimizer_params,
    initial_point_tensor,
    num_iterations=200,
    lr=0.1,
    tol=1e-5,
):
    """
    Optimize the initial point using the specified Riemannian optimizer.
    """
    point_to_optimize = ManifoldParameter(initial_point_tensor.clone().requires_grad_(True), manifold=manifold)

    if optimizer_class.__name__ == "RiemannianAdan":
        current_optimizer_params = optimizer_params.copy()
        current_optimizer_params.setdefault("betas", (0.92, 0.98, 0.99))
        optimizer = optimizer_class([point_to_optimize], lr=lr, **current_optimizer_params)
    else:
        optimizer = optimizer_class([point_to_optimize], lr=lr, **optimizer_params)

    losses = []
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = objective_function(point_to_optimize, target_point_tensor.to(point_to_optimize.device), manifold)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i > 0 and abs(losses[-1] - losses[-2]) < tol:
            break

    return losses[-1], point_to_optimize.data.cpu().numpy(), losses


def test_radan_vs_adam():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    product_manifold, target_point_tensor = get_product_manifold_and_target(device_str)
    target_point_tensor = target_point_tensor.to(device_str)

    initial_point_tensor = torch.tensor([0.0, 0.5, 0.0, -0.5], dtype=torch.float32).to(device_str)
    num_iterations = 100
    learning_rate = 0.1
    tolerance = 1e-6

    loss_adam, point_adam, _ = optimize_and_compare(
        product_manifold,
        target_point_tensor,
        RiemannianAdam,
        {},
        initial_point_tensor.clone(),
        num_iterations=num_iterations,
        lr=learning_rate,
        tol=tolerance,
    )

    loss_radan, point_radan, _ = optimize_and_compare(
        product_manifold,
        target_point_tensor,
        RiemannianAdan,
        {"betas": [0.7, 0.999, 0.999]},
        initial_point_tensor.clone(),
        num_iterations=num_iterations,
        lr=learning_rate,
        tol=tolerance,
    )

    print("\n--- Comparison Results ---")
    print(f"Target Point:     {target_point_tensor.cpu().numpy()}")
    print(f"Initial Point:    {initial_point_tensor.cpu().numpy()}")
    print(f"Adam Final Point: {point_adam} | Final Loss: {loss_adam:.6f}")
    print(f"Adan Final Point: {point_radan} | Final Loss: {loss_radan:.6f}")
    final_loss_radam = objective_function(
        torch.from_numpy(point_adam), target_point_tensor.cpu(), product_manifold
    ).item()
    final_loss_radan = objective_function(
        torch.from_numpy(point_radan), target_point_tensor.cpu(), product_manifold
    ).item()

    assert final_loss_radam < 1e-3, "Adam did not converge close enough to the target"
    assert final_loss_radan < 1e-3, "Adan did not converge close enough to the target"
    print("\n✅ Optimization test passed: Both Adam and Adan reached the target with low loss.")
