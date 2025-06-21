"""Riemannian Adan (Radan).

Radan is the Riemannian version of the Adaptive Nesterov Momentum algorithm.
This code is compatible with both Geoopt and Manify libraries, and is designed for Riemannian Fuzzy K-Means.
We recommend using the parameters `[0.7, 0.99, 0.99]` for best performance.**

For more details on the Radan algorithm, please refer to:
https://openreview.net/forum?id=9VmOgMN4Ie

If you find this work useful, please cite the paper as follows:

bibtex
@article{Yuan2025,
  title={Riemannian Fuzzy K-Means},
  author={Anonymous},
  journal={OpenReview},
  year={2025},
  url={https://openreview.net/forum?id=9VmOgMN4Ie}
}

If you're interested in Adan, you can see:

bibtex
@ARTICLE{10586270,
  author={Xie, Xingyu and Zhou, Pan and Li, Huan and Lin, Zhouchen and Yan, Shuicheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models},
  year={2024},
  volume={46},
  number={12},
  pages={9508-9520},
  keywords={Training;Convergence;Complexity theory;Deep learning;Computer architecture;Task analysis;Stochastic processes;Adaptive optimizer;fast DNN training;DNN optimizer},
  doi={10.1109/TPAMI.2024.3423382}
}

If you have questions about the code, feel free to contact: yuanjinghuiiii@gmail.com.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from geoopt import ManifoldParameter, ManifoldTensor
from geoopt.optim.mixin import OptimMixin

if TYPE_CHECKING:
    from beartype.typing import Any, Callable
    from jaxtyping import Float

from . import _adan


class RiemannianAdan(OptimMixin, _adan.Adan):
    """Riemannian Adan with the same API as :class:`adan.Adan`.

    Attributes:
        param_groups: iterable of parameter groups, each containing parameters to optimize and optimization options
        _default_manifold: the default manifold used for optimization if not specified in parameters

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing (default: (0.98, 0.92, 0.99))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
    """

    def step(self, closure: Callable | None = None) -> Float[torch.Tensor, ""] | None:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                stablilize = False
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError("RiemannianAdan does not support sparse gradients")

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        # new param
                        state["exp_avg_diff"] = torch.zeros_like(point)
                        # last step grad
                        state["last_grad"] = torch.zeros_like(point)

                    state["step"] += 1
                    # make local variables for easy access
                    exp_avg = state["exp_avg"]
                    exp_avg_diff = state["exp_avg_diff"]
                    exp_avg_sq = state["exp_avg_sq"]
                    last_grad = state["last_grad"]
                    # actual step

                    grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(point, grad)
                    # grad_last_diff
                    grad_last_diff = grad - last_grad
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    # grad_last_diff
                    exp_avg_diff.mul_(betas[1]).add_(grad_last_diff, alpha=1 - betas[1])
                    # z_t
                    zt = grad_last_diff.mul(betas[1]).add_(grad)
                    # z_t^2
                    exp_avg_sq.mul_(betas[2]).add_(manifold.component_inner(point, zt), alpha=1 - betas[2])
                    bias_correction1 = 1 - betas[0] ** state["step"]
                    bias_correction2 = 1 - betas[1] ** state["step"]
                    bias_correction3 = 1 - betas[2] ** state["step"]

                    denom = exp_avg_sq.div(bias_correction3).sqrt_()

                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = (
                        (exp_avg.div(bias_correction1)).add_((exp_avg_diff.div(bias_correction2)), alpha=betas[1])
                    ) / denom.add_(eps)

                    # transport the exponential averaging to the new point
                    new_point, exp_avg_new = manifold.retr_transp(point, -learning_rate * direction, exp_avg)

                    last_grad.copy_(manifold.transp(point, new_point, grad))
                    # transport v_t
                    exp_avg_diff.copy_(manifold.transp(point, new_point, exp_avg_diff))
                    exp_avg.copy_(exp_avg_new)
                    point.copy_(new_point)

                    if group["stabilize"] is not None and state["step"] % group["stabilize"] == 0:
                        stablilize = True
                if stablilize:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()  # type: ignore
    def stabilize_group(self, group: dict[str, Any]) -> None:
        """Stabilizes the parameters in the group by projecting them onto their respective manifolds.

        Args:
            group: A dictionary containing the parameters and their states.

        Returns:
            None
        """
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            exp_avg = state["exp_avg"]
            exp_avg_diff = state["exp_avg_diff"]
            last_grad = state["last_grad"]
            p.copy_(manifold.projx(p))
            exp_avg.copy_(manifold.proju(p, exp_avg))
            exp_avg_diff.copy_(manifold.proju(p, exp_avg_diff))
            last_grad.copy_(manifold.proju(p, last_grad))
