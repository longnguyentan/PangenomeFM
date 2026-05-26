"""
src/models/losses.py

Focal Loss for hard-negative mining in link prediction.

Reference: Lin et al. 2017, "Focal Loss for Dense Object Detection"

Standard BCE treats all negatives equally.  Focal loss down-weights easy
negatives and up-weights hard ones:

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where p_t = p if y=1, else (1-p).

For strict closure — which IS hard negatives — focal loss directly
prioritises the signal we care about.

Defaults: γ=2.0, α=0.25  (standard from the paper).
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    def focal_bce_loss(
        probs: "torch.Tensor",
        labels: "torch.Tensor",
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: float = 1.0,
    ) -> "torch.Tensor":
        """
        Focal binary cross-entropy loss.

        Args:
            probs:      (B,) predicted probabilities in [0, 1]
            labels:     (B,) binary ground truth (0 or 1)
            gamma:      focusing parameter (higher = more focus on hard examples)
            alpha:      balance factor for positive class
            pos_weight: additional positive class weighting (from class imbalance)

        Returns:
            Scalar loss tensor.
        """
        eps = 1e-7
        probs = probs.clamp(eps, 1.0 - eps)

        # p_t = p if y=1, else 1-p
        p_t = labels * probs + (1.0 - labels) * (1.0 - probs)

        # alpha_t = alpha if y=1, else 1-alpha
        alpha_t = labels * alpha + (1.0 - labels) * (1.0 - alpha)

        # Focal modulating factor
        focal_weight = (1.0 - p_t) ** gamma

        # Additional positive class weight from imbalance
        class_weight = labels * pos_weight + (1.0 - labels) * 1.0

        # Cross-entropy term
        ce = -torch.log(p_t)

        loss = alpha_t * focal_weight * class_weight * ce
        return loss.mean()
