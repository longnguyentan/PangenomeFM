"""
src/models/heads.py

Task-specific heads that plug into DualStreamPangenomeGAT node embeddings.

Currently provides:
    * NodeClsHead             - multi-class softmax head (cCRE, ATAC class, etc.)
    * multiclass_focal_loss() - focal variant of CE for imbalanced multi-class

The backbone (DualStreamPangenomeGAT.encode_nodes) returns h of shape
(N, hidden_dim). These heads consume h directly.
"""
from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class NodeClsHead(nn.Module):
        """
        Multi-class MLP head for node classification.

        Architecture:
            h (N, hidden_dim) -> [Linear -> LayerNorm -> ELU -> Dropout] x 2
                              -> Linear(n_classes)  -> logits

        Keeps the predictor explicit and compact so it doesn't overshadow the
        backbone. Roughly mirrors the capacity of `edge_predictor` in the
        existing DualStreamPangenomeGAT for parity with the link-pred head.
        """

        def __init__(
            self,
            hidden_dim: int,
            n_classes: int,
            mlp_dim: Optional[int] = None,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            if mlp_dim is None:
                mlp_dim = 2 * hidden_dim
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, mlp_dim),
                nn.LayerNorm(mlp_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, mlp_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim // 2, n_classes),
            )
            # Initialize last layer small so starting logits are near uniform
            nn.init.normal_(self.mlp[-1].weight, std=0.01)
            nn.init.zeros_(self.mlp[-1].bias)

        def forward(self, h: "torch.Tensor") -> "torch.Tensor":
            """h: (N, hidden_dim) -> logits: (N, n_classes)"""
            return self.mlp(h)

    # ------------------------------------------------------------------
    # Multi-class focal loss
    # ------------------------------------------------------------------

    def multiclass_focal_loss(
        logits: "torch.Tensor",  # (B, C)
        targets: "torch.Tensor",  # (B,) int64
        gamma: float = 2.0,
        class_weights: Optional["torch.Tensor"] = None,  # (C,) float32
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> "torch.Tensor":
        """
        Multi-class focal loss with optional per-class weights and label smoothing.

            FL = -class_weight[y] * (1 - p_y)^gamma * log(p_y)

        This is the natural multi-class generalization of Lin et al. 2017,
        and complements models.losses.focal_bce_loss which is binary only.

        Args:
            logits         : raw model outputs (B, C)
            targets        : int64 class indices, (B,)
            gamma          : focusing parameter; 0 -> plain CE
            class_weights  : optional tensor of length C to reweight classes
                             (e.g. inverse frequency)
            label_smoothing: 0..1 smoothing applied to one-hot targets
            ignore_index   : targets equal to this value are ignored
                             (useful for UNK on alt-haplotype nodes)

        Returns scalar mean loss over non-ignored samples.
        """
        valid = targets != ignore_index
        if valid.sum() == 0:
            return logits.sum() * 0.0  # graph-safe zero
        logits = logits[valid]
        targets = targets[valid]

        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
        probs = log_probs.exp()  # (B, C)

        # Gather p_t = prob of the true class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - p_t).clamp(min=1e-7) ** gamma

        if class_weights is not None:
            w = class_weights[targets]
        else:
            w = torch.ones_like(p_t)

        ce_term = -log_p_t
        if label_smoothing > 0.0:
            # LS: interpolate between one-hot and uniform
            smooth_loss = -log_probs.mean(dim=-1)
            ce_term = (1.0 - label_smoothing) * ce_term + label_smoothing * smooth_loss

        loss = (w * focal_factor * ce_term).mean()
        return loss
