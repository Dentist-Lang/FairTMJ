"""Adversarial alignment (AA) for FairTMJ.

Source: Supplementary Methods, "Model architecture and loss functions".
A sex classifier receives hidden features through gradient reversal (GRL).
The default adversary uses a 128-unit ReLU hidden layer and mean
cross-entropy to predict binary sex from the fused MRI representation.

The loss is UNWEIGHTED. In the training loop, apply
``alpha_t * AA_WEIGHT * adversarial_alignment_loss(features, sex, adversary)``
exactly once. Use alpha_t = min(1.0, epoch_number / 10.0), counting from 1.
The manuscript specifies AA_WEIGHT = 0.1 and GRL_BETA = 1.0.
Include both diagnosis-model and adversary parameters in the optimizer.
The adversary minimizes sex cross-entropy; the GRL reverses its gradient to
the feature extractor. Do not negate the returned loss again or detach h.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


AA_WEIGHT = 0.1
GRL_BETA = 1.0


class GradientReversalLayer(Function):
    """Identity in the forward pass; multiply feature gradients by -beta."""

    @staticmethod
    def forward(ctx, features, beta):
        ctx.beta = beta
        return features.view_as(features)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.beta * grad_output, None


def grad_reverse(features, beta=GRL_BETA):
    return GradientReversalLayer.apply(features, beta)


class Adversary(nn.Module):
    """Binary sex classifier; ResNet50 uses 4096-dimensional fused features.

    Set input_dim to the fused feature size when using a different backbone.
    """

    def __init__(self, input_dim=2048 * 2, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, feature_representation, beta=GRL_BETA):
        return self.network(grad_reverse(feature_representation, beta))


def adversarial_alignment_loss(
    feature_representation, sensitive_attributes, adversary_model, beta=GRL_BETA
):
    """Return unweighted mean sex cross-entropy with reversed feature gradients.

    Features have shape [batch, feature_dim]; binary sex codes have shape
    [batch] or [batch, 1]. Place the adversary on the same device as features.
    Labels are moved to that device, supporting CPU and accelerators without
    a hard-coded CUDA call. A single-sex batch still trains the adversary.
    """
    if feature_representation.ndim != 2 or feature_representation.shape[0] == 0:
        raise ValueError("features must have nonempty shape [batch, feature_dim]")
    sex = sensitive_attributes
    if sex.ndim == 2 and sex.shape[1] == 1:
        sex = sex.squeeze(1)
    if sex.ndim != 1 or sex.shape[0] != feature_representation.shape[0]:
        raise ValueError("sex codes must have shape [batch] or [batch, 1]")
    sex = sex.to(device=feature_representation.device)
    if not torch.all((sex == 0) | (sex == 1)):
        raise ValueError("sex codes must be binary values 0 or 1")

    sex_logits = adversary_model(feature_representation, beta=beta)
    return F.cross_entropy(sex_logits, sex.long(), reduction="mean")
