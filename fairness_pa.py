"""Prediction alignment (PA) for FairTMJ.

Source: Supplementary Methods, "Model architecture and loss functions".
    L_PA = mean over observed sex groups of ||mean(z_group) - mean(z_batch)||_2

Use raw classifier logits, without softmax or diagnosis-conditioned grouping.
The function returns the UNWEIGHTED loss. In the training loop, apply
``alpha_t * PA_WEIGHT * prediction_alignment_loss(logits, sex)`` exactly once,
where ``alpha_t = min(1.0, epoch_number / 10.0)`` and epoch_number starts at 1.
The manuscript specifies PA_WEIGHT = 0.2. The distance is the ordinary,
unsquared L2 norm.
"""

import torch


PA_WEIGHT = 0.2


def prediction_alignment_loss(logits, sensitive_attributes):
    """Return the mean sex-to-batch logit distance as a scalar tensor.

    Args:
        logits: Floating tensor [batch, classes]; FairTMJ uses three classes.
        sensitive_attributes: Binary sex codes (0/1), shaped [batch] or
            [batch, 1]. Either code may represent either sex consistently.

    Groups receive equal weight; the overall mean includes every sample.
    Both means remain in the computation graph. Only groups present in the
    batch are included. A single-sex batch has zero PA loss and gradient.
    No diagnosis labels or clinical covariates are needed.
    """
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have nonempty shape [batch, classes]")
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")

    sex = sensitive_attributes
    if sex.ndim == 2 and sex.shape[1] == 1:
        sex = sex.squeeze(1)
    if sex.ndim != 1 or sex.shape[0] != logits.shape[0]:
        raise ValueError("sex codes must have shape [batch] or [batch, 1]")
    sex = sex.to(device=logits.device)
    if not torch.all((sex == 0) | (sex == 1)):
        raise ValueError("sex codes must be binary values 0 or 1")

    overall_mean = logits.mean(dim=0)
    distances = [
        torch.linalg.vector_norm(logits[sex == group].mean(dim=0) - overall_mean, ord=2)
        for group in torch.unique(sex)
    ]
    return torch.stack(distances).mean()
