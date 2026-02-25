import torch
import torch.nn.functional as F


def pgd_self_overwrite_attack(
    x_w, m_target, decoder, epsilon=0.04, alpha=0.007, num_iter=40
):
    """
    PGD-based adversarial attack to overwrite the watermark in x_w so that
    the decoder outputs m_target instead of the original message,
    while keeping perturbations within an L_inf ball of radius epsilon.

    Args:
      x_w       (torch.Tensor): watermarked image, shape (B, C, H, W), values in [0,1]
      m_target  (torch.Tensor): target message bits, shape (B, L), in {0,1}
      decoder   (callable):     the decoder D: X → logits of shape (B, L)
      epsilon   (float):        max per-pixel perturbation
      alpha     (float):        step size for each PGD iteration
      num_iter  (int):          number of PGD steps

    Returns:
      torch.Tensor: adversarial image x_adv with same shape as x_w
    """

    if x_w.device != next(decoder.parameters()).device:
        x_w = x_w.to(next(decoder.parameters()).device)
    device = x_w.device

    target = m_target.float().to(device)
    x_w = 2 * x_w - 1

    x_adv = x_w.clone().detach().to(device)

    for _ in range(num_iter):
        x_adv.requires_grad_(True)

        # Forward pass through decoder and compute BCE loss
        # print(x_adv.min(), x_adv.max())
        logits = decoder(x_adv)

        loss = F.binary_cross_entropy_with_logits(logits, target)

        # Backpropagate to get gradient w.r.t. the image
        grad = torch.autograd.grad(loss, x_adv)[0]

        # PGD update (L_inf):
        x_adv = x_adv - alpha * grad.sign()

        # Project back into the L_inf ball around x_w and valid pixel range
        delta = torch.clamp(x_adv - x_w, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_w + delta, min=-1.0, max=1.0).detach()

    return x_adv
