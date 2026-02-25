import torch.nn as nn
import torch.nn.functional as F
import lpips


class WatermarkLossWrapper:
    def __init__(self, cfg, device="cuda"):
        self.lpips_fn = lpips.LPIPS(net=cfg.loss.fidelity_loss.lpips_net).to(device)
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.lambda_lpips = cfg.loss.fidelity_loss.lpips
        self.lambda_mse = cfg.loss.fidelity_loss.mse
        self.device = device

    def __call__(
        self,
        *,
        original_img,
        watermarked_img,
        message,
        recovered_message,
        recovered_message_adv=None,
        recovered_message_re_emb=None,
        weights,
    ):
        """
        Returns a dictionary of individual loss terms.
        """

        losses = {}

        # --- Encoder / image fidelity losses ---
        losses["mse"] = F.mse_loss(watermarked_img, original_img)
        losses["lpips"] = self.lpips_fn(watermarked_img, original_img).mean()
        losses["encoder"] = (
            self.lambda_mse * losses["mse"] + self.lambda_lpips * losses["lpips"]
        )

        # --- Decoder loss (clean) ---
        losses["decoder"] = self.bce_fn(recovered_message, message)

        # --- Adversarial / overwrite losses ---
        if recovered_message_adv is not None:
            losses["adv"] = self.bce_fn(recovered_message_adv, message)

        if recovered_message_re_emb is not None:
            losses["re_emb"] = self.bce_fn(recovered_message_re_emb, message)

        total = 0.0
        for key, weight in weights.items():
            if key in losses:
                total += weight * losses[key]

        losses["total"] = total
        return losses
