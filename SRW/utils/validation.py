import torch
from SRW.utils.helpers import TrainHelper
from SRW.attacks.pgd import pgd_self_overwrite_attack
from SRW.utils.losses import WatermarkLossWrapper


def validate(encoder, decoder, val_dataloader, device, weights, cfg):
    encoder.eval()
    decoder.eval()
    total_bits = 0
    decoder_errors = 0
    overwriter_errors = 0
    overwriter_pgd_error = 0
    total_mse = 0.0
    total_loss = 0.0
    total_samples = 0
    message_size = cfg.data.message_size
    val_loss_wrapper = WatermarkLossWrapper(cfg=cfg)

    for img in val_dataloader:
        img = img.to(device)
        batch_size = img.size(0)
        message = torch.randint(0, 2, (img.size(0), message_size)).float().to(device)
        adv_message = (
            torch.randint(0, 2, (img.size(0), message_size)).float().to(device)
        )

        with torch.no_grad():
            wm_image = encoder(img, message)
            recovered_message = decoder(wm_image)
            overwritten_image_re_emb = encoder(wm_image, adv_message)
            recovered_message_ow_re_emb = decoder(overwritten_image_re_emb)

        overwriiten_image = pgd_self_overwrite_attack(
            x_w=(wm_image + 1) / 2,
            m_target=adv_message,
            decoder=decoder,
            epsilon=cfg.val.pgd.epsilon,
            alpha=cfg.val.pgd.alpha,
            num_iter=cfg.val.pgd.num_iter,
        )

        recovered_message_ow = decoder(overwriiten_image)
        losses = val_loss_wrapper(
            original_img=img,
            watermarked_img=wm_image,
            message=message,
            recovered_message=recovered_message,
            recovered_message_adv=recovered_message_ow,
            recovered_message_re_emb=recovered_message_ow_re_emb,
            weights=weights,
        )

        total_mse += losses["mse"].item() * batch_size
        total_loss += losses["total"].item() * batch_size
        total_samples += batch_size
        decoder_errors += (
            TrainHelper.calculate_ber(message, recovered_message)
            * batch_size
            * message_size
        )
        overwriter_errors += (
            TrainHelper.calculate_ber(message, recovered_message_ow_re_emb)
            * batch_size
            * message_size
        )
        overwriter_pgd_error += (
            TrainHelper.calculate_ber(message, recovered_message_ow)
            * batch_size
            * message_size
        )

        total_bits += batch_size * message_size

    avg_mse = total_mse / total_samples
    avg_total = total_loss / total_samples
    ber_decoder = decoder_errors / total_bits
    ber_overwriter = overwriter_errors / total_bits
    ber_overwriter_pgd = overwriter_pgd_error / total_bits

    encoder.train()
    decoder.train()

    ber = {
        "Decoder": ber_decoder,
        "Overwritten": ber_overwriter,
        "PGD": ber_overwriter_pgd,
    }

    losses = {"MSE": avg_mse, "TotalLoss": avg_total}
    return ber, losses
