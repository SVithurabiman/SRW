import torch
import os
from SRW.utils.setup import setup_training, parse_args
from SRW.utils.config import load_config
from SRW.utils.helpers import TrainHelper
from SRW.utils.validation import validate
from SRW.attacks.pgd import pgd_self_overwrite_attack
from SRW.utils.config import load_config



def main():
    args = parse_args()
    cfg = load_config(
        data_cfg=args.dataset_config,
        exp_cfg=args.exp_config,
        config_type="Train",
    )
    setup = setup_training(cfg)

    loss_wrapper = setup.loss_wrapper
    weights = setup.weights.copy()
    best_val_loss = setup.best_val_loss
    ber_decoder = 1

    for epoch in range(setup.epochs):
        for batch_idx, img in enumerate(setup.train_dataloader):

            img = img.to(setup.device)
            global_step = epoch * len(setup.train_dataloader) + batch_idx

            message = (
                torch.randint(0, 2, (img.size(0), setup.message_size))
                .float()
                .to(setup.device)
            )
            adv_message = (
                torch.randint(0, 2, (img.size(0), setup.message_size))
                .float()
                .to(setup.device)
            )

            watermarked_image = setup.encoder(img, message)

            if ber_decoder < cfg.noise_layers.ber_threshold and global_step > 0:
                watermarked_image_pert = setup.perturbation_model(watermarked_image)
            else:
                watermarked_image_pert = watermarked_image

            overrtiten_image_re_emb = setup.encoder(watermarked_image_pert, adv_message)

            pgd_values = TrainHelper.get_curriculum_eps_alpha(
                global_step=global_step,
                warmup_step=cfg.pgd.warmup_step,
                ramp_step=setup.max_global_steps // 5,
                max_epsilon=float(cfg.pgd.max_epsilon),
                max_alpha=float(cfg.pgd.max_epsilon),
                min_alpha=float(cfg.pgd.min_alpha),
            )

            overwritten_image = pgd_self_overwrite_attack(
                x_w=(watermarked_image_pert + 1) / 2,
                m_target=adv_message,
                decoder=setup.decoder,
                epsilon=pgd_values["epsilon"],
                alpha=pgd_values["alpha"],
                num_iter=cfg.pgd.num_iter,
            )

            overwritten_image_pert = overwritten_image
            overrtiten_image_re_emb_pert = overrtiten_image_re_emb

            recovered_message = setup.decoder(watermarked_image_pert)
            recovered_message_re_emb = setup.decoder(overrtiten_image_re_emb_pert)
            recovered_message_ow = setup.decoder(overwritten_image_pert)
            try:
                losses = loss_wrapper(
                    original_img=img,
                    watermarked_img=watermarked_image,
                    message=message,
                    recovered_message=recovered_message,
                    recovered_message_adv=recovered_message_ow,
                    recovered_message_re_emb=recovered_message_re_emb,
                    weights=weights,
                )

                setup.optimizer_enc.zero_grad()
                setup.optimizer_dec.zero_grad()
                losses["total"].backward()
                setup.optimizer_enc.step()
                setup.optimizer_dec.step()

            except Exception as e:
                print(
                    f"Error during backward pass: {e} at step {global_step}, epoch {epoch}",
                    flush=True,
                )
                continue

            ber_decoder = TrainHelper.calculate_ber(message, recovered_message)
            ber_overwritten_orig = TrainHelper.calculate_ber(
                message, recovered_message_ow
            )
            ber_overwritten_re_emb = TrainHelper.calculate_ber(
                message, recovered_message_re_emb
            )

            prev_weights = weights.copy()
            weights = TrainHelper.adjust_weights(
                weights,
                global_step,
                ber_decoder,
                (ber_overwritten_orig + ber_overwritten_re_emb) / 2,
                prev_weights=prev_weights,
                smoothing=0.8,
                max_epochs=setup.max_global_steps,
                base_wights=setup.weights,
            )

            TrainHelper.log_scalars(
                writer=setup.writer, scalar_dict=losses, step=global_step, prefix="Loss"
            )

            TrainHelper.log_scalars(
                writer=setup.writer,
                scalar_dict=weights,
                step=global_step,
                prefix="Weights",
            )

            TrainHelper.log_scalars(
                writer=setup.writer,
                scalar_dict=pgd_values,
                step=global_step,
                prefix="PGD",
            )

            TrainHelper.log_scalars(
                writer=setup.writer,
                scalar_dict={
                    "decoder": ber_decoder,
                    "overwritenPGD-Orig": ber_overwritten_orig,
                    "overwriten-re-emb-Orig": ber_overwritten_re_emb,
                },
                step=global_step,
                prefix="BER",
            )

            if global_step % 1000 == 0:
                print("Validating at global step", global_step, flush=True)
                ber, losses = validate(
                    setup.encoder,
                    setup.decoder,
                    setup.val_dataloader,
                    setup.device,
                    weights,
                    cfg,
                )

                TrainHelper.log_scalars(
                    setup.writer, losses, global_step, prefix="Val/Loss"
                )

                TrainHelper.log_scalars(
                    setup.writer, ber, global_step, prefix="Val/BER"
                )

                if losses["TotalLoss"] < best_val_loss:
                    best_val_loss = losses["TotalLoss"]
                    torch.save(
                        setup.encoder.state_dict(),
                        f"{setup.results_dir}/{setup.experiment_name}/best_encoder.pth",
                    )
                    torch.save(
                        setup.decoder.state_dict(),
                        f"{setup.results_dir}/{setup.experiment_name}/best_decoder.pth",
                    )
                    print(
                        f"New best model for experiment name {setup.experiment_name} saved at global step {global_step} with loss {losses['TotalLoss']:.4f}"
                    )

                torch.save(
                    setup.encoder.state_dict(),
                    f"{setup.results_dir}/{setup.experiment_name}/encoder_epoch_{epoch}_step_{global_step}.pth",
                )
                torch.save(
                    setup.decoder.state_dict(),
                    f"{setup.results_dir}/{setup.experiment_name}/decoder_epoch_{epoch}_step_{global_step}.pth",
                )

            if global_step % 200 == 0:
                TrainHelper.log_image_group(
                    setup.writer,
                    "Images",
                    {
                        "original": img,
                        "watermarked": watermarked_image,
                        "overwritten": overwritten_image,
                        "watermarked_perturbed": watermarked_image_pert,
                        "overwritten_perturbed": overwritten_image_pert,
                        "overwritten_re_emb": overrtiten_image_re_emb,
                    },
                    global_step,
                )


if __name__ == "__main__":
    main()
