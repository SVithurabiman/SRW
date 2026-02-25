import os
import sys
import torch
import numpy as np
from SRW.utils.watermark import encode_image, decode_wm, overwrite_wm
from SRW.utils.config import load_config
from SRW.utils.helpers import EvalHelper
from SRW.utils.metrics import calculate_visual_metrics, calculate_bitacc
from SRW.utils.setup import setup_testing, parse_args
from SRW.attacks.pgd import pgd_self_overwrite_attack


def main():

    args = parse_args()
    cfg = load_config(
        data_cfg=args.dataset_config,
        exp_cfg=args.exp_config,
        config_type="Test",
    )
    setup = setup_testing(cfg)
    encoded_img, orig_imgs = encode_image(
        setup.encoder,
        cfg.dataset_path,
        setup.messages,
        limit=cfg.limit,
    )
    decoded_message = decode_wm(setup.decoder, encoded_img)
    encode_image_pil = EvalHelper.tensor2PIL(encoded_img)

    reencoded_img = overwrite_wm(setup.encoder, encoded_img, setup.adv_messages)

    decoded_message_adv = decode_wm(setup.decoder, reencoded_img)

    reencoded_img_pil = EvalHelper.tensor2PIL(reencoded_img)
    orig_imgs_pil = EvalHelper.tensor2PIL(orig_imgs)

    bit_acc = calculate_bitacc(decoded_message, setup.messages)
    bit_acc_adv = calculate_bitacc(decoded_message_adv, setup.adv_messages)
    bit_acc_rec = calculate_bitacc(decoded_message_adv, setup.messages)

    print(
        f"Bit accuracy for original message:  {np.mean(bit_acc):.2f} ± {np.std(bit_acc):.2f}"
    )
    print(
        f"Bit accuracy for recovered &  adversarial message:  {np.mean(bit_acc_adv):.2f} ± {np.std(bit_acc_adv):.2f}"
    )
    print(
        f"Bit accuracy for recovered  & Original message:  {np.mean(bit_acc_rec):.2f} ± {np.std(bit_acc_rec):.2f}"
    )

    psnrs_orig_enc, ssims_orig_enc = calculate_visual_metrics(
        orig_imgs_pil, encode_image_pil
    )
    psnrs_orig_reenc, ssim_orig_reenc = calculate_visual_metrics(
        orig_imgs_pil, reencoded_img_pil
    )

    print(
        f"Mean PSNR and SSIM between Original and Watermarked Images: , {np.mean(psnrs_orig_enc):.2f} ± {np.std(psnrs_orig_enc):.2f}, {np.mean(ssims_orig_enc):.4f} ± {np.std(ssims_orig_enc):.4f}"
    )
    print(
        f"Mean PSNR and SSIM between Original and Re-Watermarked Images: , {np.mean(psnrs_orig_reenc):.2f} ± {np.std(psnrs_orig_reenc):.2f}, {np.mean(ssim_orig_reenc):.4f} ± {np.std(ssim_orig_reenc):.4f}"
    )

    if cfg.output_path:
        EvalHelper.save_images(encode_image_pil, cfg.output_path)

    if cfg.pgd.attack:
        print ("Performing PGD Attack..")
        assert len(cfg.pgd.epsilon) == len(cfg.pgd.alpha), "Epsilon and alpha lists must be of the same length. Check the configs/test.yaml."
        assert len(cfg.pgd.epsilon) == len(cfg.pgd.num_iter), "Epsilon and num_iter lists must be of the same length. Check the configs/test.yaml."

        bit_acc_adv_dict={}
        bit_accs_org_adv_pgd_dict={}
        for eps,alph,num_iter in zip(cfg.pgd.epsilon,cfg.pgd.alpha,cfg.pgd.num_iter):
            bit_accs_adv= []
            bit_accs_org_adv_pgd = []
            adv_imgs = []
            print(f"Attacking with epsilon: {eps}, alpha:{alph} and num_iter: {num_iter}")
            for i in range(len(encoded_img[:50])):
                adv_img = pgd_self_overwrite_attack(
                                                    encoded_img[i], 
                                                    setup.adv_messages[i], 
                                                    setup.decoder, 
                                                    epsilon=eps, 
                                                    alpha=alph, 
                                                    num_iter=num_iter
                                                    )
                adv_imgs.append((adv_img+1)/2)
                if i%100==0:
                    print(f"Processed {i+1} images for PGD attack")
            print(f"Completed{eps}")
            decoded_adv_message = decode_wm(setup.decoder, adv_imgs)

            bit_accs_adv.append(calculate_bitacc(decoded_adv_message, setup.adv_messages))
            bit_accs_org_adv_pgd.append(calculate_bitacc(decoded_adv_message, setup.messages))

            print(f"Bit accuracy for adv message after PGD attack: {np.mean(bit_accs_adv):.4f} ± {np.std(bit_accs_adv):.4f}")
            print(f"Bit accuracy for original message after PGD attack: {np.mean(bit_accs_org_adv_pgd):.4f} ± {np.std(bit_accs_org_adv_pgd):.2f}")


if __name__ == "__main__":
    main()
