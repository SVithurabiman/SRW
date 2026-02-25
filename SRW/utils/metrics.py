import torch
import kornia
from torchvision import transforms


def calculate_ssim_rgb(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate SSIM using Kornia.

    Args:
        img1, img2: PyTorch Tensors of shape (C, H, W)
                    with values in [0, 1].
    """
    # Ensure inputs are tensors
    if not torch.is_tensor(img1) or not torch.is_tensor(img2):
        raise TypeError("Inputs must be PyTorch tensors.")

    # Add Batch Dimension: (C, H, W) -> (1, C, H, W)
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    # Calculate SSIM Loss (which is 1 - SSIM)
    # window_size=11 is standard for SSIM
    ssim_score = kornia.metrics.ssim(img1, img2, window_size=11).mean()

    return ssim_score.item()


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    PSNR calculation for tensors with values in [0, 1].
    """
    if not torch.is_tensor(img1) or not torch.is_tensor(img2):
        raise TypeError("Inputs must be PyTorch tensors.")

    if img1.shape != img2.shape:
        raise ValueError("Input tensors must have the same shape.")

    # mse is kept as a tensor here for torch.log10 compatibility
    mse = torch.mean((img1 - img2) ** 2)

    if mse.item() == 0:
        return float("inf")

    data_range = 1.0

    # Use 10 * log10(MAX^2 / MSE)
    psnr = 10 * torch.log10((data_range**2) / mse).item()

    return psnr


def calculate_visual_metrics(
    original_imgs: list, watermarked_imgs: list
) -> (list, list):
    """
    original_imgs : List of PIL Images
    watermarked_imgs : List of PIL Images
    """
    psnrs = []
    ssims = []

    # Define the transform once
    to_tensor = transforms.ToTensor()

    for o, w in zip(original_imgs, watermarked_imgs):
        # Convert PIL -> Tensor (C, H, W) and scale to [0, 1]
        o_tensor = to_tensor(o)
        w_tensor = to_tensor(w)

        # Calculate metrics
        # Note: We pass the tensors, not the PIL images
        psnrs.append(calculate_psnr(o_tensor, w_tensor))
        ssims.append(calculate_ssim_rgb(o_tensor, w_tensor))

    return psnrs, ssims


def calculate_bitacc(watermark: list, decoded_watermark: list) -> list:
    """
    Calculate Bit Error Rate (BER) from a list of watermark and decode_watermark in bits
    """
    accs = []

    for orig, decoded in zip(watermark, decoded_watermark):
        orig = orig.flatten().to("cpu")
        decoded = decoded.flatten().to("cpu")

        if orig.shape != decoded.shape:
            raise ValueError(f"Shape mismatch: {orig.shape} vs {decoded.shape}")

        acc = (orig == decoded).float().mean().item()

        accs.append(acc)

    return accs
