from SRW.data.dataset import ImageDataset
import torch
import time


def encode_image(
    encoder=None,
    img_path="/home/s224286626/UNAUTHEMB_EXTR/test_imgs",
    message=None,
    limit=3000,
):
    """
    Encodes images with the given encoder and message.

    img_path : Path to the directory containing images.
    message : List of Tensors of shape (N, message_size) in [0, 1], where N is the number of images to encode.
    limit : Maximum number of images to process.
    """

    dataset = ImageDataset(
        image_paths=img_path, H=encoder.H, W=encoder.W, up_limit=limit
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    if len(dataloader) < limit:
        limit = len(dataloader)
        print(f"Reduced limit to {limit} due to dataset size.")

    encode_images = []
    orig_imgs = []
    start_time = time.time()
    for i, img in enumerate(dataloader):
        with torch.no_grad():
            if img.device != next(encoder.parameters()).device:
                img = img.to(next(encoder.parameters()).device)

            encoded_img = encoder(img, message[i])  # Encode the image with the message

            encode_images.append(
                (1 + encoded_img.cpu()) / 2
            )  # Move to CPU and keep in range [-1, 1]
            orig_imgs.append(
                (1 + img.cpu()) / 2
            )  # Keep original images in range [-1, 1]

    print(
        f"Encoding completed in {time.time() - start_time:.2f} seconds for {len(dataloader)} images at {len(dataloader)/ (time.time() - start_time):.2f} images/sec."
    )

    return encode_images, orig_imgs


def decode_wm(decoder, watermarked_images, bits=True, verbose=False):
    """
    watermarked_images : List of Tensors, each of shape (1, 3, H, W) in range [0, 1]
    output : List of Decoded message as a binary numpy array of shape (1, message_size)
    """
    decoded_messages = []
    with torch.no_grad():
        start_time = time.time()
        for watermarked_image in watermarked_images:
            if watermarked_image.device != next(decoder.parameters()).device:
                watermarked_image = watermarked_image.to(
                    next(decoder.parameters()).device
                )
            watermarked_image = torch.clip(watermarked_image, 0, 1)
            watermarked_image = 2 * watermarked_image - 1  # Scale to [-1, 1]
            decoded_message = decoder(watermarked_image)  # Decode the watermarked image
            if bits:
                decoded_message = (
                    decoded_message.gt(0).detach().cpu()
                )  # Threshold to get binary message
            decoded_messages.append(decoded_message)
        if verbose:
            print(
                f"Decoding completed in {time.time() - start_time:.2f} seconds for {len(watermarked_images)} images at {len(watermarked_images) / (time.time() - start_time):.2f} images/sec."
            )
    return decoded_messages


def overwrite_wm(overwriter, imgs, msg):
    re_encoded_imgs = []
    with torch.no_grad():
        for i, img in enumerate(imgs):
            img = img.clamp(0, 1)  # Ensure the image is in the range [0, 1]
            if img.device != next(overwriter.parameters()).device:
                img = img.to(next(overwriter.parameters()).device)
            img = 2 * img - 1  # Scale to [-1, 1]

            reencoded_img = overwriter(img, msg[i])

            reencoded_img = reencoded_img.clamp(-1, 1)
            re_encoded_imgs.append(
                (1 + reencoded_img.cpu()) / 2
            )

    return re_encoded_imgs
