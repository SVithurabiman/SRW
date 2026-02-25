import torch
import numpy as np
from torchvision import transforms
import torch.nn.utils as nn_utils
import torchvision.utils as vutils
import os

class Helper:
    pass


class EvalHelper(Helper):
    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)

    @staticmethod
    def load_model(model, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def load_checkpoints(encoder,decoder,dir,device):
        encoder = EvalHelper.load_model(encoder,os.path.join(dir,"encoder.pth"),device=device)
        decoder = EvalHelper.load_model(decoder,os.path.join(dir,"decoder.pth"),device=device)
        
        return encoder, decoder
        
    @staticmethod
    def tensor2PIL(imgs):
        """
        Convert a list of tensors in range [0,1] to a list of PIL images
        imgs : List of Tensors of shape (1, 3, H, W) in range [0, 1]
        output : List of PIL Images
        """
        pil_images = []

        for img in imgs:
            img = img.clamp(0, 1)  # Ensure the image is in the range [0, 1]
            pil_img = transforms.ToPILImage()(img.squeeze(0).cpu())
            pil_images.append(pil_img)
        return pil_images

    @staticmethod
    def save_images(images, path):
        """
        Save a list of PIL images to the specified path.
        images : List of PIL Images
        path : str, directory to save images
        """
        import os

        if not os.path.exists(path):
            os.makedirs(path)

        for i, img in enumerate(images):
            img.save(os.path.join(path, f"Image_{i+1}.png"))


class TrainHelper(Helper):

    @staticmethod
    def adjust_weights(
        weights,
        epoch,
        ber_decoder,
        ber_decoder_org_rec_after_overwritten,
        prev_weights=None,
        smoothing=0.8,
        max_epochs=100,
        base_wights =None
    ):
        """
        Adjusts training weights based on clean decoding, overwrite robustness, and visual quality.
        All three metrics are minimized, so confidence = 1 - normalized(metric).
        """

        decoder_confidence = 1.0 - min(ber_decoder / 0.2, 1.0)
        overwrite_recovery_confidence = 1.0 - min(
            ber_decoder_org_rec_after_overwritten / 0.2, 1.0
        )

        transition_readiness = (
            decoder_confidence + overwrite_recovery_confidence
        ) / 2.0  

        # Blend with time-based ramp
        epoch_progress = min((epoch + 1) / max_epochs, 1.0)
        alpha = 0.7 * epoch_progress + 0.3 * transition_readiness

        # Define weight targets
        target_weights = {
            "encoder": base_wights["encoder"] + 2.5 * alpha,
            "decoder": base_wights["decoder"] - 1.5 * alpha,  # 6.0 - 1.5 * alpha,
            "adv": base_wights["adv"] - 1.0 * alpha,  # 5 - 1.5 * alpha,
            "re_emb": base_wights["re_emb"] - 1.0*alpha
            #'overwriter-quality': 0, #'overwriter-quality': 2.0 + 4.5 * alpha,
        }

        # target_weights = {
        #     'encoder': 5.0 + 2.5 * alpha,
        #     'decoder': 4.0 - 3.5 * alpha,
        #     'overwriter-rec': 5 - 1.5 * alpha,
        #     #'overwriter-quality': 0, #'overwriter-quality': 2.0 + 4.5 * alpha,
        # }

        # Apply smoothing
        if prev_weights is None:
            prev_weights = target_weights.copy()

        smoothed_weights = {
            k: smoothing * prev_weights.get(k, target_weights[k])
            + (1 - smoothing) * target_weights[k]
            for k in target_weights
        }
        return smoothed_weights

    @staticmethod
    def remove_spectral_norm_from_model(model):
        """
        Recursively removes spectral normalization from all layers in the model.

        Args:
            model (nn.Module): The model instance to process.
        """
        for name, module in model.named_children():
            # If module has spectral norm applied, remove it
            if isinstance(module, torch.nn.Conv2d) or isinstance(
                module, torch.nn.Linear
            ):
                try:
                    nn_utils.remove_spectral_norm(module)
                    print(f"Removed spectral norm from {name}", flush=True)
                except ValueError:
                    pass  # spectral norm was not applied

            # Recurse into children
            TrainHelper.remove_spectral_norm_from_model(module)

    @staticmethod
    def check_spectral_norm(model):
        """
        Check if the model has spectral normalization applied.
        """
        for name, module in model.named_modules():
            if hasattr(module, "weight_u"):
                print(f"{name} has spectral norm", flush=True)
                return True
        print("No spectral norm found in the model", flush=True)
        return False

    @staticmethod
    def contract_spectral_norm(model, alpha=0.95):
        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, "weight_g"):
                    module.weight_g.data.mul_(alpha)

    @staticmethod
    def get_curriculum_eps_alpha(
        global_step,
        warmup_step=0,
        ramp_step=30,
        max_epsilon=0.05,
        max_alpha=0.009,
        min_alpha=1e-4,
    ):
        if global_step < warmup_step:
            epsilon = 0.0
            alpha = min_alpha  # Small stable value
        elif global_step < ramp_step:
            progress = (global_step - warmup_step) / (ramp_step - warmup_step)
            epsilon = progress * max_epsilon
            alpha = max(progress * max_alpha, 1e-4)
        else:
            epsilon = max_epsilon
            alpha = max_alpha

        return {"epsilon": epsilon, "alpha": alpha}

    @staticmethod
    def check_gradient_norm(models):
        for model in models:
            for name, param in model.named_parameters():
                if "bias" in name:  # skip biases
                    continue
                if param.grad is None:
                    print(
                        f"{model.__class__.__name__} layer: {name} has no gradient",
                        flush=True,
                    )
                else:
                    grad_norm = param.grad.norm().item()
                    if grad_norm < 1e-5:
                        print(
                            f"{model.__class__.__name__} layer: {name}, Gradient norm: {grad_norm}",
                            flush=True,
                        )

    @staticmethod
    def calculate_ber(message: torch.Tensor, extracted_message: torch.Tensor):
        message = message.clone().detach().cpu().numpy()
        extracted_message = extracted_message.clone().detach().cpu().numpy()

        extracted_message[extracted_message < 0] = 0
        extracted_message[extracted_message > 0] = 1

        ber = np.mean(message != extracted_message)

        return ber

    @staticmethod
    def log_scalars(writer, scalar_dict, step, prefix=None):
        """
        Log multiple scalars to TensorBoard.

        Args:
            writer (SummaryWriter): TensorBoard writer
            scalar_dict (dict): {name: value}
            step (int): global step
            prefix (str, optional): namespace prefix
        """
        if writer is None:
            return

        for key, value in scalar_dict.items():
            tag = f"{prefix}/{key}" if prefix else key
            writer.add_scalar(tag, value, step)

    @staticmethod
    def log_images(writer, tag, images, step, max_images=2, normalize=True):
        if writer is None or images is None:
            return
        images = images[:max_images].detach()
        img_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)
        writer.add_image(tag, img_grid, step)

    @staticmethod
    def log_image_group(
        writer, prefix, images_dict, step, max_images=2, normalize=True
    ):
        """
        Log multiple image tensors under a common TensorBoard prefix.

        Args:
            images_dict (dict): {name: image_tensor}
        """
        if writer is None:
            return

        for name, images in images_dict.items():
            if images is None:
                continue

            TrainHelper.log_images(
                writer=writer,
                tag=f"{prefix}/{name}",
                images=images,
                step=step,
                max_images=max_images,
                normalize=normalize,
            )
