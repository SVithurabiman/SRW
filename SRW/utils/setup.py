from collections import namedtuple
import os
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from SRW.models.architectures import (
    DeeperSpectralUNetEncoderwithFA_ResNet50,
    MessageDecoder,
)
from SRW.pertubation import noise_model
from SRW.data.dataset import ImageDataset
from SRW.utils.losses import WatermarkLossWrapper
from SRW.utils.helpers import EvalHelper

TrainingSetup = namedtuple(
    "TrainingSetup",
    [
        "device",
        "encoder",
        "decoder",
        "perturbation_model",
        "train_dataloader",
        "val_dataloader",
        "optimizer_enc",
        "optimizer_dec",
        "writer",
        "loss_wrapper",
        "weights",
        "max_global_steps",
        "batch_size",
        "message_size",
        "experiment_name",
        "best_val_loss",
        "prev_weights",
        "ber_decoder",
        "global_step",
        "epochs",
        "results_dir",
    ],
)
TestingSetup = namedtuple(
    "TestingSetup",
    ["device", "encoder", "decoder", "messages", "adv_messages"],
)


def parse_args():
    parser = argparse.ArgumentParser(description="SRW model")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="/configs/dataset.yaml",
        help="Path to dataset config YAML file (e.g., configs/dataset.yaml)",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="/configs/train.yaml",
        help="Path to experiment config YAML file (e.g., configs/train.yaml or configs/test.yaml )",
    )
    return parser.parse_args()


def setup_training(cfg):
    print("Setting up training environment...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths and parameters
    train_data_dir = cfg.train_data_dir
    val_data_dir = cfg.val_data_dir
    H, W = cfg.data.H, cfg.data.W
    batch_size = cfg.batch_size
    message_size = cfg.data.message_size
    experiment_name = cfg.experiment_name
    results_dir = cfg.results_dir
    os.makedirs(f"{results_dir}/{experiment_name}", exist_ok=True)

    # Models
    encoder = DeeperSpectralUNetEncoderwithFA_ResNet50(
        img_channels=cfg.common.img_channels,
        message_channels=cfg.encoder.message_channels,
        message_dim=message_size,
        H=H,
        W=W,
    ).to(device)

    decoder = MessageDecoder(
        input_channels=cfg.common.img_channels, H=H, W=W, message_dim=message_size
    ).to(device)

    encoder.train()
    decoder.train()

    # Perturbation / noise model
    perturbation_model = noise_model.NoiseModel(device=device)

    # Datasets
    train_dataset = ImageDataset(train_data_dir, H, W, low_limit=0, up_limit=20000)
    val_dataset = ImageDataset(val_data_dir, H, W, low_limit=0, up_limit=1000)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Optimizers
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=cfg.encoder.lr)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=cfg.decoder.lr)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"{results_dir}/{experiment_name}/runs")

    # Loss wrapper
    loss_wrapper = WatermarkLossWrapper(cfg, device=device)

    # Initial weights
    weights = {
        "encoder": float(cfg.loss.weights.encoder),
        "decoder": float(cfg.loss.weights.decoder),
        "adv": float(cfg.loss.weights.adv),
        "re_emb": float(cfg.loss.weights.adv),
    }
    print("Training environment setup complete.", flush=True)
    return TrainingSetup(
        device=device,
        encoder=encoder,
        decoder=decoder,
        perturbation_model=perturbation_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer_enc=optimizer_enc,
        optimizer_dec=optimizer_dec,
        writer=writer,
        loss_wrapper=loss_wrapper,
        weights=weights,
        max_global_steps=cfg.epochs * len(train_loader),
        batch_size=batch_size,
        message_size=message_size,
        experiment_name=experiment_name,
        best_val_loss=float("inf"),
        prev_weights=None,
        ber_decoder=1,
        global_step=0,
        epochs=cfg.epochs,
        results_dir=results_dir,
    )


def setup_testing(cfg):

    print("Setting up testing environment...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = DeeperSpectralUNetEncoderwithFA_ResNet50(
        img_channels=cfg.model.common.img_channels,
        message_channels=cfg.model.encoder.message_channels,
        message_dim=cfg.data.message_size,
        H=cfg.data.H,
        W=cfg.data.W,
        enhance=cfg.model.encoder.enhance,
        kernel_size=cfg.model.encoder.kernel_size,
        sigma=cfg.model.encoder.sigma,
        s_f=cfg.model.encoder.s_f,
        threshold=cfg.model.encoder.threshold,
    )
    decoder = MessageDecoder(
        input_channels=cfg.model.common.img_channels,
        H=cfg.data.H,
        W=cfg.data.W,
        message_dim=cfg.data.message_size,
    )
    encoder, decoder = EvalHelper.load_checkpoints(encoder, decoder, cfg.path, device)
    EvalHelper.set_seed(cfg.seed)
    messages = [
        torch.randint(0, 2, (1, cfg.data.message_size)).float().to(device)
        for _ in range(cfg.limit)
    ]
    adv_messages = [
        torch.randint(0, 2, (1, cfg.data.message_size)).float().to(device)
        for _ in range(cfg.limit)
    ]

    return TestingSetup(
        device=device,
        encoder=encoder,
        decoder=decoder,
        messages=messages,
        adv_messages=adv_messages,
    )
