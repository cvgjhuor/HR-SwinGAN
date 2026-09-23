import os
import sys
import csv
import yaml
import random
import logging
import argparse
from types import SimpleNamespace
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Networks import WNet, PatchGAN
from utils.dataset import IXIdataset
from utils.metrics import psnr as compute_psnr, ssim as compute_ssim, nmse as compute_nmse
from loss import netLoss, set_grad


def set_seed(seed: int = 42):
    """Set deterministic random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def gradient_penalty(D: nn.Module, real_samples: torch.Tensor, fake_samples: torch.Tensor, device: str = 'cuda'):
    """Calculate WGAN-GP gradient penalty."""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake_grad_output = torch.ones_like(d_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_grad_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


class MetricsLogger:
    """Logs training and validation statistics to CSV files."""
    def __init__(self, log_dir: str):
        """Initialize logger with output directory and create CSV files."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.train_csv = os.path.join(log_dir, "train_metrics.csv")
        self.val_csv = os.path.join(log_dir, "val_metrics.csv")
        self._init_headers()

    def _init_headers(self):
        """Initialize CSV column headers if files do not exist."""
        if not os.path.exists(self.train_csv):
            with open(self.train_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'g_loss', 'd_loss', 'im_l2', 'im_l1', 'kspace_l2', 'adv_loss', 'ffl_loss', 'lr'])
        if not os.path.exists(self.val_csv):
            with open(self.val_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'epoch', 'val_loss', 'im_l2', 'im_l1', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std'])

    def log_train(self, epoch: int, metrics: dict):
        """Append training metrics for current epoch to CSV."""
        with open(self.train_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epoch,
                f"{metrics.get('g_loss', 0.0):.6f}",
                f"{metrics.get('d_loss', 0.0):.6f}",
                f"{metrics.get('im_l2', 0.0):.6f}",
                f"{metrics.get('im_l1', 0.0):.6f}",
                f"{metrics.get('kspace_l2', 0.0):.6f}",
                f"{metrics.get('adv_loss', 0.0):.6f}",
                f"{metrics.get('ffl_loss', 0.0):.6f}",
                f"{metrics.get('lr', 0.0):.8f}"
            ])

    def log_val(self, epoch: int, metrics: dict):
        """Append validation metrics for current epoch to CSV."""
        with open(self.val_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epoch,
                f"{metrics.get('val_loss', 0.0):.6f}",
                f"{metrics.get('im_l2', 0.0):.6f}",
                f"{metrics.get('im_l1', 0.0):.6f}",
                f"{metrics.get('psnr_mean', 0.0):.4f}",
                f"{metrics.get('psnr_std', 0.0):.4f}",
                f"{metrics.get('ssim_mean', 0.0):.4f}",
                f"{metrics.get('ssim_std', 0.0):.4f}"
            ])


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation/test split at the batch level."""
    model.eval()
    tot_val_loss = 0.0
    tot_im_l2 = 0.0
    tot_im_l1 = 0.0
    batch_psnrs = []
    batch_ssims = []

    with torch.no_grad():
        for batch in val_loader:
            masked_k = batch['masked_Kspaces'].to(device=device, dtype=torch.float32)
            tar_k = batch['target_Kspace'].to(device=device, dtype=torch.float32)
            tar_im = batch['target_img'].to(device=device, dtype=torch.float32)

            rec_im, rec_k, _ = model(masked_k)
            loss, im_l2, im_l1, _, _, _ = criterion.calc_gen_loss(rec_im, rec_k, tar_im, tar_k)

            tot_val_loss += loss.item()
            tot_im_l2 += im_l2.item()
            tot_im_l1 += im_l1.item()

            rec_np = rec_im.cpu().numpy()
            tar_np = tar_im.cpu().numpy()

            b_psnr = compute_psnr(tar_np, rec_np)
            b_ssim = compute_ssim(tar_np, rec_np)
            batch_psnrs.append(b_psnr)
            batch_ssims.append(b_ssim)

    n_batches = len(batch_psnrs)
    mean_psnr = float(np.mean(batch_psnrs))
    std_psnr = float(np.std(batch_psnrs))
    mean_ssim = float(np.mean(batch_ssims))
    std_ssim = float(np.std(batch_ssims))

    metrics = {
        'val_loss': tot_val_loss / n_batches if n_batches > 0 else 0.0,
        'im_l2': tot_im_l2 / n_batches if n_batches > 0 else 0.0,
        'im_l1': tot_im_l1 / n_batches if n_batches > 0 else 0.0,
        'psnr_mean': mean_psnr,
        'psnr_std': std_psnr,
        'ssim_mean': mean_ssim,
        'ssim_std': std_ssim
    }
    model.train()
    return metrics


def train(args):
    """Main training workflow."""
    set_seed(getattr(args, 'seed', 42))

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Training initialized")
    logging.info(f"Target Device: {args.device}")

    metrics_logger = MetricsLogger(log_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    # Build Networks
    G_model = WNet(args).to(args.device)
    D_model = PatchGAN(input_nc=1, ndf=args.ndf, crop_center=args.crop_center).to(args.device)
    criterion = netLoss(args)

    # Optimizers & Scheduler
    initial_g_lr = args.lr * 0.5
    initial_d_lr = args.lr
    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=initial_g_lr, betas=(0.9, 0.999))
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=initial_d_lr, betas=(0.5, 0.999))
    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, mode='min', patience=20, factor=0.5)

    # Data Loaders
    train_dataset = IXIdataset(args.train_data_dir, args, validation_flag=False)
    val_dataset = IXIdataset(args.val_data_dir, args, validation_flag=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Resume checkpoint if specified
    start_epoch = 0
    best_psnr = float('-inf')
    if args.load_cp and os.path.exists(str(args.load_cp)):
        logging.info(f"Loading checkpoint: {args.load_cp}")
        ckpt = torch.load(args.load_cp, map_location=args.device)
        G_model.load_state_dict(ckpt['G_model_state_dict'])
        D_model.load_state_dict(ckpt['D_model_state_dict'])
        if args.resume_training:
            G_optimizer.load_state_dict(ckpt['G_optimizer_state_dict'])
            D_optimizer.load_state_dict(ckpt['D_optimizer_state_dict'])
            G_scheduler.load_state_dict(ckpt['G_scheduler_state_dict'])
            start_epoch = int(ckpt.get('epoch', 0)) + 1
            best_psnr = float(ckpt.get('best_psnr', float('-inf')))

    logging.info(f"Starting training from epoch {start_epoch + 1} to {args.epochs_n}")

    # Training Loop
    for epoch in range(start_epoch, args.epochs_n):
        G_model.train()
        D_model.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_im_l2 = 0.0
        epoch_im_l1 = 0.0
        epoch_k_l2 = 0.0
        epoch_adv = 0.0
        epoch_ffl = 0.0
        num_batches = len(train_loader)

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs_n}", unit="batch") as pbar:
            for b_idx, batch in enumerate(pbar):
                masked_k = batch['masked_Kspaces'].to(device=args.device, dtype=torch.float32)
                tar_k = batch['target_Kspace'].to(device=args.device, dtype=torch.float32)
                tar_im = batch['target_img'].to(device=args.device, dtype=torch.float32)

                # ==========================================================
                # Train Discriminator
                # ==========================================================
                train_d = (b_idx % (1 if epoch < 5 else 2) == 0) and (args.GAN_training == 1)
                d_loss_val = 0.0

                if train_d:
                    set_grad(D_model, True)
                    D_optimizer.zero_grad()

                    # Generate reconstruction without updating generator gradients
                    with torch.no_grad():
                        rec_im_d, _, _ = G_model(masked_k)

                    # Compute discriminator loss on real target and generated fake
                    d_real = D_model(tar_im)
                    d_fake = D_model(rec_im_d.detach())
                    _, _, total_d_loss = criterion.calc_disc_loss(d_real, d_fake)

                    # Optional gradient penalty for training stability
                    if getattr(args, 'GP', True):
                        gp = gradient_penalty(D_model, tar_im, rec_im_d.detach(), device=args.device)
                        total_d_loss = total_d_loss + 10.0 * gp

                    total_d_loss.backward()
                    D_optimizer.step()
                    d_loss_val = total_d_loss.item()

                # ==========================================================
                # Train Generator
                # ==========================================================
                set_grad(D_model, False)
                G_optimizer.zero_grad()

                # Forward pass through dual-domain generator
                rec_im, rec_k, _ = G_model(masked_k)

                # Compute composite objective across spatial, frequency, and adversarial terms
                d_fake_g = D_model(rec_im) if (args.GAN_training == 1) else None
                g_loss, im_l2, im_l1, k_l2, adv_l, ffl_l = criterion.calc_gen_loss(
                    rec_im, rec_k, tar_im, tar_k, D_fake=d_fake_g
                )

                g_loss.backward()
                G_optimizer.step()

                # Accumulate epoch metrics
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss_val
                epoch_im_l2 += im_l2.item()
                epoch_im_l1 += im_l1.item()
                epoch_k_l2 += k_l2.item()
                epoch_adv += adv_l.item()
                epoch_ffl += ffl_l.item()

                pbar.set_postfix({
                    'G_loss': f"{g_loss.item():.4f}",
                    'D_loss': f"{d_loss_val:.4f}",
                    'L2': f"{im_l2.item():.4f}"
                })

        # Epoch Summary
        train_summary = {
            'g_loss': epoch_g_loss / num_batches,
            'd_loss': epoch_d_loss / num_batches,
            'im_l2': epoch_im_l2 / num_batches,
            'im_l1': epoch_im_l1 / num_batches,
            'kspace_l2': epoch_k_l2 / num_batches,
            'adv_loss': epoch_adv / num_batches,
            'ffl_loss': epoch_ffl / num_batches,
            'lr': G_optimizer.param_groups[0]['lr']
        }
        metrics_logger.log_train(epoch + 1, train_summary)

        # Validation Step
        val_metrics = evaluate(G_model, val_loader, criterion, args.device)
        metrics_logger.log_val(epoch + 1, val_metrics)
        G_scheduler.step(val_metrics['val_loss'])

        logging.info(
            f"Epoch {epoch + 1} Done | Val PSNR: {val_metrics['psnr_mean']:.2f} +/- {val_metrics['psnr_std']:.2f} dB | "
            f"Val SSIM: {val_metrics['ssim_mean']:.4f} +/- {val_metrics['ssim_std']:.4f}"
        )

        # Tensorboard logging
        writer.add_scalar('Train/G_Loss', train_summary['g_loss'], epoch + 1)
        writer.add_scalar('Train/D_Loss', train_summary['d_loss'], epoch + 1)
        writer.add_scalar('Val/PSNR', val_metrics['psnr_mean'], epoch + 1)
        writer.add_scalar('Val/SSIM', val_metrics['ssim_mean'], epoch + 1)

        # Save latest checkpoint
        ckpt_state = {
            'epoch': epoch,
            'best_psnr': best_psnr,
            'G_model_state_dict': G_model.state_dict(),
            'D_model_state_dict': D_model.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
            'G_scheduler_state_dict': G_scheduler.state_dict(),
            'config': vars(args)
        }
        torch.save(ckpt_state, os.path.join(ckpt_dir, 'latest_checkpoint.pth'))

        # Save best model
        if val_metrics['psnr_mean'] > best_psnr:
            best_psnr = val_metrics['psnr_mean']
            ckpt_state['best_psnr'] = best_psnr
            torch.save(ckpt_state, os.path.join(ckpt_dir, 'best_model.pth'))
            logging.info(f"==> Saved New Best Model (PSNR: {best_psnr:.2f} dB)")

    writer.close()
    logging.info("Training Finished Successfully.")


def parse_args():
    """Parse command-line arguments and load configuration parameters."""
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to YAML configuration file')
    parser.add_argument('--data_dir', type=str, default=None, help='Override train/val data root directory')
    parser.add_argument('--train_data_dir', type=str, default=None, help='Train HDF5 directory')
    parser.add_argument('--val_data_dir', type=str, default=None, help='Validation HDF5 directory')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to sampling mask file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None, dest='epochs_n', help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Compute device (cuda or cpu)')
    parser.add_argument('--load_cp', type=str, default=None, help='Path to checkpoint for resuming or fine-tuning')
    parser.add_argument('--resume_training', type=int, default=None, help='Set to 1 to resume training state from checkpoint')
    cli_args = parser.parse_args()

    # Load YAML configuration
    config_dict = {}
    if os.path.exists(cli_args.config):
        with open(cli_args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

    # CLI arguments override YAML settings
    for key, value in vars(cli_args).items():
        if value is not None:
            config_dict[key] = value

    if cli_args.data_dir is not None:
        config_dict['train_data_dir'] = os.path.join(cli_args.data_dir, 'train')
        config_dict['val_data_dir'] = os.path.join(cli_args.data_dir, 'val')

    return SimpleNamespace(**config_dict)


if __name__ == '__main__':
    args = parse_args()
    train(args)
