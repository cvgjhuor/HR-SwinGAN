import os
import sys
import yaml
import argparse
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

from Networks import WNet
from utils.dataset import IXIdataset
from utils.metrics import psnr as compute_psnr, ssim as compute_ssim, nmse as compute_nmse


def parse_args():
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model weights (.pth)')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to YAML configuration file')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory containing test HDF5 files')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to evaluation undersampling mask')
    parser.add_argument('--output_dir', type=str, default='results/test_outputs', help='Directory to store evaluation results')
    parser.add_argument('--batch_size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference (cuda or cpu)')
    parser.add_argument('--save_images', action='store_true', help='Save reconstruction comparison PNG images')
    parser.add_argument('--max_save_samples', type=int, default=20, help='Maximum number of comparison images to save')
    return parser.parse_args()


def save_comparison_image(undersampled_img, rec_img, target_img, save_path):
    """
    Save side-by-side comparison figure:
    [Zero-Fill Undersampled | Reconstructed | Ground Truth | Absolute Error (x5)]
    """
    def to_u8(img):
        clipped = np.clip(img, 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)

    zf_u8 = to_u8(undersampled_img)
    rec_u8 = to_u8(rec_img)
    gt_u8 = to_u8(target_img)

    error = np.abs(rec_img - target_img)
    error_vis = np.clip(error * 5.0 * 255.0, 0, 255).astype(np.uint8)

    if HAS_CV2:
        error_color = cv2.applyColorMap(error_vis, cv2.COLORMAP_JET)
        zf_bgr = cv2.cvtColor(zf_u8, cv2.COLOR_GRAY2BGR)
        rec_bgr = cv2.cvtColor(rec_u8, cv2.COLOR_GRAY2BGR)
        gt_bgr = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([zf_bgr, rec_bgr, gt_bgr, error_color])
        cv2.imwrite(save_path, combined)
    else:
        combined = np.hstack([zf_u8, rec_u8, gt_u8, error_vis])
        Image.fromarray(combined).save(save_path)


def run_evaluation():
    """Execute evaluation pipeline, report quantitative results, and save outputs."""
    cli_args = parse_args()

    # Load configuration
    config_dict = {}
    if os.path.exists(cli_args.config):
        with open(cli_args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

    # CLI overrides
    if cli_args.data_dir is not None:
        config_dict['predict_data_dir'] = cli_args.data_dir
    if cli_args.mask_path is not None:
        config_dict['mask_path'] = cli_args.mask_path
    config_dict['device'] = cli_args.device
    config_dict['batch_size'] = cli_args.batch_size

    config = SimpleNamespace(**config_dict)

    os.makedirs(cli_args.output_dir, exist_ok=True)
    if cli_args.save_images:
        images_dir = os.path.join(cli_args.output_dir, 'visualizations')
        os.makedirs(images_dir, exist_ok=True)

    print("=" * 70)
    print("Evaluation")
    print(f"Checkpoint: {cli_args.checkpoint}")
    print(f"Data Dir:   {getattr(config, 'predict_data_dir', 'data/test')}")
    print(f"Mask Path:  {getattr(config, 'mask_path', 'default')}")
    print(f"Device:     {cli_args.device}")
    print("=" * 70)

    # Initialize model
    model = WNet(config).to(cli_args.device)

    # Load checkpoint weights
    if not os.path.exists(cli_args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {cli_args.checkpoint}")

    checkpoint = torch.load(cli_args.checkpoint, map_location=cli_args.device)
    if 'G_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['G_model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model weights loaded successfully.")

    # Initialize test dataset & dataloader
    test_data_dir = getattr(config, 'predict_data_dir', None)
    if test_data_dir is None or not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"Test data directory does not exist: {test_data_dir}")

    test_dataset = IXIdataset(test_data_dir, config, validation_flag=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'val_num_workers', 4),
        pin_memory=True,
        drop_last=False
    )

    batch_psnrs = []
    batch_ssims = []
    batch_nmses = []
    saved_images_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            masked_k = batch['masked_Kspaces'].to(device=cli_args.device, dtype=torch.float32)
            tar_im = batch['target_img'].to(device=cli_args.device, dtype=torch.float32)

            # Generator forward pass
            rec_im, rec_k, rec_mid_img = model(masked_k)

            rec_np = rec_im.cpu().numpy()
            tar_np = tar_im.cpu().numpy()
            mid_np = rec_mid_img.cpu().numpy()

            # Compute batch-level quantitative metrics: PSNR, SSIM, NMSE
            b_psnr = compute_psnr(tar_np, rec_np)
            b_ssim = compute_ssim(tar_np, rec_np)
            b_nmse = compute_nmse(tar_np, rec_np)

            batch_psnrs.append(b_psnr)
            batch_ssims.append(b_ssim)
            batch_nmses.append(b_nmse)

            # Export side-by-side comparison: [Zero-Fill | Reconstructed | Ground Truth | Error]
            if cli_args.save_images and saved_images_count < cli_args.max_save_samples:
                for s in range(min(tar_im.size(0), cli_args.max_save_samples - saved_images_count)):
                    save_path = os.path.join(images_dir, f"sample_{saved_images_count:04d}.png")
                    save_comparison_image(mid_np[s].squeeze(), rec_np[s].squeeze(), tar_np[s].squeeze(), save_path)
                    saved_images_count += 1

    # Aggregate mean and standard deviation across all evaluated batches
    total_batches = len(batch_psnrs)
    mean_batch_psnr = float(np.mean(batch_psnrs))
    std_batch_psnr = float(np.std(batch_psnrs))
    mean_batch_ssim = float(np.mean(batch_ssims))
    std_batch_ssim = float(np.std(batch_ssims))
    mean_batch_nmse = float(np.mean(batch_nmses))
    std_batch_nmse = float(np.std(batch_nmses))

    # Print summary
    print("\n" + "=" * 70)
    print("Quantitative Evaluation Results")
    print("=" * 70)
    print(f"Total Batches Evaluated: {total_batches}")
    print("-" * 70)
    print(f"PSNR: {mean_batch_psnr:.2f} +/- {std_batch_psnr:.2f} dB")
    print(f"SSIM: {mean_batch_ssim:.4f} +/- {std_batch_ssim:.4f}")
    print(f"NMSE: {mean_batch_nmse:.6f} +/- {std_batch_nmse:.6f}")
    print("=" * 70)

    # Save summary report to CSV
    csv_report_path = os.path.join(cli_args.output_dir, "test_summary.csv")
    with open(csv_report_path, 'w', encoding='utf-8') as f:
        f.write("metric,mean,std\n")
        f.write(f"psnr,{mean_batch_psnr:.4f},{std_batch_psnr:.4f}\n")
        f.write(f"ssim,{mean_batch_ssim:.6f},{std_batch_ssim:.6f}\n")
        f.write(f"nmse,{mean_batch_nmse:.8f},{std_batch_nmse:.8f}\n")

    print(f"Summary metrics saved to: {csv_report_path}")
    if cli_args.save_images:
        print(f"Visualization images saved to: {images_dir}")


if __name__ == '__main__':
    run_evaluation()
