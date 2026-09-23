import os
import pickle
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
import torch.nn as nn
from .swin_transformer_unet import SwinUnet


# Dual-domain cascade generator combining k-space and image-domain networks
class WNet(nn.Module):
    def __init__(self, config, masked_kspace: bool = True):
        """Initialize dual-domain cascade generator with k-space and image UNet branches."""
        super(WNet, self).__init__()
        self.config = config
        self.masked_kspace = masked_kspace
        self.device = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.bilinear = getattr(config, 'bilinear', True)

        # Load sampling mask
        self._load_mask()

        # Each slice contains real and imaginary channels (2 * num_slices)
        num_slices = getattr(config, 'num_input_slices', 3)
        kspace_in_chans = num_slices * 2
        kspace_out_chans = 2

        # K-space reconstruction branch: estimates full complex k-space
        self.kspace_Unet = SwinUnet(
            config,
            img_size=getattr(config, 'img_size', 256),
            num_classes=kspace_out_chans,
            in_chans=kspace_in_chans
        )

        # Image domain refinement branch: refines reconstructed spatial image
        self.img_UNet = SwinUnet(
            config,
            img_size=getattr(config, 'img_size', 256),
            num_classes=1,
            in_chans=1
        )

    def _load_mask(self):
        """Load undersampling mask from file."""
        mask_path = getattr(self.config, 'mask_path', None)
        if mask_path is None or not os.path.exists(mask_path):
            # Fallback to all-ones mask if path not specified
            img_size = getattr(self.config, 'img_size', 256)
            self.register_buffer('mask', torch.ones((img_size, img_size), dtype=torch.bool))
            self.register_buffer('maskNot', torch.zeros((img_size, img_size), dtype=torch.bool))
            return

        ext = os.path.splitext(mask_path)[1].lower()
        if ext == '.mat':
            mat_dict = sio.loadmat(mask_path)
            mask_arr = None
            for key in ['Umask', 'maskRS2', 'population_matrix', 'mask1']:
                if key in mat_dict:
                    mask_arr = mat_dict[key]
                    break
            if mask_arr is None:
                for k, v in mat_dict.items():
                    if isinstance(v, np.ndarray) and v.ndim == 2:
                        mask_arr = v
                        break
            if mask_arr is None:
                raise KeyError(f"Unable to find valid mask array in {mask_path}")
            mask_tensor = torch.tensor(mask_arr == 1, dtype=torch.bool)
        elif ext in ('.tif', '.png', '.jpg'):
            with Image.open(mask_path) as img:
                mask_img = np.array(img.convert('L'))
            mask_shift = mask_img.astype(np.float32) / 255.0
            if getattr(self.config, 'mask_type', '') == 'cartesian' and getattr(self.config, 'sampling_percentage', 0) in (30, 50):
                mask_shift = np.fft.fftshift(mask_shift)
                mask_shift = np.rot90(mask_shift)
            mask_tensor = torch.tensor(mask_shift > 0.5, dtype=torch.bool)
        elif ext in ('.pkl', '.pickle'):
            with open(mask_path, 'rb') as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict) and 'mask1' in loaded:
                    loaded = loaded['mask1']
                mask_tensor = torch.tensor(loaded == 1, dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported mask file extension: {ext}")

        self.register_buffer('mask', mask_tensor)
        self.register_buffer('maskNot', ~mask_tensor)

    def fftshift(self, img: torch.Tensor) -> torch.Tensor:
        """Apply 2D FFT shift along spatial dimensions."""
        h_half = img.shape[-2] // 2
        w_half = img.shape[-1] // 2
        out = torch.zeros_like(img)
        out[..., :h_half, :w_half] = img[..., h_half:, w_half:]
        out[..., h_half:, w_half:] = img[..., :h_half, :w_half:]
        out[..., :h_half, w_half:] = img[..., h_half:, :w_half:]
        out[..., h_half:, :w_half] = img[..., :h_half, w_half:]
        return out

    # Convert complex k-space [B, 2, H, W] to spatial magnitude image [B, 1, H, W]
    def inverseFT(self, Kspace: torch.Tensor) -> torch.Tensor:
        """Compute 2D inverse Fourier transform from complex k-space to magnitude image."""
        k_perm = Kspace.permute(0, 2, 3, 1)  # [B, H, W, 2]
        k_complex = torch.view_as_complex(k_perm.contiguous())
        img_complex = torch.fft.ifft2(k_complex)
        img_mag = torch.abs(img_complex)[:, None, :, :]
        return img_mag

    def forward(self, Kspace: torch.Tensor):
        """Forward pass through frequency reconstruction, data consistency, and spatial refinement."""
        # 1. Frequency domain reconstruction: estimate full k-space
        rec_all_Kspace = self.kspace_Unet(Kspace)

        # 2. Data consistency (DC): keep acquired points, fill missing points with network predictions
        if self.masked_kspace:
            center_idx = Kspace.shape[1] // 2
            sampled_acquired = Kspace[:, center_idx - 1:center_idx + 1, :, :]
            rec_Kspace = self.mask * sampled_acquired + self.maskNot * rec_all_Kspace
            rec_mid_img = self.inverseFT(rec_Kspace)
        else:
            rec_Kspace = rec_all_Kspace
            rec_mid_img = self.fftshift(self.inverseFT(rec_Kspace))

        # 3. Spatial domain refinement: eliminate residual artifacts and refine fine textures
        refine_Img = self.img_UNet(rec_mid_img)
        rec_img = torch.tanh(refine_Img + rec_mid_img)
        rec_img = torch.clamp(rec_img, 0.0, 1.0)

        return rec_img, rec_Kspace, rec_mid_img
