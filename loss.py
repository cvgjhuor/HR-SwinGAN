import os
import pickle
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
import torch.nn as nn
from focal_frequency_loss import FocalFrequencyLoss as FFL


def set_grad(network: nn.Module, requires_grad: bool):
    """Enable or disable gradients for all parameters in a network."""
    for param in network.parameters():
        param.requires_grad = requires_grad


# Composite loss computation across spatial and frequency domains
class netLoss:
    def __init__(self, config, masked_kspace_flag: bool = True):
        """Initialize composite loss criteria, weights, and sampling mask."""
        self.config = config
        self.device = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.masked_kspace_flag = masked_kspace_flag

        # Loss weights: [image_L2, image_L1, kspace_L2, adversarial, focal_frequency]
        loss_weights = getattr(config, 'loss_weights', [1000, 1000, 5, 0.5, 10])
        self.ImL2_weights = float(loss_weights[0])
        self.ImL1_weights = float(loss_weights[1])
        self.KspaceL2_weights = float(loss_weights[2])
        self.AdverLoss_weight = float(loss_weights[3])
        self.FFLLoss_weight = float(loss_weights[4])

        # Individual loss criteria
        self.ImL2Loss = nn.MSELoss()
        self.ImL1Loss = nn.SmoothL1Loss()
        self.AdverLoss = nn.BCEWithLogitsLoss()

        if self.masked_kspace_flag:
            self.KspaceL2Loss = nn.MSELoss(reduction='sum')
        else:
            self.KspaceL2Loss = nn.MSELoss()

        self.ffl = FFL()
        self._load_mask()

    def _load_mask(self):
        """Load undersampling mask and initialize binary complement mask."""
        mask_path = getattr(self.config, 'mask_path', None)
        if mask_path is None or not os.path.exists(mask_path):
            img_size = getattr(self.config, 'img_size', 256)
            self.mask = torch.ones((img_size, img_size), dtype=torch.bool, device=self.device)
            self.maskNot = torch.zeros((img_size, img_size), dtype=torch.bool, device=self.device)
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
            mask_tensor = torch.tensor(mask_arr == 1, dtype=torch.bool, device=self.device)
        elif ext in ('.tif', '.png', '.jpg'):
            with Image.open(mask_path) as img:
                mask_img = np.array(img.convert('L'))
            mask_shift = mask_img.astype(np.float32) / 255.0
            if getattr(self.config, 'mask_type', '') == 'cartesian' and getattr(self.config, 'sampling_percentage', 0) in (30, 50):
                mask_shift = np.fft.fftshift(mask_shift)
                mask_shift = np.rot90(mask_shift)
            mask_tensor = torch.tensor(mask_shift > 0.5, dtype=torch.bool, device=self.device)
        elif ext in ('.pkl', '.pickle'):
            with open(mask_path, 'rb') as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict) and 'mask1' in loaded:
                    loaded = loaded['mask1']
                mask_tensor = torch.tensor(loaded == 1, dtype=torch.bool, device=self.device)
        else:
            mask_tensor = torch.ones((256, 256), dtype=torch.bool, device=self.device)

        self.mask = mask_tensor
        self.maskNot = ~mask_tensor

    def img_space_loss(self, pred_Im: torch.Tensor, tar_Im: torch.Tensor):
        """Compute spatial domain Smooth L1 and L2 reconstruction losses."""
        return self.ImL1Loss(pred_Im, tar_Im), self.ImL2Loss(pred_Im, tar_Im)

    def k_space_loss(self, pred_K: torch.Tensor, tar_K: torch.Tensor):
        """Compute normalized frequency domain L2 loss over unacquired regions."""
        if self.masked_kspace_flag:
            norm_factor = torch.sum(self.maskNot).clamp(min=1.0) * tar_K.max().clamp(min=1e-8)
            return self.KspaceL2Loss(pred_K, tar_K) / norm_factor
        else:
            return self.KspaceL2Loss(pred_K, tar_K)

    def gen_adver_loss(self, D_fake: torch.Tensor) -> torch.Tensor:
        """Compute binary cross-entropy adversarial loss for generator."""
        real_target = torch.ones_like(D_fake, device=self.device)
        return self.AdverLoss(D_fake, real_target)

    def disc_adver_loss(self, D_real: torch.Tensor, D_fake: torch.Tensor):
        """Compute binary cross-entropy adversarial loss for discriminator."""
        real_target = torch.ones_like(D_real, device=self.device)
        fake_target = torch.zeros_like(D_fake, device=self.device)
        real_loss = self.AdverLoss(D_real, real_target)
        fake_loss = self.AdverLoss(D_fake, fake_target)
        return real_loss, fake_loss

    def calc_gen_loss(self, pred_Im: torch.Tensor, pred_K: torch.Tensor,
                      tar_Im: torch.Tensor, tar_K: torch.Tensor,
                      D_fake: torch.Tensor = None):
        """Compute total weighted generator loss across spatial, frequency, and adversarial terms."""
        ImL1, ImL2 = self.img_space_loss(pred_Im, tar_Im)
        KspaceL2 = self.k_space_loss(pred_K, tar_K)
        FFLloss = self.ffl(pred_Im, tar_Im)

        if D_fake is not None:
            advLoss = self.gen_adver_loss(D_fake)
        else:
            advLoss = torch.tensor(0.0, device=self.device)

        total_loss = (
            self.ImL2_weights * ImL2 +
            self.ImL1_weights * ImL1 +
            self.KspaceL2_weights * KspaceL2 +
            self.AdverLoss_weight * advLoss +
            self.FFLLoss_weight * FFLloss
        )
        return total_loss, ImL2, ImL1, KspaceL2, advLoss, FFLloss

    def calc_disc_loss(self, D_real: torch.Tensor, D_fake: torch.Tensor):
        """Compute average discriminator loss over real and generated samples."""
        real_loss, fake_loss = self.disc_adver_loss(D_real, D_fake)
        total_disc_loss = 0.5 * (real_loss + fake_loss)
        return real_loss, fake_loss, total_disc_loss
