import os
import h5py
import pickle
import logging
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset


class IXIdataset(Dataset):
    def __init__(self, data_dir: str, config, validation_flag: bool = False):
        """Initialize dataset index from directory containing HDF5 MRI volumes."""
        self.config = config
        self.data_dir = data_dir
        self.validation_flag = validation_flag

        self.num_input_slices = getattr(config, 'num_input_slices', 3)
        self.img_size = getattr(config, 'img_size', 256)
        self.slice_range = getattr(config, 'slice_range', [35, 115])
        self.minmax_noise_val = getattr(config, 'minmax_noise_val', [-0.01, 0.01])

        # Scan for HDF5 files
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        all_files = [
            f for f in os.listdir(data_dir)
            if not f.startswith('.') and (f.endswith('.hdf5') or f.endswith('.h5'))
        ]
        self.file_names = sorted([os.path.splitext(f)[0] for f in all_files])

        # Build index table (file_name, slice_idx)
        self.ids = []
        for file_name in self.file_names:
            full_path = os.path.join(self.data_dir, file_name + ('.hdf5' if os.path.exists(os.path.join(self.data_dir, file_name + '.hdf5')) else '.h5'))
            try:
                with h5py.File(full_path, 'r') as h5f:
                    data_shape = h5f['data'].shape
                    # Data layout could be (H, W, Slices) or (Slices, H, W)
                    if data_shape[2] >= self.slice_range[1]:
                        num_slices = data_shape[2]
                        self.slice_dim = 2
                    elif data_shape[0] >= self.slice_range[1]:
                        num_slices = data_shape[0]
                        self.slice_dim = 0
                    else:
                        continue

                for s_idx in range(self.slice_range[0], min(self.slice_range[1], num_slices)):
                    self.ids.append((file_name, s_idx))
            except Exception as e:
                logging.warning(f"Skipping corrupted or inaccessible file {file_name}: {e}")
                continue

        split_type = "validation/test" if self.validation_flag else "training"
        logging.info(f"Loaded {len(self.ids)} slices from {len(self.file_names)} subjects for {split_type} set.")

        # Load undersampling mask
        self._load_mask()

    def _load_mask(self):
        """Load and cache undersampling mask."""
        mask_path = getattr(self.config, 'mask_path', None)
        if mask_path is None or not os.path.exists(mask_path):
            self.mask = np.ones((self.img_size, self.img_size), dtype=np.float32)
            self.maskedNot = np.zeros((self.img_size, self.img_size), dtype=np.float32)
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
            mask_float = (mask_arr == 1).astype(np.float32)
        elif ext in ('.tif', '.png', '.jpg'):
            with Image.open(mask_path) as img:
                mask_img = np.array(img.convert('L'))
            mask_float = (mask_img.astype(np.float32) / 255.0)
            if getattr(self.config, 'mask_type', '') == 'cartesian' and getattr(self.config, 'sampling_percentage', 0) in (30, 50):
                mask_float = np.fft.fftshift(mask_float)
                mask_float = np.rot90(mask_float)
            mask_float = (mask_float > 0.5).astype(np.float32)
        elif ext in ('.pkl', '.pickle'):
            with open(mask_path, 'rb') as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict) and 'mask1' in loaded:
                    loaded = loaded['mask1']
                mask_float = (loaded == 1).astype(np.float32)
        else:
            raise ValueError(f"Unsupported mask format: {ext}")

        self.mask = mask_float
        self.maskedNot = 1.0 - mask_float

    def __len__(self) -> int:
        """Return total number of extracted 2D slices."""
        return len(self.ids)

    def crop_to_shape(self, kspace_cplx: np.ndarray) -> np.ndarray:
        """Crop k-space to target spatial dimension."""
        h, w = kspace_cplx.shape[:2]
        if h == self.img_size and w == self.img_size:
            return kspace_cplx
        if h % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :]
        if w % 2 == 1:
            kspace_cplx = kspace_cplx[:, :-1]
        h_crop = max(0, (kspace_cplx.shape[0] - self.img_size) // 2)
        w_crop = max(0, (kspace_cplx.shape[1] - self.img_size) // 2)
        if h_crop > 0:
            kspace_cplx = kspace_cplx[h_crop:h_crop + self.img_size, :]
        if w_crop > 0:
            kspace_cplx = kspace_cplx[:, w_crop:w_crop + self.img_size]
        return kspace_cplx

    def ifft2(self, kspace_cplx: np.ndarray) -> np.ndarray:
        """Compute 2D inverse FFT to yield magnitude image."""
        return np.absolute(np.fft.ifft2(kspace_cplx))[None, :, :]

    def fft2(self, img: np.ndarray) -> np.ndarray:
        """Compute centered 2D FFT from spatial image."""
        return np.fft.fftshift(np.fft.fft2(img))

    def slice_preprocess(self, kspace_cplx: np.ndarray):
        """Prepare real/imaginary channels, apply mask, and add background noise."""
        kspace_cplx = self.crop_to_shape(kspace_cplx)
        h, w = kspace_cplx.shape[:2]

        kspace = np.zeros((h, w, 2), dtype=np.float32)
        kspace[:, :, 0] = np.real(kspace_cplx).astype(np.float32)
        kspace[:, :, 1] = np.imag(kspace_cplx).astype(np.float32)
        image = self.ifft2(kspace_cplx).astype(np.float32)

        # Transpose from (H, W, 2) to (2, H, W)
        kspace = kspace.transpose((2, 0, 1))

        # Apply undersampling mask
        masked_kspace = kspace * self.mask

        # Add small synthetic noise in unacquired sampling region
        noise = np.random.uniform(
            low=self.minmax_noise_val[0],
            high=self.minmax_noise_val[1],
            size=masked_kspace.shape
        ).astype(np.float32)
        masked_kspace += noise * self.maskedNot

        return masked_kspace, kspace, image

    def __getitem__(self, idx: int):
        """Load multi-slice k-space input and center slice ground truth target."""
        file_name, slice_num = self.ids[idx]
        half_window = self.num_input_slices // 2

        full_file_path = os.path.join(
            self.data_dir,
            file_name + ('.hdf5' if os.path.exists(os.path.join(self.data_dir, file_name + '.hdf5')) else '.h5')
        )

        # Load adjacent slices for 2.5D spatial-temporal context
        with h5py.File(full_file_path, 'r') as h5f:
            vol = h5f['data']
            if getattr(self, 'slice_dim', 2) == 2:
                # Shape: [H, W, Slices]
                imgs = vol[:, :, slice_num - half_window:slice_num + half_window + 1]
            else:
                # Shape: [Slices, H, W]
                imgs_sliced = vol[slice_num - half_window:slice_num + half_window + 1, :, :]
                imgs = np.transpose(imgs_sliced, (1, 2, 0))

        # Initialize input and target arrays
        masked_kspaces = np.zeros((self.num_input_slices * 2, self.img_size, self.img_size), dtype=np.float32)
        target_kspace = np.zeros((2, self.img_size, self.img_size), dtype=np.float32)
        target_img = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)

        # Simulate k-space and apply sampling mask for each input slice
        for s in range(self.num_input_slices):
            img_slice = imgs[:, :, s]
            kspace_complex = self.fft2(img_slice)
            s_masked_k, s_full_k, s_full_img = self.slice_preprocess(kspace_complex)
            masked_kspaces[s * 2:s * 2 + 2, :, :] = s_masked_k
            # Center slice serves as reconstruction ground truth
            if s == half_window:
                target_kspace = s_full_k
                target_img = s_full_img

        return {
            'masked_Kspaces': torch.from_numpy(masked_kspaces),
            'target_Kspace': torch.from_numpy(target_kspace),
            'target_img': torch.from_numpy(target_img)
        }
