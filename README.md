# HR-SwinGAN

PyTorch implementation for MRI reconstruction.

## Installation

### Prerequisites
- Linux or Windows
- Python >= 3.8
- CUDA >= 11.8
- PyTorch >= 2.0.0

### Setup Environment
```bash
# Clone repository
git clone https://github.com/your-username/HR-SwinGAN.git
cd HR-SwinGAN

# Create conda environment
conda create -n hrswingan python=3.10 -y
conda activate hrswingan

# Install dependencies
pip install -r requirements.txt
```

## Structure

```
HR-SwinGAN/
├── configs/
│   └── config.yaml
├── Masks/
│   ├── cartesian/
│   ├── gaussian/
│   ├── poisson/
│   └── radial/
├── Networks/
│   ├── discriminator_model.py
│   ├── generator_model.py
│   ├── irpe.py
│   ├── lrcab.py
│   └── swin_transformer_unet.py
├── utils/
│   ├── dataset.py
│   └── metrics.py
├── loss.py
├── train.py
├── test.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Data Preparation

Organize preprocessed HDF5 files as follows:

```
data/
└── IXI/
    ├── train/
    ├── val/
    └── test/
```

## Training

```bash
python train.py --config configs/config.yaml
```

To resume training from a checkpoint:

```bash
python train.py --config configs/config.yaml --load_cp path/to/checkpoint.pth --resume_training 1
```

## Evaluation

```bash
python test.py --checkpoint path/to/checkpoint.pth --config configs/config.yaml --data_dir data/IXI/test
```

## License

MIT License
