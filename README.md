# BPCNet

This repository provides the research code for **BPCNet**, a wavelet-guided blind pixel compensation network for infrared images. The model is designed to compensate defective infrared sensor pixels while preserving local structural details through Haar wavelet features and high-frequency defect-aware masking.

This release is intended to support academic reference, code inspection, and independent re-implementation of the method described in our manuscript.

## Repository Structure

```text
.
|-- config.py              # Paths and training/testing hyperparameters
|-- datasets.py            # PyTorch dataset loaders
|-- train.py               # Training script
|-- test.py                # Inference script
|-- unet.py                # BPCNet/UNet architecture with wavelet modules
|-- models/
|   `-- FCSTN.py           # Auxiliary module used by the network
`-- README.md
```

## Environment

The code was developed with Python and PyTorch. A CUDA-enabled GPU is recommended for training.

Main dependencies:

```text
python
torch
torchvision
opencv-python
numpy
matplotlib
tqdm
kornia
pytorch-wavelets
```

Example installation:

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm kornia pytorch-wavelets
```

Please install a PyTorch version compatible with your CUDA driver. See the official PyTorch installation guide if GPU acceleration is required.

## Data Format

The training and validation loaders expect paired images arranged as follows:

```text
DATA_ROOT/
|-- train/
|   |-- imgs/      # Ground-truth clean infrared images
|   `-- noisy/     # Corresponding defective/blind-pixel images
|-- val/
|   |-- imgs/
|   `-- noisy/
`-- test/
    `-- noisy/     # Images for inference
```

Supported image formats include `.png`, `.jpg`, and `.bmp`.

The code assumes grayscale input images. During loading, images are read as single-channel images and converted to PyTorch tensors.

## Configuration

Before training or testing, edit `config.py`:

```python
data_dir = '/path/to/your/DATA_ROOT'
train_dir = 'train'
val_dir = 'val'
test_dir = os.path.join(data_dir, 'test', 'noisy')

models_dir = 'models'
losses_dir = 'losses'
res_dir = 'results'

lr = 1e-5
epochs = 200
batch_size = 2
test_bs = 2
```

If you want to run inference from a checkpoint, set `ckpt` to the checkpoint path. Pretrained weights and checkpoints are not provided in this repository; please use a checkpoint trained on your own data or another checkpoint that you are authorized to use.

## Training

After preparing the dataset and updating `config.py`, run:

```bash
python train.py
```

The script saves:

```text
models/    # Model checkpoints
losses/    # Training/validation loss curves
```

The training objective combines image-domain MSE reconstruction loss with Haar wavelet sub-band losses.

## Inference

To run inference, first set the path of a locally available checkpoint in `config.py`:

```python
ckpt = '/path/to/model_checkpoint.pth'
```

Then run:

```bash
python test.py
```

The restored images will be saved to:

```text
results/
```

## Availability Notice

Due to laboratory data confidentiality and security regulations, the infrared datasets used in the paper and the trained model weights cannot be publicly released. This repository is therefore a **code-only release**. The training data, validation data, test data, private infrared images, pretrained weights, and training checkpoints will not be publicly distributed.

The dataset construction procedure has been described in the paper. Users who want to run the code should prepare their own paired infrared blind-pixel compensation dataset following that procedure and the directory structure described below.

## Maintenance Policy

This repository is released as an **archival research code release** accompanying the paper. The code is provided as-is for academic use. We may not be able to provide ongoing maintenance, feature updates, environment debugging, dataset access support, or pretrained-weight access support.

Issues and pull requests may be reviewed when time permits, but no regular maintenance schedule is planned.

## Citation

If this code is useful for your research, please cite our paper:

```bibtex
@article{CUI2026106721,
  title   = {BPCNet: Blind pixel compensation for infrared images via wavelet transform network},
  journal = {Infrared Physics & Technology},
  pages   = {106721},
  year    = {2026},
  issn    = {1350-4495},
  doi     = {https://doi.org/10.1016/j.infrared.2026.106721},
  author  = {Guohao Cui and Lei Deng and Shengkun Wu and Yingying Gao and Heng Yu and Mingli Dong and Lianqing Zhu}
}
```

## Contact

For essential questions about the code release, please contact:

```text
yebidaxiong2025@163.com
```
