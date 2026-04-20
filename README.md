# Semantic-Guided Image Compression

This repository contains the implementation of a semantic-guided learned image compression reconstruction system developed for the CUH603CMD Graduation Project.

The project investigates whether semantic information extracted from a large multimodal model can improve compressed image reconstruction quality. It compares a baseline learned image compression model with an enhanced semantic-guided model. The enhanced model introduces CLIP-based semantic guidance at the decoder side to improve perceptual reconstruction quality.

## Project Title

**Effective Compressed Image Construction using Semantic Prompt**

## Author

**Jinghan Yang**

## Project Overview

This project builds a semantic-guided learned image compression reconstruction system based on **CLIP** and a **CompressAI-based learned image compression backbone**.

The system compares two main models:

1. **Baseline Model**
   - A learned image compression model without semantic guidance.
   - It uses the same compression backbone as the enhanced model.
   - It directly reconstructs images from compressed latent features.

2. **Enhanced Semantic-Guided Model**
   - A semantic-guided reconstruction model using CLIP semantic features.
   - It introduces semantic guidance at the decoder side.
   - It uses semantic adaptation and semantic refinement modules to improve reconstruction quality.

The aim is to test whether semantic features can improve reconstruction quality while maintaining conventional fidelity metrics.

## Main Features

- Learned image compression based on CompressAI
- CLIP ViT-B/32 semantic feature extraction
- Decoder-side semantic guidance
- Semantic adapter module
- Semantic refiner module
- Baseline and enhanced model comparison
- Ablation experiments
- Matched-capacity control experiment
- Semantic perturbation testing
- Evaluation using PSNR, SSIM, LPIPS, and FID

## Repository Structure

```text
semantic_guided_image_compression/
│
├── configs/
│   └── base_config.py
│
├── data/
│   └── dataloader.py
│
├── models/
│   ├── enhanced_vit.py
│   ├── semantic_adapter.py
│   └── vit_compressor.py
│
├── utils/
│   ├── clip_utils.py
│   ├── common.py
│   ├── fid_score.py
│   └── metrics.py
│
├── experiments.py
├── train.py
├── test.py
├── ablation.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SuTaiHui1/semantic_guided_image_compression.git
cd semantic_guided_image_compression
```

### 2. Create a Python environment

It is recommended to use Conda:

```bash
conda create -n semantic_compression python=3.10
conda activate semantic_compression
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If the CLIP package fails to install automatically, install it manually:

```bash
pip install git+https://github.com/openai/CLIP.git
```

## Dataset Preparation

This project uses image datasets for training and testing.

Recommended dataset setting:

- Training dataset: **DIV2K_train_HR**
- Testing dataset: **Kodak Lossless True Color Image Suite**

Large datasets are not included in this repository.

Please prepare the datasets manually and place them in the following structure:

```text
dataset/
├── train/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
│
└── test/
    ├── kodim01.png
    ├── kodim02.png
    └── ...
```

Then make sure the dataset paths in `configs/base_config.py` are set as follows:

```python
DIV2K_TRAIN_PATH = "./dataset/train"
KODAK_TEST_PATH = "./dataset/test"
```

If you use a different dataset location, change these two paths to your actual local paths.

## Configuration

Main configuration values are defined in:

```text
configs/base_config.py
```

Important settings include:

```python
BATCH_SIZE = 8
TEST_BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 2e-5
AUX_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SEMANTIC_LR_MULTIPLIER = 10.0

IMAGE_SIZE = 256
VAL_SPLIT = 0.1
RANDOM_SEED = 42

VIT_QUALITY = 6
VIT_PRETRAINED = True
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_EMBED_DIM = 512
```

If GPU memory is limited, reduce the batch size:

```python
BATCH_SIZE = 4
```

or:

```python
BATCH_SIZE = 2
```

For a faster test run, the number of epochs can also be reduced:

```python
EPOCHS = 50
```

## Training

To start training, run:

```bash
python train.py
```

By default, the training script trains the baseline model and then trains the enhanced semantic-guided model.

Training logs are saved to:

```text
logs/
```

Model checkpoints are saved to:

```text
checkpoints/
```

## Testing

After training, run:

```bash
python test.py
```

The testing script evaluates the trained models on the Kodak test set.

The evaluation metrics include:

- PSNR
- SSIM
- LPIPS
- FID

Higher PSNR and SSIM indicate better pixel-level fidelity.

Lower LPIPS and FID indicate better perceptual quality.

If reconstructed image saving is enabled in `configs/base_config.py`:

```python
SAVE_RECON_IMAGES = True
```

then reconstructed images will be saved to:

```text
recon_images/
```

Testing results will be printed in the console and saved into the log file.

## Ablation Experiments

To run ablation experiments, use:

```bash
python ablation.py
```

The project includes several experiment variants:

| Experiment Name | Description |
|---|---|
| baseline | Baseline model without semantic modules |
| full_optimized | Main enhanced semantic-guided model |
| structure_only | Uses semantic modules with zero semantic input |
| loss_only | Uses semantic loss without semantic modules |
| full_ablation | Uses semantic modules and semantic loss |
| aux_only_matched | Matched-capacity control with auxiliary objectives |
| semantic_guided_strict | Strict semantic-guided variant with additional semantic constraints |

These experiments help determine whether performance improvements come from semantic guidance, additional model capacity, loss design, or other factors.

## Semantic Perturbation Testing

The testing workflow also supports semantic perturbation settings, including:

- normal CLIP semantic input
- zero semantic input
- random semantic input
- shuffled semantic input

These tests are used to examine whether the enhanced model truly depends on meaningful semantic information during reconstruction.

## Evaluation Metrics

The project reports four main metrics:

| Metric | Meaning | Better Direction |
|---|---|---|
| PSNR | Pixel-level reconstruction fidelity | Higher is better |
| SSIM | Structural similarity | Higher is better |
| LPIPS | Learned perceptual similarity | Lower is better |
| FID | Distribution-level perceptual quality | Lower is better |

## Model Design

The enhanced model contains three main parts:

### 1. Compression Backbone

The compression backbone is based on CompressAI. It encodes the input image into latent features and reconstructs the image through a decoder.

### 2. CLIP Semantic Extractor

The semantic branch uses CLIP ViT-B/32 to extract a 512-dimensional semantic embedding from the input image.

The CLIP extractor is used as a frozen semantic feature extractor.

### 3. Semantic Guidance Modules

The enhanced model introduces decoder-side semantic guidance through:

- Semantic adapter
- Semantic refiner

The semantic adapter uses CLIP semantic embeddings to guide decoder features.

The semantic refiner applies residual enhancement to improve the reconstructed image.

## Outputs

During training and testing, the following folders may be generated:

```text
checkpoints/
logs/
recon_images/
```

These folders are usually ignored by Git because they may contain large generated files.

## Notes on Large Files

Large files such as datasets, trained checkpoints, logs, and reconstructed images are not included in this repository by default.

If trained checkpoints are required, they should be shared separately through GitHub Releases, Google Drive, OneDrive, or another external storage service.

## Reproducibility

To reproduce the main workflow:

1. Install dependencies.
2. Prepare the training and testing datasets.
3. Make sure dataset paths in `configs/base_config.py` are correct:

```python
DIV2K_TRAIN_PATH = "./dataset/train"
KODAK_TEST_PATH = "./dataset/test"
```

4. Run training:

```bash
python train.py
```

5. Run testing:

```bash
python test.py
```

6. Optionally run ablation experiments:

```bash
python ablation.py
```

## Troubleshooting

### 1. CUDA out of memory

If GPU memory is insufficient, reduce the batch size in:

```text
configs/base_config.py
```

For example:

```python
BATCH_SIZE = 4
```

or:

```python
BATCH_SIZE = 2
```

### 2. Dataset path error

If the program reports that no images are found, check whether the dataset paths in `configs/base_config.py` are correct.

Recommended setting:

```python
DIV2K_TRAIN_PATH = "./dataset/train"
KODAK_TEST_PATH = "./dataset/test"
```

### 3. CLIP installation error

If CLIP cannot be installed through `requirements.txt`, run:

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 4. FID computation error

FID requires reconstructed images to be saved. Please make sure:

```python
SAVE_RECON_IMAGES = True
```

is enabled in `configs/base_config.py`.

## Acknowledgement

This project uses open-source tools and research frameworks including:

- PyTorch
- TorchVision
- CompressAI
- CLIP
- LPIPS
- pytorch-fid
- scikit-image
- OpenCV

## Author

Jinghan Yang

CUH603CMD Graduation Project  
Communication University of China, Hainan International College  
Coventry University