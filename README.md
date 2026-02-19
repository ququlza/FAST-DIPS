# FAST‑DIPS: Adjoint‑Free Analytic Steps and Hard‑Constrained Likelihood Correction for Diffusion‑Prior Inverse Problems

Code for the paper [*"FAST‑DIPS: Adjoint‑Free Analytic Steps and Hard‑Constrained Likelihood Correction for Diffusion‑Prior Inverse Problems"*](https://openreview.net/forum?id=voMeZVAkKL), published at ICLR 2026.

![FAST-DIPS Paper Figure](/assets/figure1.png)
![FAST-DIPS Paper Figure 2](/assets/figure2.png)
## Abstract
$\textbf{FAST-DIPS}$ is a training-free solver for diffusion-prior inverse problems, including nonlinear forward operators. At each noise level, a pretrained denoiser provides an anchor $\mathbf{x}_ {0|t}$; we then perform a hard-constrained proximal correction in measurement space (AWGN) by solving
$\min_\mathbf{x} \tfrac{1}{2\gamma_t}\|\mathbf{x}-\mathbf{x}_{0|t}\|^2 \ \text{s.t.}\ \|\mathcal{A}(\mathbf{x})-\mathbf{y}\|\le\varepsilon$.
The correction is implemented via an adjoint-free ADMM with a closed-form projection onto the Euclidean ball and a few steepest-descent updates whose step size is analytic and computable from one VJP and one JVP—or a forward-difference surrogate—followed by decoupled re-annealing. We show this step minimizes a local quadratic model (with backtracking-based descent), any ADMM fixed point satisfies KKT for the hard-constraint, and mode substitution yields a bounded time-marginal error. We also derive a latent variant $\mathcal{A}\mapsto\mathcal{A}\circ\mathcal{D}$ and a one-parameter pixel → latent hybrid schedule. FAST-DIPS delivers comparable or better PSNR/SSIM/LPIPS while being substantially faster, requiring only autodiff access to $\mathcal{A}$ and no hand-coded adjoints or inner MCMC.

## Installation

1. Create and activate the conda environment:

Required versions:

- Python: `3.10`
- PyTorch: `2.3.0`
- CUDA: `12.1`

```bash
conda create -n fastdips python=3.10 -y
conda activate fastdips
pip install -r requirements.txt
```

2. Checkpoints:

Create the checkpoint directory:

```bash
# in FAST-DIPS folder
mkdir -p checkpoints
```

Required files (`./checkpoints`):

- `ffhq256.pt` (DDPM)
- `imagenet256.pt` (DDPM)
- `ldm_ffhq256.pt` (LDM)
- `ldm_imagenet256.pt` (LDM)
- `GOPRO_wVAE.pth` (nonlinear blur)

Download checkpoints:

- FFHQ DDPM:

```bash
gdown https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O checkpoints/ffhq256.pt
```

- ImageNet DDPM:

```bash
gdown https://drive.google.com/uc?id=1HAy7P19PckQLczVNXmVF-e_CRxq098uW -O checkpoints/imagenet256.pt
```

- FFHQ LDM:

```bash
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./checkpoints
unzip checkpoints/ffhq.zip -d ./checkpoints
mv checkpoints/model.ckpt checkpoints/ldm_ffhq256.pt
rm checkpoints/ffhq.zip
```

- ImageNet LDM:

```bash
wget https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt -P ./checkpoints/
mv checkpoints/model.ckpt checkpoints/ldm_imagenet256.pt
```

- Nonlinear blur:

```bash
gdown https://drive.google.com/uc?id=1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy -O checkpoints/GOPRO_wVAE.pth
```

3. Datasets:

Create the dataset directory:

```bash
# in FAST-DIPS folder
mkdir -p datasets
```

Download datasets:

- FFHQ:

```bash
gdown https://drive.google.com/uc?id=1i0oI8nt_b9XCHNPKM5KR92Y4t8ZVMDvR -O datasets/test-ffhq.zip
unzip datasets/test-ffhq.zip -d ./datasets
rm datasets/test-ffhq.zip
```

- ImageNet:

```bash
gdown https://drive.google.com/uc?id=1ezXMhLt2UPaqNJnYNQAFM9ZLUW52ulz5 -O datasets/test-imagenet.zip
unzip datasets/test-imagenet.zip -d ./datasets
rm datasets/test-imagenet.zip
```

## Quick Start

Predefined CLI command sets:

- `configs_fast_dips_cli.txt`

Supported tasks:

- `down_sampling`
- `inpainting`
- `inpainting_rand`
- `gaussian_blur`
- `motion_blur`
- `phase_retrieval`
- `nonlinear_blur`
- `hdr`

Supported spaces: `pixel`, `latent`

Supported datasets: `FFHQ (256 x 256)`, `ImageNet (256 x 256)`

Example (FFHQ, pixel, task: `down_sampling`):

```bash
python main.py \
  --data test-ffhq --model ffhq256ddpm --task down_sampling --sampler edm_FAST_DIPS \
  --save_dir results/FAST-DIPS/pixel/ffhq --num_runs 1 --batch_size 1 \
  --data_start_id 0 --data_end_id 100 \
  --T 75 --K 3 --S 1 --rho 200 --epsilon 0.05 \
  --name down_sampling_T75 --gpu 0
```

Example (FFHQ, latent, task: `phase_retrieval`):

```bash
python main.py \
  --data test-ffhq --model ffhq256ldm --task phase_retrieval --sampler latent_edm_FAST_DIPS \
  --save_dir results/FAST-DIPS/ldm/ffhq --num_runs 4 --batch_size 1 \
  --data_start_id 0 --data_end_id 100 \
  --T 25 --sigma_switch 5 --K_x 10 --K_z 10 --S_x 3 --S_z 3 \
  --rho_x 200 --rho_z 200 --epsilon 0.05 \
  --name phase_retrieval_T25 --gpu 0
```

Example (ImageNet, pixel, task: `down_sampling`):

```bash
python main.py \
  --data test-imagenet --model imagenet256ddpm --task down_sampling --sampler edm_FAST_DIPS \
  --save_dir results/FAST-DIPS/pixel/imagenet --num_runs 1 --batch_size 1 \
  --data_start_id 0 --data_end_id 100 \
  --T 75 --K 3 --S 1 --rho 200 --epsilon 0.05 \
  --name down_sampling_T75 --gpu 0
```

Example (ImageNet, latent, task: `phase_retrieval`):

```bash
python main.py \
  --data test-imagenet --model imagenet256ldm --task phase_retrieval --sampler latent_edm_FAST_DIPS \
  --save_dir results/FAST-DIPS/ldm/imagenet --num_runs 4 --batch_size 1 \
  --data_start_id 0 --data_end_id 100 \
  --T 25 --sigma_switch 5 --K_x 10 --K_z 10 --S_x 3 --S_z 3 \
  --rho_x 200 --rho_z 200 --epsilon 0.05 \
  --name phase_retrieval_T25 --gpu 0
```

Results are saved under `results/FAST-DIPS`, organized by space (`pixel`/`ldm`) and dataset (`ffhq`/`imagenet`).

## Acknowledgements

This implementation builds upon:

- [DAPS](https://github.com/zhangbingliang2019/DAPS)
- Nonlinear blur operator from [BKSE](https://github.com/VinAIResearch/blur-kernel-space-exploring)
- Motion blur operator from [motionblur](https://github.com/LeviBorodenko/motionblur)

## Citation

```bibtex
@inproceedings{
kim2026fastdips,
title={{FAST}\nobreakdash-{DIPS}: Adjoint\nobreakdash-Free Analytic Steps and Hard\nobreakdash-Constrained Likelihood Correction for Diffusion\nobreakdash-Prior Inverse Problems},
author={Minwoo Kim and Seunghyeok Shin and Hongki Lim},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=voMeZVAkKL}
}
```
