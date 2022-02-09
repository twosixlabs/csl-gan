# Cooperative Secure Learning through Generative Adversarial Networks using Differential Privacy

## Introduction

This repository allows for training GANs on the MNIST and CelebA datasets with gradient clipping and immediate sensitivity using a fork of the Opacus library (see requirements.txt).

## Notes for Use

Default options are provided for each dataset (CelebA with deep convolutional ResNet GAN, MNIST with vanilla GAN) and can be found at the top of options.py

Only the gradient clipping (dp_mode=gc) and immediate sensitivity (dp_mode=is) privacy methods are fully implemented, others are very experimental.


## Example Invocations

MNIST conditional GAN on GPU0 with 600 batch size
```bash
python train.py MNIST -gd cuda:0 -dd cuda:0 -bs 600 --conditional
```

CelebA GAN on GPU0 and GPU1 with 128 batch size
```bash
python train.py CelebA -gd cuda:0 -dd cuda:1 -bs 128
```

MNIST conditional GAN with gradient clipping for differential privacy with a noise scale of 10
```bash
python train.py MNIST -gd cuda:0 -dd cuda:0 -bs 600 --conditional --dp_mode gc --sigma 10
```

MNIST conditional GAN with immediate sensitivity for approximate differential privacy with a noise scale of 10
```bash
python train.py MNIST -gd cuda:0 -dd cuda:0 -bs 600 --conditional --dp_mode is --sigma 10
```

CelebA GAN with gradient clipping using mean samples for adaptive clipping and for gradient penalty
```bash
python train.py CelebA -gd cuda:0 -dd cuda:0 -bs 128 -nms 32 --dp_mode gc -gcm adaptive-pl
```

CelebA GAN with per-parameter immediate sensitivity using mean samples for gradient penalty
```bash
python train.py CelebA -gd cuda:0 -dd cuda:0 -bs 128 -nms 32 --dp_mode is -ispp True
```


## Scripts

**train.py**: Main training script (see example invocations)
**budget_analysis.py**: Calculates epsilon delta budget for a given configuration and number of epochs
**downstream.py**: Downstream evaluation using sklearn classifiers for MNIST conditional model
**gensamples.py**: Loads a generator and generates samples
**mem_inf_attack.py**: Executes membership inference attack of Hayes et al. (2018) ([link](https://arxiv.org/abs/1705.07663)) and can calculate FID


## Implementation Notes

**Conditional Models**: The Vanilla GAN can use either a standard conditional GAN (CGAN) or an [Auxiliary Classifier GAN](https://arxiv.org/abs/1610.09585) (ACGAN). The WGAN model can use a standard conditional GAN (CGAN) or a custom conditional architecture for WGANs (WCGAN), which works by having the last layer of the critic give one output for each class, and only the output that corresponds to the true class is used.
