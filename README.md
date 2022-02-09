# Cooperative Secure Learning through Generative Adversarial Networks using Differential Privacy

## Notes for Use

Default options are provided for each dataset (CelebA with deep convolutional ResNet GAN, MNIST with vanilla GAN) and can be found at the top of options.py

Only the gradient clipping (dp_mode=gc) and immediate sensitivity (dp_mode=is) privacy methods are fully implemented, others are very experimental.


## Example Invocations

MNIST GAN on GPU0 with 600 batch size
```bash
python train.py MNIST -gd cuda:0 -dd cuda:0 -bs 600
```

MNIST GAN with adaptive gradient clipping using adaptive clipping with mean samples
```bash
python train.py MNIST -gd cuda:0 -dd cuda:0 -bs 600 -ugc -gcm adaptive -nms 20
```

CelebA GAN on GPU0 and GPU1 with 128 batch size
```bash
python train.py CelebA -gd cuda:0 -dd cuda:1 -bs 128
```

CelebA GAN with gradient clipping using mean samples for adaptive clipping and for gradient penalty
```bash
python train.py CelebA -gd cuda:0 -dd cuda:0 -bs 64 -nms 32 -ugc -gcm adaptive-pl
```

CelebA GAN with gradient clipping using a public set of size 4000 for per-layer adaptive clipping, gradient penalty, and a warm start with 1000 iterations
```bash
python train.py CelebA -gd cuda:0 -dd cuda:0 -bs 64 -ugc -gcm adaptive-pl -pss 4000 -wi 1000
```

## Scripts

**train.py**: Main training script (see example invocations)
**budget_analysis.py**: Calculates epsilon delta budget for a given configuration and number of epochs
**downstream.py**: Downstream evaluation using sklearn classifiers for MNIST conditional model
**gensamples.py**: Loads a generator and generates samples
**mem_inf_attack.py**: Executes membership inference attack of Hayes et al. (2018) ([link](https://arxiv.org/abs/1705.07663)) and can calculate FID


## Implementation Notes

**Conditional Models**: The Vanilla GAN can use either a standard conditional GAN (CGAN) or an [Auxiliary Classifier GAN](https://arxiv.org/abs/1610.09585) (ACGAN). The WGAN model can use a standard conditional GAN (CGAN) or a custom conditional architecture for WGANs (WCGAN), which works by having the last layer of the critic give one output for each class, and only the output that corresponds to the true class is used.
