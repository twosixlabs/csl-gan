## Notes

See options.py for default options for each dataset.

## Example Configurations

MNIST GAN on GPU0 with 600 batch size
```bash
python train.py MNIST -gd cuda:0 -dd cuda:0 -bs 600
```

MNIST GAN with adaptive gradient clipping using moving average of gradient norm
```bash
python train.py MNIST -gd cuda:0 -dd cuda:0 -bs 600 -ugc -gcm moving-avg-pl
```

CelebA GAN on GPU0 and GPU1 with 128 batch size
```bash
python train.py CelebA -gd cuda:0 -dd cuda:1 -bs 128
```

CelebA GAN with gradient clipping using mean samples for adaptive clipping and for gradient penalty
```bash
python train.py CelebA -gd cuda:0 -dd cuda:0 -bs 64 -nms 32 -ugc -gcm adaptive-pl
```

CelebA GAN with gradient clipping using a public set of size 4000 for adaptive clipping, gradient penalty, and a warm start with 1000 iterations
```bash
python train.py CelebA -gd cuda:0 -dd cuda:0 -bs 64 -ugc -gcm adaptive-pl -pss 4000 -wi 1000
```
