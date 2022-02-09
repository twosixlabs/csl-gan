import torch, torchvision, os, argparse, csv
import pandas as pd
import init_util
import util
import options

parser = argparse.ArgumentParser()

parser.add_argument("path", type=str, help="Path to the output folder containing the generator save")
parser.add_argument("-e", "--epochs", type=int, default=-1, help="Epochs trained for the generator save")
parser.add_argument("-n", "--num_samples", type=int, default=100)
parser.add_argument("-bs", "--batch_size", type=int, default=50)
parser.add_argument("-d", "--device", type=str, default="cpu")

opt = parser.parse_args()

opt.path = util.add_slash(opt.path)

output_dir = opt.path + "G-" + str(opt.epochs) + "-samples/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Load model

train_opt = options.load_opt(opt.path + "opt.txt")

torch.set_grad_enabled(False)
G, _ = init_util.init_models(train_opt, init_D=False)
G.to(opt.device)
util.load_model(opt.path + "saves/G-" + str(opt.epochs), G, device=opt.device)
G.eval()

with torch.no_grad():
    for i in range(opt.num_samples // opt.batch_size):
        fake_images = G(torch.empty((opt.batch_size, train_opt.g_latent_dim), device=opt.device).normal_(0.0, 1.0))
        if train_opt.dataset == "CelebA":
            fake_images = util.denorm_celeba(fake_images)

        for k in range(fake_images.size(0)):
            torchvision.utils.save_image(fake_images[k].data, os.path.join(output_dir, '%d.png')%(i*opt.batch_size+k+1), nrow=1)
