import argparse, torch
import init_util
import util
import options

parser = argparse.ArgumentParser()

parser.add_argument("path", type=str, help="Path to the output folder")
parser.add_argument("-e", "--epochs", type=int, default=-1, help="Epochs trained for desired save")
parser.add_argument("-d", "--device", type=str, default="cpu")

opt = parser.parse_args()
opt.path = util.add_slash(opt.path)

train_opt = options.load_opt(opt.path + "opt.txt")
train_opt.g_device = opt.device
train_opt.d_device = opt.device

torch.set_grad_enabled(False)
G, D = init_util.init_models(train_opt)

util.load_model(opt.path + "saves/G-" + str(opt.epochs), G, opt.device)
G.eval()

util.load_model(opt.path + "saves/D-" + str(opt.epochs), D, opt.device)
D.eval()


z = torch.empty((1, train_opt.g_latent_dim), device=opt.device).normal_(0.0, 1.0)
y = torch.empty((1)).random_(0, train_opt.n_classes).long() if train_opt.conditional else None
D(G(z, y), y)
