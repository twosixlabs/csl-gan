import numpy as np
import argparse
from pathlib import Path
import torch
from torch import nn

import options
import util
import init_util


torch.set_default_dtype(torch.double)

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("epoch", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("-d", "--device", type=str, default="cpu")
parser.add_argument("-ni", "--num_iter", type=int, default=100)
opt = parser.parse_args()

opt.path = util.add_slash(opt.path)
train_opt = options.load_opt(opt.path + "opt.txt")

if not opt.batch_size is None:
    train_opt.batch_size = opt.batch_size
train_opt.d_device = opt.device

# _, D = init_util.init_models(train_opt, init_G=False)
# D.to(opt.device)
# path = opt.path + "saves/D-" + str(opt.epoch)
# if not Path(path).is_file():
#     raise Exception("Save not found for epoch.")
# util.load_model(path, D)
# D = D.double()


class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 5)
        self.lin2 = nn.Linear(5, 1)

    def forward(self, x, y):
        return self.lin2(self.lin1(x.reshape(x.size(0), -1))), None

D = Disc().double()


_, dl, _, _ = init_util.init_data(train_opt)

def get_imm_sens(loss, inp):
    # (1) first-order gradient (wrt parameters)
    first_order_grads = torch.autograd.grad(loss, D.parameters(), retain_graph=True, create_graph=True, allow_unused=True)

    # (2) L2 norm of the gradient from (1)
    grad_l2_norm = torch.norm(torch.cat([x.reshape(-1) for i, x in enumerate(first_order_grads) if not x is None]), p=2)

    # (3) Gradient (wrt inputs) of the L2 norm of the gradient from (2)
    sensitivity_vec = torch.autograd.grad(grad_l2_norm, inp, create_graph=True, retain_graph=True)[0]

    # (4) L2 norm of (3) - "immediate sensitivity"
    s = [torch.norm(v, p=2) for v in sensitivity_vec]

    print(torch.autograd.grad(s[0], inp, retain_graph=True, create_graph=True, allow_unused=True))

    grad_imm_sens_wrt_input = np.array([torch.autograd.grad(si, inp[i], retain_graph=True, create_graph=True, allow_unused=True)[0] for i, si in enumerate(s)]).mean()

    grad_imm_sens_wrt_weight = [torch.autograd.grad(si, next(iter(D.parameters())).norm(2), retain_graph=True, create_graph=True, allow_unused=True) for si in s]
    grad_imm_sens_wrt_weight = np.array([np.array([0 if g is None else g.norm(2).cpu().item() for g in grads]).sum() for grads in grad_imm_sens_wrt_weight]).sum()
    print(grad_imm_sens_wrt_weight)

    return grad_imm_sens_wrt_input

def gen_y(size):
    if train_opt.conditional:
        if train_opt.n_classes < 3:
            label1_prob = 0.5
            if train_opt.dataset == "CelebA":
                label1_prob = dataset.label_true_count / train_opt.train_set_size
            return (torch.empty((size)).random_(0, 2) < label1_prob).long()
        else:
            return torch.empty((size)).random_(0, train_opt.n_classes).long()

shape = (opt.batch_size, 1 if train_opt.dataset == "MNIST" else 3, train_opt.im_size, train_opt.im_size)
in_labels = gen_y(opt.batch_size).to(opt.device)
negate_mask = torch.ones(opt.batch_size, device=opt.device)
negate_mask[:opt.batch_size//2+1] *= -1

in_white = torch.ones(shape, device=opt.device).clamp(min=1e-5, max=1-1e-5)
in_white.requires_grad_()
out_white, _ = D(in_white, in_labels)
loss = (out_white * negate_mask).mean()
loss.backward(retain_graph=True)
print("White sens:", get_imm_sens(loss, in_white))
util.zero_grad(D)

in_black = (torch.zeros(shape, device=opt.device) - (1 if train_opt.dataset == "CelebA" else 0)).clamp(min=1e-5, max=1-1e-5)
in_black.requires_grad_()
out_black, _ = D(in_black, in_labels)
loss = (out_black * negate_mask).mean()
loss.backward(retain_graph=True)
print("Black sens:", get_imm_sens(loss, in_black))
util.zero_grad(D)

sensitivities = []
for mean in np.arange(0, 1, 0.1):
    for std in [1e-3, 0.01, 0.1, 0.2, 0.5]:
        inp = torch.empty(shape, device=opt.device).normal_(mean, std).clamp(min=1e-5, max=1-1e-5)
        inp.requires_grad_()
        out, _ = D(inp, in_labels)
        loss = (out * negate_mask).mean()
        imm_sens = get_imm_sens(loss, inp)
        sensitivities.append(imm_sens)

sensitivities = np.array(sensitivities)
print("\nNoise stats:")
print("Mean:", sensitivities.mean())
print("Std:", sensitivities.std())
print("Min:", sensitivities.min())
print("Max:", sensitivities.max())
print("Max-Min:", sensitivities.max() - sensitivities.min())


sensitivities = []
for i, (batch_x, batch_y) in enumerate(dl):
    batch_x = batch_x.clamp(min=1e-5, max=1-1e-5)
    batch_x.requires_grad_()
    out, _ = D(batch_x.to(opt.device), batch_y.to(opt.device))
    loss = (out * negate_mask).mean()
    loss.backward(retain_graph=True)

    imm_sens = get_imm_sens(loss, batch_x)
    sensitivities.append(imm_sens)
    util.zero_grad(D)

    if i >= opt.num_iter - 1:
        break

sensitivities = np.array(sensitivities)

print("\nBatch stats:")
print("Mean:", sensitivities.mean())
print("Std:", sensitivities.std())
print("Min:", sensitivities.min())
print("Max:", sensitivities.max())
print("Max-Min:", sensitivities.max() - sensitivities.min())
