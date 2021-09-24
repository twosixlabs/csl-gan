import numpy as np
import torch
import random, os, argparse, csv, json, glob, shutil
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision
from torch import autograd
from opacus import PrivacyEngine

from logger import *
from mean_sampler import MeanSampler
from gradient_penalty import *
import util
import options


# # # # # # # # # # # # # # # # # # # # # #
#  Parse arguments and configure options  #
# # # # # # # # # # # # # # # # # # # # # #

opt = options.parse()

# Save config to file
with open(opt.output_dir + "opt.txt", "w") as f:
    json.dump(opt.__dict__, f)

# Copy all code files to output directory
for file in glob.glob("*.py"):
    if os.path.isfile(file):
        shutil.copy2(file, opt.output_dir+"code/")


# # # # # # # # # # # # # # # #
#  Create dataset and models  #
# # # # # # # # # # # # # # # #

G, D = util.init_models(opt)
dataset, dataloader = util.init_data(opt)

g_optimizer = optim.Adam(G.parameters(), lr=opt.g_lr, betas=(opt.adam_b1, opt.adam_b2))
d_optimizer = optim.Adam(D.parameters(), lr=opt.d_lr, betas=(opt.adam_b1, opt.adam_b2))

start_epoch = 0
if opt.resume_epochs > 0:
    util.load_model(opt.resume_path + "saves/G-" + str(opt.resume_epochs), G, g_optimizer)
    start_epoch = util.load_model(opt.resume_path + "saves/D-" + str(opt.resume_epochs), D, d_optimizer)

if opt.use_dp:
    privacy_engine = PrivacyEngine(
        D,
        batch_size=opt.batch_size,
        sample_size=opt.train_set_size,
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), # Recommended by Opacus
        noise_multiplier=opt.sigma,
        max_grad_norm=opt.C,
        use_imm_sens=opt.use_imm_sens or opt.use_per_sample_imm_sens,
        accum_passes=not opt.use_split_grad_clip,
        per_sample=not opt.use_imm_sens,
        auto_clip_and_accum_on_step=False
    )
    privacy_engine.attach(d_optimizer)


# # # # # # # # # # # # # # #
#  Utilities for training   #
# # # # # # # # # # # # # # #

def profiler_trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=opt.n_classes)
    print(output)
    #p.export_chrome_trace("trace_" + str(p.step_num) + ".json")

def print_grad_info(model):
    for p in model.parameters():
        if hasattr(p, "grad"):
            print("None grad" if p.grad is None else p.grad.shape)
        print(p.__dict__.keys())

def gen_z(size):
    return torch.empty((size, opt.g_latent_dim), device=opt.g_device).normal_(0.0, 1.0), F.one_hot(torch.empty((size)).random_(0, opt.n_classes).long(), num_classes=opt.n_classes)

fixed_z, _ = gen_z(opt.sample_num)
fixed_y = F.one_hot(torch.cat([torch.arange(opt.n_classes) for _ in range(opt.sample_num//opt.n_classes)]), num_classes=opt.n_classes).to(opt.g_device)

def eval_G_D(z, y):
    if opt.g_device == opt.d_device or opt.batch_split_size < opt.batch_size * 2:
        img = G(z, y.to(opt.g_device)).to(opt.d_device)
        return D(img.to(opt.d_device), y.to(opt.d_device)), img
    else:
        # Split batch to run G and D concurrently
        z_split = torch.split(z, opt.batch_split_size)
        y_split = torch.split(y, opt.batch_split_size)

        G_last = G(z_split[0], y_split[0].to(opt.d_device)).to(opt.d_device)
        ret_img = []
        ret_out = []
        for i in range(1, len(z_split)):
            ret_img.append(G_last)
            ret_out.append(D(G_last, y_split[i].to(opt.d_device)))
            G_last = G(z_split[i], y_split[i].to(opt.g_device)).to(opt.d_device)

        ret_img.append(G_last)
        ret_out.append(D(G_last, y_split[-1].to(opt.d_device)))

        return torch.cat(ret_out), torch.cat(ret_img)

logger = Logger(
    "G Loss: {:4.4f} | D Loss: {:4.4f} (Real: {:4.4f} / {:3.1f}%, Fake: {:4.4f} / {:3.1f}%, Penalty: {:4.4f})",
    ["G Loss", "D Loss", "D Real Loss", "D Real Acc", "D Fake Loss", "D Fake Acc", "D Penalty"],
    opt.log_every / opt.batch_size,
    opt.output_dir + "log.csv"
)

if opt.use_dp:
    privacy_log = open(opt.output_dir + "privacy_log.csv", "a")
    privacy_logger = csv.writer(privacy_log)
    if opt.resume_path is None:
        privacy_logger.writerow(["Epoch", "Epsilon"])
        privacy_log.flush()

batches_per_epoch = opt.train_set_size / opt.batch_size

def log(epoch, epoch_progress):
    logger.log(epoch, epoch_progress)
    if opt.use_dp:
        epsilon, best_alpha = privacy_engine.get_privacy_spent(opt.delta)
        print("({}, {})-DP for alpha={}".format(epsilon, opt.delta, best_alpha))

def sample(epoch, batch):
    G.eval()

    with torch.no_grad():
        fake_images = G(fixed_z, fixed_y).to("cpu")
        if opt.dataset == "CelebA":
            fake_images = util.denorm_celeba(fake_images)
        torchvision.utils.save_image(fake_images.data, os.path.join(opt.output_dir+"samples/", "%d-%d.png")%(
    epoch+1, batch), nrow=opt.n_classes)

    G.train()


# # # # # # # # # #
#  Training loop  #
# # # # # # # # # #

def train_D(img, labels, z, y):
    d_optimizer.zero_grad()
    util.freeze(G)
    batch_size = img.size(0)

    if opt.use_imm_sens:
        img.requires_grad = True
        labels.requires_grad = True

    d_fake, fake_img = eval_G_D(z, y)
    fake_img = fake_img.detach()
    d_fake_loss = D.fake_loss(d_fake)

    d_real = D(img, labels)
    d_real_loss = D.real_loss(d_real)

    d_loss = d_real_loss + d_fake_loss

    if opt.use_imm_sens:
        privacy_engine.calc_sens(d_real_loss, img if opt.unconditional else (img, labels))

    penalty = torch.tensor(0)
    if len(opt.penalty) > 0:
        penalty_real_data = (None if opt.penalty_use_mean_samples else img).to(opt.d_device)
        penalty_per_sample = opt.use_dp_per_sample and not opt.penalty_use_mean_samples

        if opt.use_dp:
            if not opt.use_dp_per_sample:
                d_loss.backward()

            privacy_engine.disable_hooks()
            privacy_engine.clip()
            if opt.use_split_grad_clip:
                privacy_engine.accum_grads_across_passes()
            if not penalty_per_sample:
                privacy_engine.accumulate_batch()

        if penalty_per_sample:
            # Penalty accesses sensitive data and must be calculated per-sample (which is slow)
            penalties = calc_penalty(D, opt.penalty, penalty_real_data, fake_img, device=opt.d_device, per_sample=True)
            penalty = penalties.mean(dim=0)
            d_loss += penalty

            for i in range(len(penalties)):
                penalty_grad = autograd.grad(penalties[i], D.parameters(), create_graph=True, retain_graph=True, allow_unused=True)
                with torch.no_grad():
                    for j, p in enumerate(D.parameters()):
                        p.grad_sample[0,i,j] += 0 if penalty_grad[j] is None else penalty_grad[j]

            privacy_engine.clip()
            privacy_engine.accumulate_batch()
        else:
            penalty = calc_penalty(D, opt.penalty, penalty_real_data, fake_img, device=opt.d_device)
            d_loss += penalty

            if opt.use_dp_per_sample:
                penalty_grad = batch_size * autograd.grad(penalty, D.parameters(), create_graph=False, retain_graph=False, allow_unused=True)
                with torch.no_grad():
                    for j, p in enumerate(D.parameters()):
                        p.summed_grad[j] += 0 if penalty_grad[j] is None else penalty_grad[j]
            else:
                d_loss.backward()

        if opt.use_dp:
            privacy_engine.enable_hooks()

    d_optimizer.step()
    util.unfreeze(G)

    # Update discriminator logging
    logger.stats["D Loss"] += d_loss.item()
    logger.stats["D Real Loss"] += d_real_loss.item()
    logger.stats["D Real Acc"] += 100*np.mean(d_real.detach().cpu().numpy() > 0)
    logger.stats["D Fake Loss"] += d_fake_loss.item()
    logger.stats["D Fake Acc"] += 100*np.mean(d_fake.detach().cpu().numpy() < 0)
    logger.stats["D Penalty"] += penalty.item()

def train_G(z, x):
    g_optimizer.zero_grad()
    util.freeze(D)

    d_fake, _ = eval_G_D(*gen_z(opt.batch_size))
    g_loss = G.loss(d_fake)

    g_loss.backward()
    g_optimizer.step()
    util.unfreeze(D)

    # Update generator logging
    logger.stats["G Loss"] += g_loss.item() * opt.n_d_steps  # Multiply by n_d_steps for consistency with logging

print("Starting training...")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if opt.profile_training else [],
    schedule=torch.profiler.schedule(
        wait=1 if opt.profile_training else 99999,
        warmup=1,
        active=5
    ),
    on_trace_ready=profiler_trace_handler
) as profiler:
    for epoch in range(opt.resume_epochs, opt.n_epochs):
        logger.reset_stats()
        for batch_i, (real_images_batch, real_labels_batch) in enumerate(dataloader):
            real_images_batch = real_images_batch.to(opt.d_device)
            real_labels_batch = F.one_hot(real_labels_batch.to(opt.d_device), num_classes=opt.n_classes).float()
            batch_size = real_images_batch.size(0)

            if opt.use_imm_sens:
                real_images_batch.requires_grad = True
                real_labels_batch.requires_grad = True

            # Train discriminator
            train_D(real_images_batch, real_labels_batch, *gen_z(batch_size))

            # Train generator
            if batch_i % opt.n_d_steps == 0: # only train generator every n_d_steps iterations
                train_G(*gen_z(batch_size))

            # Logging and profiling
            if opt.profile_training:
                profiler.step()
            if ((batch_i+1) * opt.batch_size) % opt.log_every == 0:
                log(epoch, 100*batch_i / batches_per_epoch)
            if ((batch_i+1) * opt.batch_size) % opt.sample_every == 0:
                sample(epoch, batch_i)

        if opt.log_every_epochs > 0 and (epoch+1) % opt.log_every_epochs == 0:
            log(epoch, 100)
        if opt.sample_every_epochs > 0 and (epoch+1) % opt.sample_every_epochs == 0:
            sample(epoch, batch_i)

        if opt.use_dp:
            eps, _ = privacy_engine.get_privacy_spent(opt.delta)
            privacy_logger.writerow([epoch, eps])
            privacy_log.flush()

        if opt.use_dp and not opt.epsilon_budget is None and eps > opt.epsilon_budget:
            break

        if (epoch+1) % opt.save_every == 0:
            util.save_model(epoch, D, d_optimizer, d_loss, opt.output_dir+"saves/D-"+str(epoch+1))
            util.save_model(epoch, G, g_optimizer, g_loss, opt.output_dir+"saves/G-"+str(epoch+1))


print("Finished training.")
util.save_model(opt.n_epochs, D, d_optimizer, d_loss, opt.output_dir+"saves/D-"+str(epoch+1))
util.save_model(opt.n_epochs, G, g_optimizer, g_loss, opt.output_dir+"saves/G-"+str(epoch+1))
logger.close()
