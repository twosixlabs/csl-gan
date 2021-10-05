import numpy as np
import torch
import random, os, argparse, csv, json, glob, shutil
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision
from torch import autograd

import opacus
from opacus import PrivacyEngine
from opacus import ISPrivacyEngine

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
dataset, dataloader, public_dataset, public_dataloader = util.init_data(opt)

if opt.num_mean_samples > 0:
    print("Generating mean samples...")
    batch_size_backup = opt.batch_size
    opt.batch_size = opt.mean_sample_size
    mean_dataloader = util.init_data(opt)[1]
    opt.batch_size = batch_size_backup
    mean_sampler = MeanSampler(
        dataloader=mean_dataloader,
        save_path=opt.output_dir + "mean_samples/",
        noise_std=opt.mean_sample_noise_std,
        num_samples=opt.num_mean_samples,
        mean_size=opt.mean_sample_size,
        default_batch_size=opt.batch_size
    )
    mean_sample_privacy_cost, _ = mean_sampler.get_privacy_cost(target_delta=opt.delta)
    print("Privacy Cost from Mean Samples:", mean_sample_privacy_cost)
else:
    mean_sample_privacy_cost = 0

def init_optimizers():
    return optim.Adam(G.parameters(), lr=opt.g_lr, betas=(opt.adam_b1, opt.adam_b2)), optim.Adam(D.parameters(), lr=opt.d_lr, betas=(opt.adam_b1, opt.adam_b2))
g_optimizer, d_optimizer = init_optimizers()

start_epoch = 0
if opt.resume_epochs > 0:
    util.load_model(opt.resume_path + "saves/G-" + str(opt.resume_epochs), G, g_optimizer)
    start_epoch = util.load_model(opt.resume_path + "saves/D-" + str(opt.resume_epochs), D, d_optimizer)

privacy_engine = None
def setup_privacy_engine():
    privacy_params = {
        "sample_rate": opt.batch_size/opt.train_set_size,
        "alphas": [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), # Recommended by Opacus
        "noise_multiplier": opt.sigma,
    }
    if opt.use_imm_sens:
        privacy_engine = ISPrivacyEngine(
            D, **privacy_params, per_param=opt.imm_sens_per_param,
            scaling_vec=None if opt.imm_sens_scaling_mode is None else opt.imm_sens_scaling_vec
        )
    elif opt.use_grad_clip:
        privacy_engine = PrivacyEngine(
            D, **privacy_params,
            accum_passes=not opt.grad_clip_split,
            num_private_passes=1 if opt.grad_clip_split else None,
            auto_clip_and_accum_on_step=False,
            max_grad_norm=opt.clipping_param if opt.grad_clip_mode == "standard" else opt.clipping_param_per_layer,
            use_moving_avg_mgn=opt.grad_clip_mode == "moving-avg-pl",
            moving_avg_mgn_beta=opt.moving_avg_beta,
            moving_avg_mgn_target=opt.moving_avg_target
        )
        privacy_engine.disable_hooks()

    privacy_engine.attach(d_optimizer)
    privacy_engine._set_seed(opt.manual_seed)

    return privacy_engine


# # # # # # # #
#  Utilities  #
# # # # # # # #

def profiler_trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=opt.n_classes)
    print(output)
    #p.export_chrome_trace("trace_" + str(p.step_num) + ".json")

def gen_z_y(size):
    z = torch.empty((size, opt.g_latent_dim), device=opt.g_device).normal_(0.0, 1.0)
    y = F.one_hot(torch.empty((size)).random_(0, opt.n_classes).long(), num_classes=opt.n_classes) if opt.conditional else None
    return z, y

def eval_G_D(z, y=None):
    if opt.g_device == opt.d_device or opt.batch_split_size < opt.batch_size * 2:
        img = G(z, None if y is None else y.to(opt.g_device)).to(opt.d_device)
        return D(img.to(opt.d_device), None if y is None else y.to(opt.g_device)), img
    else:
        # Split batch to run G and D concurrently
        z_split = torch.split(z, opt.batch_split_size)
        y_split = None if y is None else torch.split(y, opt.batch_split_size)

        G_last = G(z_split[0], None if y is None else y_split[0].to(opt.d_device)).to(opt.d_device)
        ret_img = []
        ret_out = []
        for i in range(1, len(z_split)):
            ret_img.append(G_last)
            ret_out.append(D(G_last, None if y is None else y_split[i].to(opt.d_device)))
            G_last = G(z_split[i], None if y is None else y_split[i].to(opt.g_device)).to(opt.d_device)

        ret_img.append(G_last)
        ret_out.append(D(G_last, None if y is None else y_split[-1].to(opt.d_device)))

        return torch.cat(ret_out), torch.cat(ret_img)

def update_adaptive_clipping_params():
    # NOTE: For some reason this is giving values about 1/10 what they should be

    img, labels = None, None
    if opt.public_set_size > 0:
        img, labels = next(iter(public_dataloader)) # TO-DO: Update to use batch_size from real images so that numbers are more accurate
        img = torch.tensor(img)
        labels = labels if opt.conditional else None
    else:
        img = mean_sampler.sample(opt.batch_size)
        labels = gen_z_y(opt.batch_size)[1]

    if opt.grad_clip_split:
        d_fake_loss = 0
    else:
        d_fake, fake_img = eval_G_D(*gen_z_y(opt.batch_size))
        fake_img = fake_img.detach()
        d_fake_loss = D.fake_loss(d_fake, opt.d_device)

    d_real = D(img.to(opt.d_device), None if labels is None else labels.to(opt.d_device))
    d_real_loss = D.real_loss(d_real, opt.d_device)

    d_loss = d_real_loss + d_fake_loss

    d_loss.backward()

    with torch.no_grad():
        r = []
        for p in D.parameters():
            gn = p.grad_sample[0].view(opt.batch_size, -1).norm(2, dim=1)

            if opt.adaptive_stat == "mean":
                r.append(gn.mean().cpu().item())
            elif opt.adaptive_stat == "max":
                r.append(gn.max().cpu().item())

        privacy_engine.set_max_grad_norm([x * opt.adaptive_scalar[i] for i, x in enumerate(r)])

    d_optimizer.zero_grad()

def update_sens_moving_avg():
    vec = privacy_engine.scaling_vec
    privacy_engine.set_scaling_vec([vec[i] * opt.moving_avg_beta + p.grad.reshape(-1).norm(2).cpu().item()*(1-opt.moving_avg_beta) for i, p in enumerate(D.parameters())])


# # # # # # # # # # #
#   Set up logging  #
# # # # # # # # # # #

fixed_z, fixed_y = gen_z_y(opt.sample_num)
if opt.conditional:
    fixed_y = F.one_hot(torch.cat([torch.arange(opt.n_classes) for _ in range(opt.sample_num//opt.n_classes)]), num_classes=opt.n_classes).to(opt.g_device)

logger = Logger(
    "G Loss: {:4.4f} | D Loss: {:4.4f} (Real: {:4.4f} / {:3.1f}%, Fake: {:4.4f} / {:3.1f}%, Penalty: {:4.4f})" + ("\n=== Grad Norms ===\nMean Per Layer: {}\nStd Per Layer: {}\nMax Per Layer: {}\nClipping Params: {}\nGrads Clipped: {}" if opt.use_grad_clip else ""),
    ["G Loss", "D Loss", "D Real Loss", "D Real Acc", "D Fake Loss", "D Fake Acc", "D Penalty"] + (["D Layer Grad Norm Means", "D Layer Grad Norm Stds", "D Layer Grad Norm Maxes", "Clipping Params", "Grads Clipped"] if opt.use_grad_clip else []),
    opt.log_every / opt.batch_size,
    opt.output_dir + "log.csv"
)
np.set_printoptions(precision=4, suppress=True, linewidth=999999)

if opt.use_dp:
    privacy_log = open(opt.output_dir + "privacy_log.csv", "a")
    privacy_logger = csv.writer(privacy_log)
    if opt.resume_path is None:
        privacy_logger.writerow(["Epoch", "Epsilon"])
        privacy_log.flush()

batches_per_epoch = opt.train_set_size / opt.batch_size

def log(epoch, epoch_progress, print_dp=False):
    logger.log(epoch, epoch_progress)
    if print_dp and privacy_engine.steps > 0:
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

def update_grad_logging():
    grad_norms = []
    for p in D.parameters():
        grad_norms.append([p.grad_sample[1,j].view(-1).norm(2).cpu().item() for j in range(p.grad_sample.size(1))])
    grad_norms = np.array(grad_norms)
    logger.stats["D Layer Grad Norm Means"] += grad_norms.mean(axis=1)
    logger.stats["D Layer Grad Norm Stds"] += grad_norms.std(axis=1)
    logger.stats["D Layer Grad Norm Maxes"] += grad_norms.max(axis=1)
    logger.stats["Clipping Params"] += np.array(privacy_engine.max_grad_norm)

    # Log clipping rates for real loss
    all_norms = opacus.utils.tensor_utils.calc_sample_norms(
        named_params=privacy_engine.clipper._named_grad_samples(),
        flat=not privacy_engine.clipper.norm_clipper.is_per_layer,
    )
    clipping_factors = privacy_engine.clipper.norm_clipper.calc_clipping_factors(all_norms)
    grads_clipped = []
    for cf in clipping_factors:
        grads_clipped.append((cf[1].reshape(-1).cpu().numpy() < 0.99).mean())
    logger.stats["Grads Clipped"] += np.array(grads_clipped)


# # # # # # # # # # # # #
#   Training functions  #
# # # # # # # # # # # # #

def train_D(img, labels, z, y, use_dp=False):
    d_optimizer.zero_grad()
    util.freeze(G)
    batch_size = img.size(0)

    use_grad_clip = opt.use_grad_clip and use_dp
    use_imm_sens = opt.use_imm_sens and use_dp

    if use_imm_sens:
        img.requires_grad = True
        if not labels is None:
            labels.requires_grad = True
        if opt.imm_sens_scaling_mode == "adaptive-pl":
            update_adaptive_is_scaling()

    if use_grad_clip:
        privacy_engine.enable_hooks()
        if opt.grad_clip_mode == "adaptive-pl":
            update_adaptive_clipping_params()

    d_fake, fake_img = eval_G_D(z, y)
    fake_img = fake_img.detach()
    d_fake_loss = D.fake_loss(d_fake, opt.d_device)

    d_real = D(img, labels)
    d_real_loss = D.real_loss(d_real, opt.d_device)

    d_loss = d_real_loss + d_fake_loss

    if use_grad_clip:
        d_loss.backward()

        with torch.no_grad():
            update_grad_logging()

        privacy_engine.disable_hooks()
        privacy_engine.clip()

        if opt.grad_clip_split:
            privacy_engine.accum_grads_across_passes()

    penalty = torch.tensor(0)
    if len(opt.penalty) > 0:
        # Calculate penalties

        # Set "real data" in penalty to actual real data, mean samples, or public data depending on configuration
        penalty_real_data = img
        if opt.penalty_use_public_data:
            if opt.public_set_size > 0:
                dl = iter(public_dataloader)
                penalty_real_data = torch.cat([next(dl)[0] for _ in range((batch_size-1)//opt.batch_size + 1)], dim=0)[:batch_size]
            elif opt.num_mean_samples > 0:
                penalty_real_data = mean_sampler.sample(batch_size)
        penalty_real_data = penalty_real_data.to(opt.d_device)

        if use_grad_clip:
            # Using per-sample grads so penalty must be added to summed_grad manually

            if opt.penalty_use_public_data:
                # Penalty uses mean samples so does not need to be calculated per-sample
                privacy_engine.accumulate_batch()

                penalty = calc_penalty(D, opt.penalty, penalty_real_data, fake_img, device=opt.d_device)
                d_loss += penalty

                # Manually calculate gradient and add to summed_grad (where accumulated per-sample grads are stored)
                penalty_grad = autograd.grad(penalty, D.parameters(), create_graph=False, retain_graph=False, allow_unused=True)
                with torch.no_grad():
                    for j, p in enumerate(D.parameters()):
                        p.summed_grad += 0 if penalty_grad[j] is None else penalty_grad[j] * opt.batch_size # multiply by batch size because summed_grad is sum and not mean

            else:
                # TO-DO: Per-sample penalty causes memory leak

                # Penalty accesses sensitive data and must be calculated per-sample (which is slow)
                penalties = calc_penalty(D, opt.penalty, penalty_real_data, fake_img, device=opt.d_device, per_sample=True)
                penalty = penalties.mean(dim=0)
                d_loss += penalty

                for i in range(len(penalties)):
                    penalty_grad = autograd.grad(penalties[i], D.parameters(), create_graph=True, retain_graph=True, allow_unused=True)
                    with torch.no_grad():
                        for j, p in enumerate(D.parameters()):
                            p.grad_sample[0,i] += 0 if penalty_grad[j] is None else penalty_grad[j]

                privacy_engine.clip()
                privacy_engine.accumulate_batch()
        else:
            # Not grad clipping, can just call backward on penalty normally
            penalty = calc_penalty(D, opt.penalty, penalty_real_data, fake_img, device=opt.d_device)
            d_loss += penalty

            if use_imm_sens:
                privacy_engine.backward(d_loss, img if labels is None else (img, labels))
                if opt.imm_sens_scaling_mode == "moving-avg-pl":
                    update_sens_moving_avg()
            else:
                d_loss.backward()

    else:
        # No penalties
        if use_grad_clip:
            privacy_engine.accumulate_batch()
        elif use_imm_sens:
            privacy_engine.backward(d_loss, img if labels is None else (img, labels))
            if opt.imm_sens_scaling_mode == "moving-avg-pl":
                update_sens_moving_avg()
        else:
            d_loss.backward()

    d_optimizer.step()
    util.unfreeze(G)

    # Update discriminator logging
    logger.stats["D Loss"] += d_loss.item()
    logger.stats["D Real Loss"] += d_real_loss.item()
    logger.stats["D Real Acc"] += 100*np.mean(d_real.detach().cpu().numpy() > 0)
    logger.stats["D Fake Loss"] += d_fake_loss.item()
    logger.stats["D Fake Acc"] += 100*np.mean(d_fake.detach().cpu().numpy() < 0)
    logger.stats["D Penalty"] += penalty.item()

def train_G(z, y):
    g_optimizer.zero_grad()
    util.freeze(D)

    d_fake, _ = eval_G_D(z, y)
    g_loss = G.loss(d_fake, opt.d_device)

    g_loss.backward()
    g_optimizer.step()
    util.unfreeze(D)

    # Update generator logging
    logger.stats["G Loss"] += g_loss.item() * opt.n_d_steps  # Multiply by n_d_steps for consistency with logging

def train(epoch, batch_i, real_images_batch, real_labels_batch, use_dp=False):
    real_images_batch = real_images_batch.to(opt.d_device)
    real_labels_batch = F.one_hot(real_labels_batch.to(opt.d_device), num_classes=opt.n_classes).float() if opt.conditional else None
    batch_size = real_images_batch.size(0)

    # Train discriminator
    train_D(real_images_batch, real_labels_batch, *gen_z_y(batch_size), use_dp=use_dp)

    # Train generator
    if batch_i % opt.n_d_steps == 0: # only train generator every n_d_steps iterations
        train_G(*gen_z_y(batch_size))

    # Logging and profiling
    if opt.profile_training:
        profiler.step()
    if ((batch_i+1) * opt.batch_size) % opt.log_every == 0:
        log(epoch, 100*batch_i / batches_per_epoch, print_dp=use_dp)
    if ((batch_i+1) * opt.batch_size) % opt.sample_every == 0:
        sample(epoch, batch_i)


# # # # # # # # # #
#  Training loop  #
# # # # # # # # # #

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
    logger.reset_stats()

    # Warmup on public data or mean samples
    for it in range(opt.warmup_iter):
        img, labels = next(iter(public_dataloader)) if opt.public_set_size > 0 else (mean_sampler.sample(opt.batch_size),gen_z_y(opt.batch_size)[1])
        train(-1, it, img, labels, use_dp=False)

    # Reset optimizers and switch to DP-SGD
    g_optimizer, d_optimizer = init_optimizers()
    if opt.use_dp:
        privacy_engine = setup_privacy_engine()

    # Train on dataset
    for epoch in range(opt.resume_epochs, opt.n_epochs):
        logger.reset_stats()
        for batch_i, (real_images_batch, real_labels_batch) in enumerate(dataloader):
            train(epoch, batch_i, real_images_batch, real_labels_batch, use_dp=opt.use_dp)

        if opt.log_every_epochs > 0 and (epoch+1) % opt.log_every_epochs == 0:
            log(epoch, 100)
        if opt.sample_every_epochs > 0 and (epoch+1) % opt.sample_every_epochs == 0:
            sample(epoch, batch_i)

        if opt.use_dp:
            eps, _ = privacy_engine.get_privacy_spent(opt.delta)
            privacy_logger.writerow([epoch, eps + mean_sample_privacy_cost])
            privacy_log.flush()

        if opt.use_dp and not opt.epsilon_budget is None and eps > opt.epsilon_budget:
            break

        if (epoch+1) % opt.save_every == 0:
            util.save_model(epoch, D, d_optimizer, 0, opt.output_dir+"saves/D-"+str(epoch+1))
            util.save_model(epoch, G, g_optimizer, 0, opt.output_dir+"saves/G-"+str(epoch+1))


print("Finished training.")
util.save_model(opt.n_epochs, D, d_optimizer, 0, opt.output_dir+"saves/D-"+str(epoch+1))
util.save_model(opt.n_epochs, G, g_optimizer, 0, opt.output_dir+"saves/G-"+str(epoch+1))
logger.close()
