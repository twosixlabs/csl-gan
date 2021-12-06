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
from opacus import PrivacyEngine, ISPrivacyEngine, TMPrivacyEngine, SVPrivacyEngine

from logger import *
from mean_sampler import MeanSampler
from gradient_penalty import *
from prop_grad_clip import *
import util
import init_util
import options


torch.backends.cudnn.benchmark = True

# # # # # # # # # # # # # # # # # # # # # #
#  Parse arguments and configure options  #
# # # # # # # # # # # # # # # # # # # # # #

opt = options.parse()

# Save config to file
with open(opt.output_dir + "opt.txt", "w") as f:
    json.dump(opt.__dict__, f)

# Copy all code files to output directory
if opt.resume_path is None:
    for file in glob.glob("*.py"):
        if os.path.isfile(file):
            shutil.copy2(file, opt.output_dir+"code/")


# # # # # # # # # # # # # # # #
#  Create dataset and models  #
# # # # # # # # # # # # # # # #

G, D = init_util.init_models(opt)
dataset, dataloader, public_dataset, public_dataloader = init_util.init_data(opt)

if opt.num_mean_samples > 0:
    print("Generating mean samples...")
    batch_size_backup = opt.batch_size
    opt.batch_size = opt.mean_sample_size * (opt.n_classes if opt.conditional else 1)
    mean_dataloader = init_util.init_data(opt)[1]
    opt.batch_size = batch_size_backup
    mean_sampler = MeanSampler(
        dataloader=mean_dataloader,
        save_path=opt.output_dir + "mean_samples/",
        noise_std=opt.mean_sample_noise_std,
        num_samples=opt.num_mean_samples,
        mean_size=opt.mean_sample_size,
        default_batch_size=opt.batch_size,
        n_classes = opt.n_classes if opt.conditional else 1,
        smallest_class_size = (min(dataset.label_true_count, opt.train_set_size - dataset.label_true_count) if opt.dataset == "CelebA" else opt.train_set_size / opt.n_classes) if opt.conditional else None
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

if opt.clip_propogating_grads:
    with torch.no_grad():
        p = (opt.prop_grad_clip_param_pl, opt.forward_input_clip_param_pl) if opt.grad_clip_mode[-3:] == "-pl" else (opt.prop_grad_clip_param, opt.forward_input_clip_param)
        prop_grad_clipper = PropogatingGradClipper(D, *p, opt.pgc_auto_activation_scale, opt.pgc_auto_weight_grad_scale, device=opt.d_device)

        clip_params = [c * opt.batch_size for c in prop_grad_clipper.grad_clip_params] # for mean loss reduction, to be compatible with opacus

        opt.clipping_param_per_layer =  clip_params
        opt.clipping_param = np.linalg.norm(clip_params, ord=2)

privacy_engine = None
def setup_privacy_engine():
    privacy_params = {
        "batch_size": opt.batch_size,
        "sample_size": opt.train_set_size,
        "alphas": [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), # Recommended by Opacus
        "noise_multiplier": opt.sigma,
    }
    if opt.dp_mode == "is":
        privacy_engine = ISPrivacyEngine(
            D, **privacy_params,
            per_param=opt.imm_sens_per_param,
            scaling_vec=None if opt.imm_sens_scaling_mode == "standard" else opt.imm_sens_scaling_vec
        )
    elif opt.dp_mode == "gc":
        opt.clipping_param_per_layer = [1 for _ in range(len(list(D.parameters())))] if opt.clipping_param_per_layer is None else opt.clipping_param_per_layer
        privacy_engine = PrivacyEngine(
            D, **privacy_params,
            accum_passes=not opt.grad_clip_split,
            num_private_passes=1 if opt.grad_clip_split else None,
            auto_clip_and_accum_on_step=False,
            max_grad_norm=opt.clipping_param_per_layer if opt.grad_clip_mode[-3:] == "-pl" else opt.clipping_param
        )
        privacy_engine.disable_hooks()
    elif opt.dp_mode == "tm":
        privacy_engine = TMPrivacyEngine(
            D, **privacy_params,
            smooth_sens_= opt.smooth_sens_t,
            m_trim = opt.tm_m,
            min_val = opt.tm_max_val,
            max_val = opt.tm_min_val,
            sens_compute_bs = opt.batch_size * 2 if opt.tm_sens_compute_bs is None else opt.tm_sens_compute_bs,
            rho_per_epoch = opt.tm_rho_per_epoch
        )
    elif opt.dp_mode == "sv":
        privacy_engine = SVPrivacyEngine(
            D, **privacy_params,
            smooth_sens_= opt.smooth_sens_t,
            rho_per_epoch = opt.tm_rho_per_epoch
        )

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

def gen_z(size):
    return torch.empty((size, opt.g_latent_dim), device=opt.g_device).normal_(0.0, 1.0)

def gen_y(size):
    if opt.conditional:
        if opt.n_classes < 3:
            label1_prob = 0.5
            if opt.dataset == "CelebA":
                label1_prob = dataset.label_true_count / opt.train_set_size
            return (torch.empty((size)).random_(0, 2) < label1_prob).long()
        else:
            return torch.empty((size)).random_(0, opt.n_classes).long()

def eval_G_D(z, y=None, g_kwarg={}, d_kwarg={}):
    if opt.g_device == opt.d_device or opt.batch_split_size < opt.batch_size * 2:
        img = G(z, None if y is None else y.to(opt.g_device), **g_kwarg).to(opt.d_device)
        d_out, d_aux_out = D(img.to(opt.d_device), None if y is None else y.to(opt.g_device), **d_kwarg)
        return d_out, d_aux_out, img
    else:
        # Split batch to run G and D concurrently
        z_split = torch.split(z, opt.batch_split_size)
        y_split = None if y is None else torch.split(y, opt.batch_split_size)

        G_last = G(z_split[0], None if y is None else y_split[0].to(opt.d_device), **g_kwarg).to(opt.d_device)
        ret_img = []
        ret_out = []
        for i in range(1, len(z_split)):
            ret_img.append(G_last)
            ret_out.append(D(G_last, None if y is None else y_split[i].to(opt.d_device), **d_kwarg))
            G_last = G(z_split[i], None if y is None else y_split[i].to(opt.g_device), **g_kwarg).to(opt.d_device)

        ret_img.append(G_last)
        ret_out.append(D(G_last, None if y is None else y_split[-1].to(opt.d_device), **d_kwarg))

        return torch.cat(o for o,_ in ret_out), torch.cat(a for _,a in ret_out) if opt.is_acgan else None, torch.cat(ret_img)

def get_penalty_data(data_in, labels_in):
    data = data_in
    labels = labels_in
    batch_size = data_in.size(0)

    if opt.penalty_use_public_data:
        if opt.public_set_size > 0:
            if labels_in is None:
                data = torch.cat([next(iter(public_dataloader))[0] for _ in range((batch_size-1)//opt.batch_size + 1)], dim=0)[:batch_size]
            else:
                data, labels = zip(*[public_dataset.get_item_with_label(l) for l in labels_in])
                data = torch.stack(data)
                labels = torch.tensor(labels)
        elif opt.num_mean_samples > 0:
            data, labels = mean_sampler.sample(batch_size, requested_labels=labels_in)

    return data.to(opt.d_device), None if labels is None else labels.to(opt.d_device)

def update_adaptive_clipping_params():
    img, labels = None, None
    util.zero_grad(D)

    if opt.public_set_size > 0:
        img, labels = next(iter(public_dataloader)) # could update to use sampled batch size for more accuracy and less efficiency
        img = torch.tensor(img)
        labels = torch.tensor(labels) if opt.conditional else None
    else:
        img, labels = mean_sampler.sample(opt.batch_size)

    img = img.to(opt.d_device)
    labels = None if labels is None else labels.to(opt.d_device)

    if opt.grad_clip_split:
        d_fake_loss = 0
        d_fake_aux_loss = 0
    else:
        d_fake, d_fake_aux, d_fake_loss, d_fake_loss_aux, fake_img = calc_d_fake_loss(img, labels, gen_z(opt.batch_size), labels)

    d_real, d_real_aux, d_real_loss, d_real_aux_loss = calc_d_real_loss(img, labels)

    d_loss = d_real_loss + d_fake_loss + d_real_aux_loss + d_fake_aux_loss
    d_loss.backward()

    with torch.no_grad():
        r = []
        for p in D.parameters():
            gn = p.grad_sample[0].view(opt.batch_size, -1).norm(2, dim=1)

            if opt.adaptive_stat == "mean":
                r.append(gn.mean().cpu().item())
            elif opt.adaptive_stat == "max":
                r.append(gn.max().cpu().item())

        if opt.use_grad_clip_per_layer:
            privacy_engine.set_max_grad_norm([x * opt.adaptive_scalar for i, x in enumerate(r)])
        else:
            privacy_engine.set_max_grad_norm(torch.tensor(r).norm(2) * opt.adaptive_scalar)

    d_optimizer.zero_grad()

def update_sens_moving_avg():
    vec = privacy_engine.scaling_vec
    privacy_engine.set_scaling_vec([vec[i] * opt.moving_avg_beta + p.grad.reshape(-1).norm(2).cpu().item()*(1-opt.moving_avg_beta) for i, p in enumerate(D.parameters())])


# # # # # # # # # # #
#   Set up logging  #
# # # # # # # # # # #

fixed_z = gen_z(opt.sample_num)
if opt.conditional:
    fixed_y = torch.cat([torch.arange(opt.n_classes) for _ in range(opt.sample_num//opt.n_classes)]).to(opt.g_device)
    fixed_z = fixed_z[:len(fixed_y)]
else:
    fixed_y = gen_y(opt.sample_num)

gc_log_str = "\n=== Grad Norms ===\nMean Per Layer: {}\nStd Per Layer: {}\nMax Per Layer: {}\nClipping Params: {}\nGrads Clipped: {}"
gc_log_stats = ["D Layer Grad Norm Means", "D Layer Grad Norm Stds", "D Layer Grad Norm Maxes", "Clipping Params", "Grads Clipped"]
is_log_str = "\nIS - Mean: {:4.8f} - Min: {:4.8f} - Max: {:4.8f}"
is_log_stats = ["IS Mean", "IS Min", "IS Max"]
logger = Logger(
    "G " + ("Adv " if opt.is_acgan else "") + "Loss: {:4.4f}" + (", G Aux: {:4.4f} / {:3.1f}%\n" if opt.is_acgan else " | ") +
        "D Adv Loss: {:4.4f} (Real: {:4.4f} / {:3.1f}%, Fake: {:4.4f} / {:3.1f}%" +
        (", Real Aux: {:4.4f} / {:3.1f}%" if opt.is_acgan else "") +
        (", Penalty: {:4.4f}" if len(opt.penalty) > 0 else "") + ")" +
        (gc_log_str if opt.dp_mode == "gc" else "") + (is_log_str if opt.dp_mode == "is" and opt.imm_sens_scaling_mode[-3:] != "-pl" else ""),
    ["G Adv Loss"] + (["G Aux Loss", "G Aux Acc"] if opt.is_acgan else []) +
        ["D Adv Loss", "D Real Loss", "D Real Acc", "D Fake Loss", "D Fake Acc"] +
        (["D Real Aux Loss", "D Real Aux Acc"] if opt.is_acgan else []) +
        (["D Penalty"] if len(opt.penalty) > 0 else []) +
        (gc_log_stats if opt.dp_mode == "gc" else []) +
        (is_log_stats if opt.dp_mode == "is" and opt.imm_sens_scaling_mode[-3:] != "-pl" else []),
    (opt.log_every_epochs * opt.train_set_size if opt.log_every_epochs > 0 else opt.log_every) // opt.batch_size,
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
    all_norms = opacus.utils.tensor_utils.calc_sample_norms(
        named_params=privacy_engine.clipper._named_grad_samples(),
        flat=not privacy_engine.clipper.norm_clipper.is_per_layer,
    )
    norms = torch.stack(all_norms).cpu().numpy()[:,1 if opt.grad_clip_split else 0]

    logger.stats["D Layer Grad Norm Means"] += norms.mean(axis=1)
    logger.stats["D Layer Grad Norm Stds"] += norms.std(axis=1)
    logger.stats["D Layer Grad Norm Maxes"] += norms.max(axis=1)

    logger.stats["Clipping Params"] += np.array(privacy_engine.max_grad_norm)

    # Log clipping rates for real loss
    clipping_factors = iter(privacy_engine.clipper.norm_clipper.calc_clipping_factors(all_norms))
    grads_clipped = []
    for _ in range(len(all_norms)):
        cf = next(clipping_factors)
        grads_clipped.append((cf[1 if opt.grad_clip_split else 0].reshape(-1).cpu().numpy() < 0.999).mean())
    logger.stats["Grads Clipped"] += np.array(grads_clipped)

def update_is_logging():
    logger.stats["IS Mean"] += privacy_engine.batch_sensitivity
    logger.stats["IS Min"] = min(99999 if logger.stats["IS Min"] < 1e-8 else logger.stats["IS Min"], privacy_engine.batch_sensitivity * logger.interval) # Scale by logger.interval so when logger divides it it is shown as the original value
    logger.stats["IS Max"] = max(logger.stats["IS Max"], privacy_engine.batch_sensitivity * logger.interval)


# # # # # # # # # # # # #
#   Training functions  #
# # # # # # # # # # # # #

def calc_d_fake_loss(img, labels, z, y):
    d_fake, d_fake_aux, fake_img = eval_G_D(z, y, d_kwarg={"aux": opt.d_fake_aux_loss})
    fake_img = fake_img.detach()
    d_fake_loss = D.fake_loss(d_fake, opt.d_device)
    d_fake_aux_loss = D.aux_loss(d_fake_aux, y.to(opt.d_device), opt.d_device) if opt.is_acgan and opt.d_fake_aux_loss else 0

    return d_fake, d_fake_aux, d_fake_loss, d_fake_aux_loss, fake_img

def calc_d_real_loss(img, labels):
    d_real, d_real_aux = D(img, labels)
    d_real_loss = D.real_loss(d_real, opt.d_device)
    d_real_aux_loss = D.aux_loss(d_real_aux, labels, opt.d_device) if opt.is_acgan else 0

    return d_real, d_real_aux, d_real_loss, d_real_aux_loss

def train_D(img, labels, z, y, use_dp=False):
    util.zero_grad(D)
    util.freeze(G)
    batch_size = img.size(0)

    use_grad_clip = opt.dp_mode == "gc" and use_dp
    use_imm_sens = opt.dp_mode == "is" and use_dp
    use_tm = opt.dp_mode == "tm" and use_dp
    use_sv = opt.dp_mode == "sv" and use_dp

    if opt.per_sample_grad and use_dp:
        privacy_engine.enable_hooks()
    if use_imm_sens:
        img.requires_grad = True
        if opt.imm_sens_scaling_mode == "adaptive-pl":
            update_adaptive_is_scaling()
    if use_grad_clip:
        if opt.grad_clip_mode[:8] == "adaptive":
            update_adaptive_clipping_params()

    d_fake, d_fake_aux, d_fake_loss, d_fake_aux_loss, fake_img = calc_d_fake_loss(img, labels, z, y)
    d_real, d_real_aux, d_real_loss, d_real_aux_loss = calc_d_real_loss(img, labels)
    d_loss = d_real_loss + d_fake_loss + d_real_aux_loss + d_fake_aux_loss

    if opt.per_sample_grad and use_dp:
        d_loss.backward()
        next(iter(D.parameters())).grad_sample.size(1)
        privacy_engine.disable_hooks()
    if opt.clip_propogating_grads and use_dp:
        if not opt.per_sample_grad:
            d_loss.backward()
        prop_grad_clipper.disable_hooks()

    if use_grad_clip:
        with torch.no_grad():
            update_grad_logging()

        privacy_engine.clip()

        if opt.grad_clip_split:
            privacy_engine.accum_grads_across_passes()

    penalty = torch.tensor(0)
    if len(opt.penalty) > 0:
        # Calculate penalties

        # Set "real data" in penalty to actual real data, mean samples, or public data depending on configuration
        penalty_real_data, penalty_real_labels = get_penalty_data(img, labels)

        if use_dp and opt.per_sample_grad:
            # Using per-sample grads so penalty must be added to summed_grad manually

            if opt.penalty_use_public_data:
                # Penalty uses mean samples so does not need to be calculated per-sample
                if use_grad_clip:
                    privacy_engine.accumulate_batch()
                elif use_tm:
                    privacy_engine.trim_grads()
                elif use_sv:
                    privacy_engine.vote_on_grads()

                penalty = calc_penalty(D, opt.penalty, penalty_real_data, penalty_real_labels, fake_img, y, device=opt.d_device, aux_penalty=opt.aux_penalty)
                d_loss += penalty

                # Manually calculate gradient and add to summed_grad (where accumulated per-sample grads are stored)
                penalty_grad = autograd.grad(penalty, D.parameters(), create_graph=False, retain_graph=False, allow_unused=True)
                with torch.no_grad():
                    for j, p in enumerate(D.parameters()):
                        if use_grad_clip:
                            p.summed_grad += 0 if penalty_grad[j] is None else penalty_grad[j] * opt.batch_size # multiply by batch size because summed_grad is sum and not mean
                        elif use_tm or use_sv:
                            p.grad += 0 if penalty_grad[j] is None else penalty_grad[j]
            else:
                # TO-DO: Per-sample penalty causes memory leak
                print("WARNING: Per sample penalty currently causes a memory leak.")

                # Penalty accesses sensitive data and must be calculated per-sample (which is slow)
                penalties = calc_penalty(D, opt.penalty, penalty_real_data, penalty_real_labels, fake_img, y, device=opt.d_device, per_sample=True, aux_penalty=opt.aux_penalty)
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
            penalty = calc_penalty(D, opt.penalty, penalty_real_data, penalty_real_labels, fake_img, y, device=opt.d_device, aux_penalty=opt.aux_penalty)
            d_loss += penalty

            if use_imm_sens:
                privacy_engine.backward(d_loss, img)
                if opt.imm_sens_scaling_mode == "moving-avg-pl":
                    update_sens_moving_avg()
                elif opt.imm_sens_scaling_mode[:-3] != "-pl":
                    update_is_logging()
            else:
                d_loss.backward()

    else:
        # No penalties
        if use_grad_clip:
            privacy_engine.accumulate_batch()
        elif use_imm_sens:
            privacy_engine.backward(d_loss, img)
            if opt.imm_sens_scaling_mode == "moving-avg-pl":
                update_sens_moving_avg()
            elif opt.imm_sens_scaling_mode[:-3] != "-pl":
                update_is_logging()
        elif use_tm:
            pass
        elif use_sv:
            pass
        else:
            d_loss.backward()

    if opt.clip_propogating_grads and use_dp:
        prop_grad_clipper.enable_hooks()

    d_optimizer.step()
    util.unfreeze(G)

    # Update discriminator logging
    logger.stats["D Adv Loss"] += d_real_loss.item() + d_fake_loss.item()
    logger.stats["D Real Loss"] += d_real_loss.item()
    logger.stats["D Fake Loss"] += d_fake_loss.item()
    logger.stats["D Real Acc"] += 100*np.mean(d_real.detach().cpu().numpy() > 0)
    logger.stats["D Fake Acc"] += 100*np.mean(d_fake.detach().cpu().numpy() < 0)

    if len(opt.penalty) > 0:
        logger.stats["D Penalty"] += penalty.item()

    if opt.is_acgan:
        logger.stats["D Real Aux Loss"] += d_real_aux_loss.item()
        logger.stats["D Real Aux Acc"] += 100*np.mean(np.argmax(d_real_aux.detach().cpu().numpy(), axis=1) == labels.detach().cpu().numpy())

def train_G(z, y):
    util.zero_grad(G)

    d_fake, d_fake_aux, _ = eval_G_D(z, y)
    g_adv_loss = G.loss(d_fake, opt.d_device)
    g_aux_loss = D.aux_loss(d_fake_aux, y.to(opt.d_device), opt.d_device) if opt.is_acgan else 0
    g_loss = g_adv_loss + g_aux_loss

    g_loss.backward()
    g_optimizer.step()

    # Update generator logging
    logger.stats["G Adv Loss"] += g_adv_loss.item() * opt.n_d_steps  # Multiply by n_d_steps for consistency with logging
    if opt.is_acgan:
        logger.stats["G Aux Loss"] += g_aux_loss.item() * opt.n_d_steps  # Multiply by n_d_steps for consistency with logging
        logger.stats["G Aux Acc"] += 100*np.mean(np.argmax(d_fake_aux.detach().cpu().numpy(), axis=1) == y.detach().cpu().numpy()) * opt.n_d_steps

def train(epoch, batch_i, real_images_batch, real_labels_batch, use_dp=False):
    real_images_batch = real_images_batch.to(opt.d_device)
    real_labels_batch = real_labels_batch.to(opt.d_device) if opt.conditional else None
    batch_size = real_images_batch.size(0)

    # Train discriminator
    train_D(real_images_batch, real_labels_batch, gen_z(batch_size), real_labels_batch, use_dp=use_dp)

    # Train generator
    if batch_i % opt.n_d_steps == 0: # only train generator every n_d_steps iterations
        train_G(gen_z(batch_size), gen_y(batch_size))

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

print("\nStarting training...\n")

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
        img, labels = next(iter(public_dataloader)) if opt.public_set_size > 0 else mean_sampler.sample(opt.batch_size)
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
