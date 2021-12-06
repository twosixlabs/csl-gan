import torch
from torch import autograd

def calc_penalty(model, penalty_types, real_data, real_labels, fake_data, fake_labels, device="cpu", per_sample=False, weights=None, aux_penalty=False):
    penalty = 0
    weights = [1/len(penalty_types) for _ in penalty_types] if weights is None else weights

    for i, penalty_type in enumerate(penalty_types):
        if penalty_type.startswith("DRAGAN"):
            p = calc_DRAGAN_penalty(model, real_data, real_labels, device=device, per_sample=per_sample, one_sided=penalty_type[-1] == "1")
        elif penalty_type.startswith("WGAN-GP"):
            p = calc_WGAN_GP_penalty(model, real_data, real_labels, fake_data, fake_labels, device=device, per_sample=per_sample, one_sided=penalty_type[-1] == "1", aux_penalty=aux_penalty)
        else:
            raise Exception("Unknown penalty type: " + penalty_type)

        penalty += weights[i] * p

    return penalty

def calc_DRAGAN_penalty(model, real_data, real_labels, device="cpu", per_sample=False, noise_std=None, one_sided=False, weight=10.0):
    if noise_std is None:
        if per_sample:
            raise Exception("Cannot calculate per-sample penalty without being given noise std")

        noise_std = real_data.std()

    noise = torch.unsqueeze(noise_std, dim=0).expand(real_data.size(0)) * torch.empty(real_data.shape).random_(0, 1)

    return weight * calc_lipschitz_penalty_WRT(model, (real_data + noise).detach(), real_labels, device=device, per_sample=per_sample, one_sided=one_sided)

def calc_WGAN_GP_penalty(model, real_data, real_labels, fake_data, fake_labels, device="cpu", per_sample=False, one_sided=False, weight=10.0, aux_penalty=False):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(*real_data.shape)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)

    return weight * calc_lipschitz_penalty_WRT(model, interpolates, real_labels, device=device, per_sample=per_sample, one_sided=one_sided, aux_penalty=aux_penalty)

def calc_lipschitz_penalty_WRT(model, inputs, input_labels=None, device="cpu", per_sample=False, one_sided=False, aux_penalty=True):
    inputs = inputs.detach()
    input_labels = None if input_labels is None else input_labels.detach()
    inputs.requires_grad_(True)

    out, aux_out = model(inputs, input_labels)

    gradients = autograd.grad(outputs=out, inputs=inputs, grad_outputs=torch.ones(out.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    norms = gradients.norm(2, dim=1)
    gradient_penalties = (norms-1).clamp(min=0)**2 if one_sided else (norms-1)**2

    if aux_penalty and not aux_out is None:
        for i in range(aux_out.size(1)):
            gradients_aux = autograd.grad(outputs=aux_out[:,i], inputs=inputs, grad_outputs=torch.ones(aux_out[:,i].size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradients_aux = gradients_aux.view(gradients_aux.size(0), -1)
            norms_aux = gradients_aux.norm(2, dim=1)
            gradient_penalties_aux = (norms_aux-1).clamp(min=0)**2 if one_sided else (norms_aux-1)**2
            gradient_penalties += gradient_penalties_aux

    return gradient_penalties if per_sample else gradient_penalties.mean()
