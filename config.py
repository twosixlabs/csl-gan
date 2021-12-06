import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf
from typing import List, Optional, Any

from hydra_configs.torch.optim import AdamConf


# NOTE: _dbo stands for defaults based on, and _po stands for per option, as in per option in the group


# # # # # # #
#  General  #
# # # # # # #

@dataclass
class ResumeConf:
    resumed: bool = MISSING
@dataclass
class NotResumedConf:
    resumed = False
@dataclass
class ResumedConf(ResumeConf):
    resumed = True
    path: str = MISSING
    override_args: List[str] = MISSING


# # # # # # # # # # # # # # #
#   Dataset and Dataloader  #
# # # # # # # # # # # # # # #

@dataclass
class ImgSizeConf:
    width: int = MISSING
    height: int = MISSING

@dataclass
class DatasetConf:
    name: str = MISSING
    train_set_size: int = MISSING  # subtracted by public_set_size during init
    public_set_size: int = 0
    img_size: ImgSizeConf = MISSING
@dataclass
class FolderDatasetConf(DatasetConf):
    path: str = MISSING
    label_path: str = MISSING
    label_attr: str = MISSING
@dataclass
class PyTorchDatasetConf(DatasetConf):
    download: bool = False


@dataclass
class DataloaderConf:
    use_uniform_random_sampler: bool = False #TO-DO: Remove default
    n_workers: int = 8
    batch_size: int = MISSING

    _dbo_dataset: Any = MISSING


# # # # # # # # #
#  Public Data  #
# # # # # # # # #

@dataclass
class PublicDataConf:
    warmup_iter: int = 0
    type: str = MISSING
    use_for_penalty: bool = False # only happens if using DP, won't use even if set to True with no DP
@dataclass
class PublicSetDataConf(PublicDataConf):
    type = "public_set"
    use_for_penalty = True
    set_size: int = 2000

    _public_set_dbo_dataset: Any = MISSING
@dataclass
class MeanSamplesDataConf(PublicDataConf):
    type = "mean_samples"
    use_for_penalty = True
    n_mean_samples: int = MISSING
    mean_sample_size: int = MISSING
    mean_sample_noise_std: float = MISSING

    _mean_samples_dbo_dataset_cond: Any = MISSING



# # # # # # #
#   Model   #
# # # # # # #

# Model architecture

@dataclass
class ModelArchConf:
    type: str = MISSING
    n_d_steps: int = MISSING
    penalty: List[str] = field(default_factory=lambda: []) # can include WGAN-GP, WGAN-GP1, DRAGAN, DRAGAN1

@dataclass
class DCRNArchConf(ModelArchConf):
    type = "dcrn"
    n_d_steps = 5
    penalty = ["WGAN-GP"]

    # Generator
    g_channels: List[int] = MISSING
    g_first_filter_size: int = MISSING

    # Discriminator
    d_channels: List[int] = MISSING
    d_last_filter_size: int = MISSING

    _dcrn_dbo_dataset_imgsize: Any = MISSING

@dataclass
class VanillaArchConf(ModelArchConf):
    # TO-DO: Make configurable
    type = "vanilla"
    n_d_steps = 1

    _vanilla_dbo_dataset_imgsize: Any = MISSING

# Model conditional architecture

@dataclass
class ConditionalArchConf:
    type: str = MISSING
@dataclass
class CGANArchConf(ConditionalArchConf):
    type = "CGAN"
    g_label_enc_mode: str = "concat"
    d_label_enc_mode: str = "concat"

@dataclass
class AuxLossConf:
    type: str = MISSING
@dataclass
class ACGANArchConf(ConditionalArchConf):
    type = "ACGAN"
    aux_loss: AuxLossConf = MISSING
    use_fake_aux_loss: bool = True # Experimentally determined that should be true in most cases
    aux_loss_scalar: float = 1
    use_aux_penalty: bool = True # Should probably be true unless some other constraint is on aux out weights

    _acgan_dbo_modelarch: Any = MISSING


@dataclass
class ModelConditionalConf:
    is_cond: bool = MISSING
@dataclass
class UnconditionalModelConf(ModelConditionalConf):
    is_cond = False
@dataclass
class ConditionalModelConf(ModelConditionalConf):
    is_cond = True
    arch: ConditionalArchConf = MISSING

# Model parent config

@dataclass
class ModelConf:
    # Generator
    z_dim: int = MISSING
    bn: bool = MISSING
    out_ch: int = MISSING

    # Discriminator
    train_d_to_threshold: Optional[float] = None # mutually exclusive with n_d_steps
    n_d_steps: Optional[int] = 1

    arch: ModelArchConf = MISSING
    cond: ModelConditionalConf = MISSING

    _dbo_dataset: Any = MISSING


# # # # # # # #
#   Privacy   #
# # # # # # # #

@dataclass
class DPConfig:
    type: str = MISSING
    delta: float = MISSING
    epsilon_budget: float = MISSING

    _dbo_dataset: Any = MISSING
@dataclass
class NoDPConfig(DPConfig):
    type: str = "none"

# Gradient clipping

@dataclass
class GCAdaptConfig:
    type: str = MISSING
@dataclass
class NoGCAdaptConfig(GCAdaptConfig):
    type = "none"
@dataclass
class MovingAvgGCAdaptConfig(GCAdaptConfig):
    type = "moving_avg"
    # v*beta + grad_norm*target_scale*(1-beta)
    target_val: float = 5
    beta: float = 0.95
@dataclass
class PublicDataGCAdaptConfig(GCAdaptConfig):
    type = "public_data"
    adaptive_scalar: float = 1

@dataclass
class ClippingParamConfig:
    per_layer: bool = MISSING
@dataclass
class PerLayerClippingParamConfig(ClippingParamConfig):
    per_layer = True
    val: List[float] = MISSING

    _per_layer_dbo_modelarch_dataset_imgsize: Any = MISSING
@dataclass
class SingleClippingParamConfig(ClippingParamConfig):
    per_layer = False
    val: float = MISSING

    _single_dbo_modelarch_dataset_imgsize: Any = MISSING

@dataclass
class GCDPConfig(DPConfig):
    type = "gc"
    split: bool = True
    clip_param: ClippingParamConfig = MISSING
    adapt: GCAdaptConfig = MISSING

@dataclass
class ISScalingConfig:
    type: str = MISSING
    scaling_vec: List[float] = MISSING

    _dbo_modelarch_dataset_imgsize: Any = MISSING
@dataclass
class NoISScalingConfig(ISScalingConfig):
    type = "none"
@dataclass
class ConstantISScalingConfig(ISScalingConfig):
    type = "constant"
@dataclass
class MovingAvgISScalingConfig(ISScalingConfig):
    type = "moving_avg"
    beta: float = 0.95

@dataclass
class ISDPConfig(DPConfig):
    type = "is"

    scaling: ISScalingConfig = MISSING



@dataclass
class SmoothSensDPConfig:
    t: float = MISSING
    rho_per_epoch: float = MISSING


# # # # # # #
#    Main   #
# # # # # # #

@dataclass
class Config:
    # General
    seed: int = MISSING
    output_path: str = MISSING
    resume: ResumeConf = MISSING

    # Data
    n_classes: int = MISSING
    dataset: DatasetConf = MISSING
    dataloader: DataloaderConf = DataloaderConf()

    adam: AdamConf = AdamConf()
    n_epochs: int = MISSING
    g_lr: float = MISSING
    d_lr: float = MISSING

    g_device: str = MISSING
    d_device: str = MISSING
    batch_split_size: int = MISSING

    public_data: PublicDataConf = MISSING

    model: ModelConf = MISSING

    dp: DPConfig = MISSING

    save_interval: int = MISSING
    log_interval: int = MISSING
    sample_interval: int = MISSING
    sample_num: int = MISSING

    profile_training: bool = False

    _dbo_dataset: Any = MISSING


cs = ConfigStore.instance()
cs.store(group="resume", name="y", node=ResumedConf)
cs.store(group="resume", name="n", node=NotResumedConf)
cs.store(group="dataset", name="folder", node=FolderDatasetConf)
cs.store(group="dataset", name="pytorch", node=PyTorchDatasetConf)
cs.store(group="public_data", name="none", node=PublicDataConf)
cs.store(group="public_data", name="base_mean_samples", node=MeanSamplesDataConf)
cs.store(group="public_data", name="base_public_set", node=PublicSetDataConf)
cs.store(group="model", name="base_model", node=ModelConf)
cs.store(group="model/arch", name="base_dcrn", node=DCRNArchConf)
cs.store(group="model/arch", name="base_vanilla", node=VanillaArchConf)
cs.store(group="model/cond", name="base_unconditional", node=UnconditionalModelConf)
cs.store(group="model/cond", name="base_conditional", node=ConditionalModelConf)
cs.store(group="model/cond/arch", name="base_cgan", node=CGANArchConf)
cs.store(group="model/cond/arch", name="base_acgan", node=ACGANArchConf)
cs.store(group="model/cond/arch/aux_loss", name="base_aux_loss", node=AuxLossConf)
cs.store(group="dp", name="none", node=NoDPConfig)
cs.store(group="dp", name="base_gc", node=GCDPConfig)
cs.store(group="dp", name="base_is", node=ISDPConfig)
cs.store(group="dp/clip_param", name="base_per_layer", node=PerLayerClippingParamConfig)
cs.store(group="dp/clip_param", name="base_single", node=SingleClippingParamConfig)
cs.store(group="dp/adapt", name="none", node=NoGCAdaptConfig)
cs.store(group="dp/adapt", name="base_moving_avg", node=MovingAvgGCAdaptConfig)
cs.store(group="dp/adapt", name="base_public_data", node=PublicDataGCAdaptConfig)
cs.store(group="dp/none", name="base_constant", node=NoISScalingConfig)
cs.store(group="dp/scaling", name="base_constant", node=ConstantISScalingConfig)
cs.store(group="dp/scaling", name="base_moving_avg", node=MovingAvgISScalingConfig)
cs.store(name="base_conf", node=Config)

@hydra.main(config_path="conf", config_name="config")
def load_conf(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))

load_conf()
