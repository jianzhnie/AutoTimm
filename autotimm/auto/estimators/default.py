"""Default configs for image classification."""
# pylint: disable=bad-whitespace,missing-class-docstring
from typing import Tuple, Union

from autocfg import dataclass, field


@dataclass
class ImageClassification:
    model: str = 'resnet18'
    use_pretrained: bool = True


@dataclass
class TrainCfg:
    pretrained_base: bool = True  # whether load the imagenet pre-trained base
    batch_size: int = 32
    epochs: int = 30
    lr: float = 0.01  # learning rate
    decay_factor: float = 0.1  # decay rate of learning rate.
    lr_decay_period: int = 0
    lr_decay_epoch: str = '20, 27'  # epochs at which learning rate decays
    lr_schedule_mode: str = 'step'  # learning rate scheduler mode. options are step, poly and cosine
    end_lr: float = 0.0  # cosine lr schedule
    warmup_lr: float = 0.0  # starting warmup learning rate.
    warmup_epochs: int = 0  # number of warmup epochs
    num_workers: int = 4
    weight_decay: float = 0.0001
    momentum: float = 0.9
    nesterov: bool = False
    input_size: int = 224
    crop_ratio: float = 0.875
    data_augment: str = ''
    data_dir: str = ''
    label_smoothing: bool = False
    resume_epoch: int = 0
    mixup: bool = False
    mixup_alpha: float = 0.1
    mixup_off_epoch: int = 0
    log_interval: int = 50
    amp: bool = False
    static_loss_scale: float = 1.0
    dynamic_loss_scale: bool = False
    start_epoch: int = 0
    transfer_lr_mult: float = 0.01  # reduce the backbone lr_mult to avoid quickly destroying the features
    output_lr_mult: float = 0.1  # the learning rate multiplier for last fc layer if trained with transfer learning
    early_stop_patience: int = -1  # epochs with no improvement after which train is early stopped, negative: disabled
    early_stop_min_delta: float = 0.001  # ignore changes less than min_delta for metrics
    # the baseline value for metric, training won't stop if not reaching baseline
    early_stop_baseline: Union[float, int] = 0.0
    early_stop_max_value: Union[
        float, int] = 1.0  # early stop if reaching max value instantly


@dataclass
class ValidCfg:
    batch_size: int = 16
    num_workers: int = 4
    log_interval: int = 50


@dataclass
class ImageClassificationCfg:
    img_cls: ImageClassification = field(default_factory=ImageClassification)
    train: TrainCfg = field(default_factory=TrainCfg)
    valid: ValidCfg = field(default_factory=ValidCfg)
    gpus: Union[Tuple,
                list] = (0,
                         )  # gpu individual ids, not necessarily consecutive
