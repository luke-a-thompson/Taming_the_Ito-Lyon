from enum import Enum, StrEnum
from pathlib import Path


class Datasets(Enum):
    OU_PROCESS = Path("data/ou_processes/ou_process_data.npz")
    ROUGH_OU_PROCESS = Path("data/rough_ou_processes/rough_ou_data_H0.70.npz")
    BLACK_SCHOLES = Path("data/rough_volatility/black-scholes_data.npz")
    BERGOMI = Path("data/rough_volatility/bergomi_data.npz")
    ROUGH_BERGOMI = Path("data/rough_volatility/rough_bergomi_data.npz")
    SIMPLE_RBERGOMI = Path("data/rough_volatility/simple_rbergomi_data.npz")
    # Backwards-compatible alias (accepts dataset_name="simple_rough_bergomi" too).
    SIMPLE_ROUGH_BERGOMI = Path("data/rough_volatility/simple_rbergomi_data.npz")
    SG_SO3_SIMULATION = Path(
        "data/sg_so3_simulation/so3_simulation_rotmats_by_damping.npz"
    )
    OXFORD_MULTIMOTION_STATIC = Path("data/oxford_multimotion/swinging_4_static.npz")
    OXFORD_MULTIMOTION_TRANSLATIONAL = Path(
        "data/oxford_multimotion/swinging_4_translational.npz"
    )
    OXFORD_MULTIMOTION_UNCONSTRAINED = Path(
        "data/oxford_multimotion/swinging_4_unconstrained.npz"
    )
    SPD_COVARIANCE_SOLAR = Path(
        "data/spd_covariance/solar_spd_covariance_trajectory.npz"
    )
    SPD_WISHART_DIFFUSION = Path(
        "data/synthetic_diffusions/wishart_diffusion_data.npz"
    )


class ModelType(StrEnum):
    NCDE = "ncde"
    LOG_NCDE = "log_ncde"
    NRDE = "nrde"
    MNRDE = "mnrde"
    GRU = "gru"


class Optimizer(StrEnum):
    ADAM = "adam"
    ADAMW = "adamw"
    MUON = "muon"


class HopfAlgebraType(StrEnum):
    SHUFFLE = "shuffle"
    GL = "gl"
    MKW = "mkw"


class StepsizeControllerType(StrEnum):
    PID = "pid"
    CONSTANT = "constant"

class ManifoldType(StrEnum):
    EUCLIDEAN = "euclidean"
    SO3 = "so3"
    SPD = "spd"

class ControlInterpolationType(StrEnum):
    HERMITE_CUBIC = "hermite_cubic"
    LINEAR = "linear"


class ExtrapolationSchemeType(StrEnum):
    LINEAR = "linear"
    HERMITE = "hermite"
    SG = "sg"
    SO3_SG = "so3_sg"
    MLP = "mlp"
    PIECEWISE_MLP = "piecewiseMLP"


class LossType(StrEnum):
    MSE = "mse"
    RGE = "rge"
    SIGKER = "sigker"
    SIGKER_BRANCHED = "sigker_branched"
    FROBENIUS = "frobenius"


class TrainingMode(StrEnum):
    CONDITIONAL = "conditional"
    UNCONDITIONAL = "unconditional"


class FinalActivation(StrEnum):
    TANH = "tanh"
    IDENTITY = "identity"
