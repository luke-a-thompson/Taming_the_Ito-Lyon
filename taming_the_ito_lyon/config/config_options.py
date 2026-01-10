from enum import Enum, StrEnum
from pathlib import Path


class Datasets(Enum):
    OU_PROCESS = Path("data/ou_processes/ou_process_data.npz")
    ROUGH_OU_PROCESS = Path("data/rough_ou_processes/rough_ou_data_H0.70.npz")
    BLACK_SCHOLES = Path("data/rough_volatility/black-scholes_data.npz")
    BERGOMI = Path("data/rough_volatility/bergomi_data.npz")
    ROUGH_BERGOMI = Path("data/rough_volatility/rough_bergomi_data.npz")
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


class ModelType(StrEnum):
    NCDE = "ncde"
    LOG_NCDE = "log_ncde"
    NRDE = "nrde"
    MNRDE = "mnrde"
    SDEONET = "sdeonet"


class Optimizer(StrEnum):
    ADAM = "adam"
    ADAMW = "adamw"
    MUON = "muon"


class HopfAlgebraType(StrEnum):
    SHUFFLE = "shuffle"
    GL = "gl"
    MKW = "mkw"


class ExtrapolationSchemeType(StrEnum):
    LINEAR = "linear"
    HERMITE = "hermite"
    SG = "sg"
    MLP = "mlp"


class LossType(StrEnum):
    MSE = "mse"
    RGE = "rge"
    SIGKER = "sigker"
    FROBENIUS = "frobenius"


class TrainingMode(StrEnum):
    CONDITIONAL = "conditional"
    UNCONDITIONAL = "unconditional"


class UnconditionalDriverKind(StrEnum):
    BM = "bm"
    FBM = "fbm"
    RL = "rl"


class FinalActivation(StrEnum):
    TANH = "tanh"
    IDENTITY = "identity"
