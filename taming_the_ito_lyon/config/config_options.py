from enum import StrEnum

DATASETS = {
    "ou_process": {
        "npz_path": "data/ou_processes/ou_process_data.npz",
    },
    "rough_ou_process": {
        "npz_path": "data/rough_ou_processes/rough_ou_data_H0.70.npz",
    },
    "sg_so3_simulation": {
        "npz_path": "data/sg_so3_simulation/sg_so3_simulation_data.npz",
    },
    "oxford_multimotion_static": {
        "npz_path": "data/oxford_multimotion/swinging_4_static.npz",
    },
    "oxford_multimotion_translational": {
        "npz_path": "data/oxford_multimotion/swinging_4_translational.npz",
    },
    "oxford_multimotion_unconstrained": {
        "npz_path": "data/oxford_multimotion/swinging_4_unconstrained.npz",
    },
}


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


class ExtrapolationScheme(StrEnum):
    LINEAR = "linear"
    CUBIC = "cubic"
    SG = "sg"
    MLP = "mlp"


class LossType(StrEnum):
    MSE = "mse"
    RGE = "rge"
    SIGKER = "sigker"
