from enum import StrEnum

DATASETS = {
    "ou_process": {
        "npz_path": "data/ou_processes/ou_process_data.npz",
    },
    "rough_ou_process": {
        "npz_path": "data/rough_ou_processes/rough_ou_data_H0.70.npz",
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
