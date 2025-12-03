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
    SDEONET = "sdeonet"


class Optimizer(StrEnum):
    ADAM = "adam"
    ADAMW = "adamw"
    MUON = "muon"
