import tomllib
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat, model_validator
from taming_the_ito_lyon.config import Optimizer


class ExperimentConfig(BaseModel):
    """Aggregated non-model experiment configuration."""

    # Dataset
    dataset_name: str = Field(
        description="Dataset name key from config_options.DATASETS"
    )

    train_fraction: PositiveFloat = Field(
        default=0.6, le=1.0, description="Fraction of data for training"
    )
    val_fraction: PositiveFloat = Field(
        default=0.2, le=1.0, description="Fraction of data for validation"
    )
    test_fraction: PositiveFloat = Field(
        default=0.2, le=1.0, description="Fraction of data for testing"
    )

    @model_validator(mode="after")
    def validate_fractions_sum(self) -> "ExperimentConfig":
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if total > 1.0:
            raise ValueError(
                f"Sum of train, val, and test fractions ({total:.4f}) cannot exceed 1.0"
            )
        return self

    # Optimizer
    optimizer: Optimizer = Field(description="Optimizer name")
    learning_rate: PositiveFloat = Field(le=1.0, description="Optimizer learning rate")
    weight_decay: PositiveFloat = Field(le=1.0, description="Optimizer weight decay")

    # Training
    seed: PositiveInt = Field(description="PRNG seed")
    batch_size: PositiveInt = Field(
        multiple_of=8, description="Batch size; divisible by 8"
    )
    epochs: PositiveInt = Field(description="Number of epochs")
    early_stopping_patience: PositiveInt = Field(
        default=25, description="Epochs with no val improvement before stopping"
    )


class NCDEConfig(BaseModel):
    """Top-level NCDE configuration composed of model params."""

    # Model params
    hidden_size: PositiveInt = Field(description="Hidden state dimension")
    width_size: PositiveInt = Field(description="MLP width")
    depth: PositiveInt = Field(description="MLP depth (number of hidden layers)")
    out_size: PositiveInt = Field(description="Output channels predicted by readout")

    # Solver tolerances
    rtol: PositiveFloat = Field(description="Relative tolerance for solver")
    atol: PositiveFloat = Field(description="Absolute tolerance for solver")
    dtmin: PositiveFloat = Field(description="Minimum time step for solver")


class NRDEConfig(BaseModel):
    """Top-level NRDE configuration composed of model params."""

    # Model params
    hidden_size: PositiveInt = Field(description="Hidden state dimension")
    width_size: PositiveInt = Field(description="MLP width")
    depth: PositiveInt = Field(description="MLP depth (number of hidden layers)")
    out_size: PositiveInt = Field(description="Output channels predicted by readout")

    # Signature config
    signature_depth: PositiveInt = Field(le=5, description="Signature depth")

    # Log-signature window size in data steps (polyline uses window_size+1 points)
    signature_window_size: PositiveInt = Field(
        default=1, description="Data steps per log-signature window"
    )


class SDEONetConfig(BaseModel):
    """Top-level SDEONet configuration composed of model params."""

    # Model params
    basis_in_dim: PositiveInt = Field(description="Number of Haar basis functions J")
    basis_out_dim: PositiveInt = Field(description="Low-rank/trunk output dimension R")
    T: PositiveFloat = Field(description="Time horizon")
    hermite_M: PositiveInt = Field(description="Max Hermite order M")
    wick_order: PositiveInt = Field(le=2, description="Wick order (1 or 2)")

    # Time positional encoding
    use_posenc: bool = Field(description="Use positional encoding for trunk input")
    pe_dim: PositiveInt = Field(description="Positional encoding dimension (even)")
    include_raw_time: bool = Field(description="Concatenate raw tau to PE features")

    # Branch MLP
    branch_width: PositiveInt = Field(description="Branch MLP width")
    branch_depth: PositiveInt = Field(description="Branch MLP depth")

    # Trunk MLP
    trunk_width: PositiveInt = Field(description="Trunk MLP width")
    trunk_depth: PositiveInt = Field(description="Trunk MLP depth")

    # Normalization/skip
    use_layernorm: bool = Field(description="Use LayerNorm in MLPs")
    residual: bool = Field(description="Use residual connections in MLPs")


class Config(BaseModel):
    experiment_config: ExperimentConfig
    nn_config: NCDEConfig | NRDEConfig | SDEONetConfig


def load_toml_config(toml_path: str) -> Config:
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return Config.model_validate(data)
