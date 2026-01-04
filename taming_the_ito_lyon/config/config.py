from __future__ import annotations
import tomllib
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat, model_validator
from taming_the_ito_lyon.config.config_options import (
    Optimizer,
    ModelType,
    ExtrapolationScheme,
    LossType,
    HopfAlgebraType,
)


class ExperimentConfig(BaseModel):
    """Aggregated non-model experiment configuration."""

    # Model selection
    model_type: ModelType = Field(description="Which model architecture to use")

    # Dataset
    dataset_name: str = Field(
        description="Dataset name key from config_options.DATASETS"
    )

    train_fraction: PositiveFloat = Field(
        default=0.8, le=1.0, description="Fraction of data for training"
    )
    val_fraction: PositiveFloat = Field(
        default=0.1, le=1.0, description="Fraction of data for validation"
    )
    test_fraction: PositiveFloat = Field(
        default=0.1, le=1.0, description="Fraction of data for testing"
    )

    # Extrapolation settings
    use_extrapolation: bool = Field(
        default=False, description="Enable extrapolation mode for training"
    )
    extrapolation_scheme: ExtrapolationScheme | None = Field(
        default=None, description="Extrapolation scheme"
    )
    n_recon: PositiveInt | None = Field(
        default=None,
        description="Number of reconstruction points for extrapolation (None for standard mode)",
    )

    @model_validator(mode="after")
    def validate_fractions_sum(self) -> ExperimentConfig:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if total > 1.0:
            raise ValueError(
                f"Sum of train, val, and test fractions ({total:.4f}) cannot exceed 1.0"
            )
        return self

    @model_validator(mode="after")
    def validate_extrapolation_params(self) -> ExperimentConfig:
        if self.use_extrapolation:
            if self.extrapolation_scheme is None:
                raise ValueError(
                    "extrapolation_scheme must be specified when use_extrapolation=True"
                )
            if self.n_recon is None:
                raise ValueError(
                    "n_recon must be specified when use_extrapolation=True"
                )
        return self

    # Optimizer
    optimizer: Optimizer = Field(description="Optimizer name")
    learning_rate: PositiveFloat = Field(le=1.0, description="Optimizer learning rate")
    weight_decay: PositiveFloat = Field(le=1.0, description="Optimizer weight decay")
    max_grad_norm: PositiveFloat | None = Field(
        default=None,
        description="Maximum gradient norm for clipping (None for no clipping)",
    )

    # Training
    loss: LossType = Field(description="Loss function to use")
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
    init_hidden_dim: PositiveInt = Field(
        description="Initial condition MLP hidden state dimension"
    )
    initial_cond_mlp_depth: PositiveInt = Field(
        description="Initial condition MLP depth (number of hidden layers)"
    )
    vf_hidden_dim: PositiveInt = Field(description="Vector field MLP width")
    vf_mlp_depth: PositiveInt = Field(
        description="Vector field MLP depth (number of hidden layers)"
    )
    cde_state_dim: PositiveInt = Field(description="CDE hidden state dimension")
    out_size: PositiveInt = Field(description="Output channels predicted by readout")

    # Solver tolerances
    rtol: PositiveFloat = Field(description="Relative tolerance for solver")
    atol: PositiveFloat = Field(description="Absolute tolerance for solver")
    dtmin: PositiveFloat = Field(description="Minimum time step for solver")


class NRDEConfig(BaseModel):
    """Top-level NRDE configuration composed of model params."""

    # Model params
    cde_state_dim: PositiveInt = Field(description="CDE hidden state dimension")
    vf_hidden_dim: PositiveInt = Field(description="Vector field MLP width")
    initial_cond_mlp_depth: PositiveInt = Field(
        description="Initial condition MLP depth (number of hidden layers)"
    )
    vf_mlp_depth: PositiveInt = Field(
        description="Vector field MLP depth (number of hidden layers)"
    )
    out_size: PositiveInt = Field(description="Output channels predicted by readout")

    # Signature config
    signature_depth: PositiveInt = Field(le=5, description="Signature depth")

    # Log-signature window size in data steps (polyline uses window_size+1 points)
    signature_window_size: PositiveInt = Field(
        default=1, description="Data steps per log-signature window"
    )


class MNRDEConfig(BaseModel):
    """Top-level M-NRDE configuration composed of model params."""

    # Model params
    cde_state_dim: PositiveInt = Field(description="CDE hidden state dimension")
    vf_hidden_dim: PositiveInt = Field(description="Vector field MLP width")
    initial_cond_mlp_depth: PositiveInt = Field(
        description="Initial condition MLP depth (number of hidden layers)"
    )
    vf_mlp_depth: PositiveInt = Field(
        description="Vector field MLP depth (number of hidden layers)"
    )
    out_size: PositiveInt = Field(description="Output channels predicted by readout")

    # Signature config
    signature_depth: PositiveInt = Field(le=5, description="Signature depth")
    signature_window_size: PositiveInt = Field(
        default=1, description="Data steps per log-signature window"
    )

    # Hopf algebra for M-NRDE
    hopf_algebra: HopfAlgebraType = Field(description="Hopf algebra to use")


class LogNCDEConfig(BaseModel):
    """Top-level Log-NCDE configuration composed of model params."""

    # Model params
    cde_state_dim: PositiveInt = Field(description="CDE hidden state dimension")
    vf_hidden_dim: PositiveInt = Field(description="Vector field MLP width")
    initial_cond_mlp_depth: PositiveInt = Field(
        description="Initial condition MLP depth (number of hidden layers)"
    )
    vf_mlp_depth: PositiveInt = Field(
        description="Vector field MLP depth (number of hidden layers)"
    )
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
    ncde_config: NCDEConfig | None = None
    log_ncde_config: LogNCDEConfig | None = None
    nrde_config: NRDEConfig | None = None
    mnrde_config: MNRDEConfig | None = None
    sdeonet_config: SDEONetConfig | None = None

    @model_validator(mode="after")
    def validate_model_config_exists(self) -> "Config":
        """Ensure the correct config section exists for the chosen model_type."""
        model_type = self.experiment_config.model_type

        config_map = {
            ModelType.NCDE: self.ncde_config,
            ModelType.LOG_NCDE: self.log_ncde_config,
            ModelType.NRDE: self.nrde_config,
            ModelType.MNRDE: self.mnrde_config,
            ModelType.SDEONET: self.sdeonet_config,
        }

        active_config = config_map.get(model_type)
        if active_config is None:
            raise ValueError(
                f"Model type '{model_type}' requires a '{model_type}_config' section in the TOML"
            )

        # Ensure no extra configs are provided
        all_configs = [
            self.ncde_config,
            self.log_ncde_config,
            self.nrde_config,
            self.mnrde_config,
            self.sdeonet_config,
        ]
        num_provided = sum(c is not None for c in all_configs)
        if num_provided > 1:
            raise ValueError("Only one model config section should be provided")

        return self

    @property
    def nn_config(
        self,
    ) -> NCDEConfig | LogNCDEConfig | NRDEConfig | MNRDEConfig | SDEONetConfig:
        """Get the active model configuration based on model_type."""
        model_type = self.experiment_config.model_type

        if model_type == ModelType.NCDE:
            assert self.ncde_config is not None
            return self.ncde_config
        elif model_type == ModelType.LOG_NCDE:
            assert self.log_ncde_config is not None
            return self.log_ncde_config
        elif model_type == ModelType.NRDE:
            assert self.nrde_config is not None
            return self.nrde_config
        elif model_type == ModelType.MNRDE:
            assert self.mnrde_config is not None
            return self.mnrde_config
        elif model_type == ModelType.SDEONET:
            assert self.sdeonet_config is not None
            return self.sdeonet_config
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def load_toml_config(toml_path: str) -> Config:
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return Config.model_validate(data)
