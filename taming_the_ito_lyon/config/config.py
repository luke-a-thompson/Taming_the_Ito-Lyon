from __future__ import annotations
import tomllib
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    PositiveFloat,
    model_validator,
    field_validator,
)
from taming_the_ito_lyon.config.config_options import (
    Optimizer,
    ModelType,
    Datasets,
    ExtrapolationScheme,
    LossType,
    HopfAlgebraType,
    TrainingMode,
    UnconditionalDriverKind,
    FinalActivation,
)


class ExperimentConfig(BaseModel):
    """Aggregated non-model experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    # Model selection
    model_type: ModelType = Field(description="Which model architecture to use")

    # Dataset
    dataset_name: Datasets = Field(
        description="Dataset name key from config_options.Datasets"
    )

    @field_validator("dataset_name", mode="before")
    @classmethod
    def coerce_dataset_name(cls, v: object) -> object:
        # Allow TOML strings like "ou_process" (matching enum member names).
        if isinstance(v, str):
            key = v.strip()
            try:
                return Datasets[key.upper()]
            except KeyError:
                return v
        return v

    train_fraction: PositiveFloat = Field(
        default=0.8, le=1.0, description="Fraction of data for training"
    )
    val_fraction: PositiveFloat = Field(
        default=0.1, le=1.0, description="Fraction of data for validation"
    )
    test_fraction: PositiveFloat = Field(
        default=0.1, le=1.0, description="Fraction of data for testing"
    )

    @model_validator(mode="after")
    def validate_fractions_sum(self) -> ExperimentConfig:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if total > 1.0:
            raise ValueError(
                f"Sum of train, val, and test fractions ({total:.4f}) cannot exceed 1.0"
            )
        return self

    # Extrapolation settings
    extrapolation_scheme: ExtrapolationScheme | None = Field(
        default=None, description="Extrapolation scheme"
    )
    n_recon: PositiveInt | None = Field(
        default=None,
        description="Number of reconstruction points for extrapolation (None for standard mode)",
    )
    n_future: PositiveInt | None = Field(
        default=None,
        description="Number of future points for extrapolation (None for standard mode)",
    )

    @model_validator(mode="after")
    def validate_extrapolation_params(self) -> ExperimentConfig:
        if self.extrapolation_scheme is not None:
            # Extrapolation requires the selected model type to support an
            # `extrapolation_scheme` interface.
            if self.model_type not in (
                ModelType.NCDE,
                ModelType.LOG_NCDE,
                ModelType.MNRDE,
            ):
                raise ValueError(
                    "extrapolation_scheme is only supported for model_type in "
                    "{ncde, log_ncde, mnrde}."
                )
            if self.n_recon is None or self.n_future is None:
                raise ValueError(
                    "n_recon and n_future must be specified when extrapolation_scheme is provided"
                )
        else:
            if self.n_recon is not None or self.n_future is not None:
                raise ValueError(
                    "n_recon and n_future must be None when extrapolation_scheme is not provided"
                )
        return self

    # Optimizer
    optimizer: Optimizer = Field(description="Optimizer name")
    learning_rate: PositiveFloat = Field(le=1.0, description="Optimizer learning rate")
    weight_decay: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Optimizer weight decay"
    )
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

    @model_validator(mode="after")
    def validate_early_stopping_patience(self) -> ExperimentConfig:
        if self.early_stopping_patience > self.epochs:
            raise ValueError(
                "early_stopping_patience must be <= epochs "
                f"(got patience={self.early_stopping_patience}, epochs={self.epochs})"
            )
        return self

    # Training mode (conditional vs unconditional generator)
    training_mode: TrainingMode = Field(
        default=TrainingMode.CONDITIONAL,
        description="Training mode: 'conditional' uses dataset controls; 'unconditional' samples a driver internally.",
    )
    unconditional_driver_kind: UnconditionalDriverKind | None = Field(
        default=None,
        description="Unconditional driver kind: bm / fbm / rl (Riemann-Liouville / rough).",
    )
    unconditional_driver_dim: PositiveInt | None = Field(
        default=None,
        description="Number of non-time driver channels for unconditional mode. The model input dim becomes (unconditional_driver_dim + 1) to include time.",
    )
    unconditional_hurst: PositiveFloat | None = Field(
        default=None,
        le=1.0,
        description="Hurst parameter used when unconditional_driver_kind is 'rl'.",
    )

    @model_validator(mode="after")
    def validate_training_mode(self) -> ExperimentConfig:
        match self.training_mode:
            case TrainingMode.CONDITIONAL:
                if self.dataset_name in (
                    Datasets.BLACK_SCHOLES,
                    Datasets.BERGOMI,
                    Datasets.ROUGH_BERGOMI,
                ):
                    raise ValueError(
                        "Rough volatility datasets do not support conditional training"
                    )
                if (
                    self.unconditional_driver_kind is not None
                    or self.unconditional_driver_dim is not None
                    or self.unconditional_hurst is not None
                ):
                    raise ValueError(
                        "unconditional_driver_kind, unconditional_driver_dim, and unconditional_hurst must be None when training_mode='conditional'"
                    )
            case TrainingMode.UNCONDITIONAL:
                if (
                    self.unconditional_driver_kind is None
                    or self.unconditional_driver_dim is None
                    or self.unconditional_hurst is None
                ):
                    raise ValueError(
                        "unconditional_driver_kind, unconditional_driver_dim, and unconditional_hurst must be set when training_mode='unconditional'"
                    )
                if self.extrapolation_scheme is not None:
                    raise ValueError(
                        "extrapolation_scheme must be None when training_mode='unconditional'"
                    )
            case _:
                raise ValueError(f"Unknown training mode: {self.training_mode}")
        return self

    @model_validator(mode="after")
    def validate_unconditional_driver_kind(self) -> ExperimentConfig:
        # Only validate driver hyperparameters when unconditional mode is active.
        if self.training_mode != TrainingMode.UNCONDITIONAL:
            return self
        if self.unconditional_driver_kind is None or self.unconditional_hurst is None:
            return self

        if (
            self.unconditional_driver_kind == UnconditionalDriverKind.BM
            and self.unconditional_hurst != 0.5
        ):
            raise ValueError(
                f"Hurst parameter (currently {self.unconditional_hurst}) must be 0.5 for Brownian motion driver"
            )
        if self.unconditional_driver_kind in (
            UnconditionalDriverKind.FBM,
            UnconditionalDriverKind.RL,
        ):
            hurst = float(self.unconditional_hurst)
            if not (0.0 < hurst < 1.0):
                raise ValueError(
                    f"Hurst parameter must be in (0, 1) for driver_kind='{self.unconditional_driver_kind}' "
                    f"(got {hurst})"
                )
        return self


class NCDEConfig(BaseModel):
    """Top-level NCDE configuration composed of model params."""

    model_config = ConfigDict(extra="forbid")

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

    @model_validator(mode="after")
    def validate_solver_tolerances(self) -> NCDEConfig:
        if not (0.0 < float(self.rtol) < 1.0):
            raise ValueError(f"rtol must be in (0, 1), got {self.rtol}")
        if not (0.0 < float(self.atol) < 1.0):
            raise ValueError(f"atol must be in (0, 1), got {self.atol}")
        if not (0.0 < float(self.dtmin) <= 1.0):
            raise ValueError(f"dtmin must be in (0, 1], got {self.dtmin}")
        return self


class NRDEConfig(BaseModel):
    """Top-level NRDE configuration composed of model params."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    # Model params
    initial_hidden_dim: PositiveInt = Field(
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

    # Signature config
    signature_depth: PositiveInt = Field(le=5, description="Signature depth")
    signature_window_size: PositiveInt = Field(
        default=1, description="Data steps per log-signature window"
    )

    # Hopf algebra for M-NRDE
    hopf_algebra: HopfAlgebraType = Field(description="Hopf algebra to use")


class LogNCDEConfig(BaseModel):
    """Top-level Log-NCDE configuration composed of model params."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    @model_validator(mode="after")
    def validate_architecture(self) -> SDEONetConfig:
        if int(self.wick_order) not in (1, 2):
            raise ValueError(f"wick_order must be 1 or 2, got {self.wick_order}")
        if self.use_posenc and int(self.pe_dim) % 2 != 0:
            raise ValueError(
                f"pe_dim must be even when use_posenc=True, got {self.pe_dim}"
            )
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
