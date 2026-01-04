import jax
import optax
from taming_the_ito_lyon.config import (
    Optimizer,
    Config,
    NCDEConfig,
    LogNCDEConfig,
    NRDEConfig,
    MNRDEConfig,
)
from taming_the_ito_lyon.models import (
    NeuralCDE,
    LogNCDE,
    NeuralRDE,
    MNDRE,
    create_scheme,
)
from taming_the_ito_lyon.models.extrapolation import (
    ExtrapolationScheme as ExtrapolationSchemeProtocol,
)
from taming_the_ito_lyon.config import DATASETS
from taming_the_ito_lyon.data.datasets import prepare_dataset


def _maybe_create_extrapolation_scheme(
    config: Config,
    *,
    input_path_dim: int,
    key: jax.Array,
) -> tuple[jax.Array, ExtrapolationSchemeProtocol | None]:
    """Optionally create an extrapolation scheme and (if needed) split the PRNG key."""

    if not config.experiment_config.use_extrapolation:
        return key, None

    # Only these models currently accept extrapolation parameters.
    if not isinstance(config.nn_config, (NCDEConfig, LogNCDEConfig, MNRDEConfig)):
        return key, None

    scheme_enum = config.experiment_config.extrapolation_scheme
    n_recon = config.experiment_config.n_recon
    assert scheme_enum is not None
    assert n_recon is not None

    model_key, scheme_key = jax.random.split(key)
    extrapolation_scheme = create_scheme(
        scheme_enum.value,
        num_points=n_recon,
        input_dim=input_path_dim,
        key=scheme_key,
    )
    return model_key, extrapolation_scheme


def create_model(
    config: Config,
    *,
    input_path_dim: int,
    output_path_dim: int,
    key: jax.Array,
) -> NeuralCDE | LogNCDE | NeuralRDE | MNDRE:
    model_key, extrapolation_scheme = _maybe_create_extrapolation_scheme(
        config, input_path_dim=input_path_dim, key=key
    )
    match config.nn_config:
        case NCDEConfig():
            return NeuralCDE(
                input_path_dim=input_path_dim,
                init_hidden_dim=config.nn_config.init_hidden_dim,
                initial_cond_mlp_depth=config.nn_config.initial_cond_mlp_depth,
                vf_hidden_dim=config.nn_config.vf_hidden_dim,
                vf_mlp_depth=config.nn_config.vf_mlp_depth,
                cde_state_dim=config.nn_config.cde_state_dim,
                output_path_dim=output_path_dim,
                key=model_key,
                rtol=config.nn_config.rtol,
                atol=config.nn_config.atol,
                dtmin=config.nn_config.dtmin,
                extrapolation_scheme=extrapolation_scheme,
                n_recon=config.experiment_config.n_recon,
            )
        case LogNCDEConfig():
            return LogNCDE(
                input_path_dim=input_path_dim,
                cde_state_dim=config.nn_config.cde_state_dim,
                vf_hidden_dim=config.nn_config.vf_hidden_dim,
                initial_cond_mlp_depth=config.nn_config.initial_cond_mlp_depth,
                vf_mlp_depth=config.nn_config.vf_mlp_depth,
                output_path_dim=output_path_dim,
                signature_depth=config.nn_config.signature_depth,
                signature_window_size=config.nn_config.signature_window_size,
                extrapolation_scheme=extrapolation_scheme,
                n_recon=config.experiment_config.n_recon,
                key=model_key,
            )
        case NRDEConfig():
            return NeuralRDE(
                input_path_dim=input_path_dim,
                cde_state_dim=config.nn_config.cde_state_dim,
                vf_hidden_dim=config.nn_config.vf_hidden_dim,
                initial_cond_mlp_depth=config.nn_config.initial_cond_mlp_depth,
                vf_mlp_depth=config.nn_config.vf_mlp_depth,
                output_path_dim=output_path_dim,
                signature_depth=config.nn_config.signature_depth,
                signature_window_size=config.nn_config.signature_window_size,
                key=key,
            )
        case MNRDEConfig():
            return MNDRE(
                input_path_dim=input_path_dim,
                cde_state_dim=config.nn_config.cde_state_dim,
                vf_hidden_dim=config.nn_config.vf_hidden_dim,
                initial_cond_mlp_depth=config.nn_config.initial_cond_mlp_depth,
                vf_mlp_depth=config.nn_config.vf_mlp_depth,
                output_path_dim=output_path_dim,
                signature_depth=config.nn_config.signature_depth,
                signature_window_size=config.nn_config.signature_window_size,
                hopf_algebra_type=config.nn_config.hopf_algebra,
                extrapolation_scheme=extrapolation_scheme,
                n_recon=config.experiment_config.n_recon,
                key=model_key,
            )
        # case SDEONetConfig():
        #     return SDEONet(
        #         basis_in_dim=config.nn_config.basis_in_dim,
        #         basis_out_dim=config.nn_config.basis_out_dim,
        #         T=config.nn_config.T,
        #         hermite_M=config.nn_config.hermite_M,
        #         wick_order=config.nn_config.wick_order,
        #         use_posenc=config.nn_config.use_posenc,
        #         pe_dim=config.nn_config.pe_dim,
        #         include_raw_time=config.nn_config.include_raw_time,
        #         branch_width=config.nn_config.branch_width,
        #         branch_depth=config.nn_config.branch_depth,
        #         trunk_width=config.nn_config.trunk_width,
        #         trunk_depth=config.nn_config.trunk_depth,
        #         use_layernorm=config.nn_config.use_layernorm,
        #         residual=config.nn_config.residual,
        #         key=key,
        #     )
        case _:
            raise ValueError(f"Unknown model: {config.model_config}")


def create_optimizer(
    optimizer_name: Optimizer,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float | None = None,
) -> optax.GradientTransformation:
    match optimizer_name:
        case Optimizer.ADAM:
            base_optim = optax.adam(learning_rate)
        case Optimizer.ADAMW:
            base_optim = optax.adamw(learning_rate, weight_decay=weight_decay)
        case Optimizer.MUON:
            base_optim = optax.contrib.muon(
                learning_rate=learning_rate, weight_decay=weight_decay
            )
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if max_grad_norm is not None:
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            base_optim,
        )
    return base_optim


def create_dataset(
    config: Config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Create dataset arrays from configuration.

    Returns (ts_batched, solution, control_values).
    """
    dataset_name = config.experiment_config.dataset_name
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset name '{dataset_name}'. Available: {list(DATASETS.keys())}"
        )
    npz_path = DATASETS[dataset_name]["npz_path"]
    ts_batched, solution, control_values = prepare_dataset(npz_path)
    return ts_batched, solution, control_values
