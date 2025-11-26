import jax
import optax
from taming_the_ito_lyon.config import (
    Optimizer,
    Config,
    NCDEConfig,
    NRDEConfig,
)
from taming_the_ito_lyon.models import NeuralCDE, NeuralRDE
from taming_the_ito_lyon.config import DATASETS
from taming_the_ito_lyon.data.datasets import prepare_dataset


def create_model(
    config: Config,
    *,
    input_path_dim: int,
    output_path_dim: int,
    key: jax.Array,
) -> NeuralCDE | NeuralRDE:
    match config.nn_config:
        case NCDEConfig():
            return NeuralCDE(
                input_path_dim=input_path_dim,
                hidden_size=config.nn_config.hidden_size,
                width_size=config.nn_config.width_size,
                depth=config.nn_config.depth,
                output_path_dim=output_path_dim,
                key=key,
                rtol=config.nn_config.rtol,
                atol=config.nn_config.atol,
                dtmin=config.nn_config.dtmin,
            )
        case NRDEConfig():
            return NeuralRDE(
                input_path_dim=input_path_dim,
                hidden_size=config.nn_config.hidden_size,
                width_size=config.nn_config.width_size,
                depth=config.nn_config.depth,
                output_path_dim=output_path_dim,
                signature_depth=config.nn_config.signature_depth,
                signature_window_size=int(config.nn_config.signature_window_size),
                key=key,
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
) -> optax.GradientTransformation:
    match optimizer_name:
        case Optimizer.ADAM:
            return optax.adam(learning_rate)
        case Optimizer.ADAMW:
            return optax.adamw(learning_rate, weight_decay=weight_decay)
        case Optimizer.MUON:
            raise NotImplementedError("Muon optimizer not implemented")
        case _:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_dataset(
    config: Config, *, key: jax.Array
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """
    Create dataset arrays from configuration.

    Returns (ts_batched, solution, coeffs_batched).
    """
    dataset_name = config.experiment_config.dataset_name
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset name '{dataset_name}'. Available: {list(DATASETS.keys())}"
        )
    npz_path = DATASETS[dataset_name]["npz_path"]
    ts_batched, solution, coeffs_batched = prepare_dataset(npz_path, key=key)
    return ts_batched, solution, coeffs_batched
