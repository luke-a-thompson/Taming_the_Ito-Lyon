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
from taming_the_ito_lyon.config.config_options import (
    LossType,
    UnconditionalDriverKind,
)
from taming_the_ito_lyon.models import (
    NeuralCDE,
    LogNCDE,
    NeuralRDE,
    MNDRE,
    create_scheme,
    Model,
)
from taming_the_ito_lyon.models.extrapolation import (
    ExtrapolationScheme as ExtrapolationSchemeProtocol,
)
from taming_the_ito_lyon.config import Datasets
import equinox as eqx
from collections.abc import Callable
from cyreal.transforms import BatchTransform, DevicePutTransform
from cyreal.loader import DataLoader


def _maybe_create_extrapolation_scheme(
    config: Config,
    *,
    input_path_dim: int,
    key: jax.Array,
) -> tuple[jax.Array, ExtrapolationSchemeProtocol | None]:
    """Optionally create an extrapolation scheme and (if needed) split the PRNG key."""

    if config.experiment_config.extrapolation_scheme is None:
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
                initial_hidden_dim=config.nn_config.initial_hidden_dim,
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


def create_dataloaders(
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    match config.experiment_config.dataset_name:
        case Datasets.BLACK_SCHOLES | Datasets.BERGOMI | Datasets.ROUGH_BERGOMI:
            from taming_the_ito_lyon.data.rough_volatility import RoughVolatilityDataset

            train = RoughVolatilityDataset(
                config=config,
                split="train",
            ).make_array_source()
            val = RoughVolatilityDataset(
                config=config,
                split="val",
            ).make_array_source()
            test = RoughVolatilityDataset(
                config=config,
                split="test",
            ).make_array_source()
        case Datasets.SG_SO3_SIMULATION:
            from taming_the_ito_lyon.data.so3_dynamics_sim import SO3DynamicsSim

            train = SO3DynamicsSim(
                config=config,
                split="train",
            ).make_disk_source()
            val = SO3DynamicsSim(
                config=config,
                split="val",
            ).make_disk_source()
            test = SO3DynamicsSim(
                config=config,
                split="test",
            ).make_disk_source()
        case _:
            raise ValueError(
                f"Unknown dataset name: {config.experiment_config.dataset_name}"
            )

    dataloaders: list[DataLoader] = []
    for source in [train, val, test]:
        pipeline = [
            source,
            BatchTransform(
                batch_size=config.experiment_config.batch_size, drop_last=True
            ),
            DevicePutTransform(),
        ]
        dataloader = DataLoader(pipeline)
        dataloader.init_state(jax.random.key(config.experiment_config.seed))
        dataloaders.append(dataloader)
    return dataloaders[0], dataloaders[1], dataloaders[2]  # train, val, test


def create_unconditional_control_sampler(
    *,
    driver_kind: UnconditionalDriverKind,
    driver_dim: int,
    hurst: float,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Create an unconditional control sampler.

    Returns a function `(ts, key) -> control_values` of shape (T, driver_dim + 1),
    where the leading channel is `ts` and the remaining channels are the sampled
    driver values on the same grid.
    """
    import jax.numpy as jnp
    import jax.random as jr
    from stochastax.controls.drivers import (
        bm_driver,
        fractional_bm_driver,
        riemann_liouville_driver,
    )

    def with_time(ts: jax.Array, values: jax.Array) -> jax.Array:
        return jnp.concatenate([ts[:, None], values], axis=-1)

    def anchor_at_basepoint(values: jax.Array) -> jax.Array:
        """Anchor the path at the origin without changing its length.

        Note: "basepoint augmentation" in the signature literature often means
        *prepending an extra point* at the start of the path. In this codebase the
        model/targets assume a fixed length `T`, so instead we simply translate the
        path to start at 0 (which does not affect signatures, as they depend on
        increments).
        """
        return values - values[:1]

    def sample(ts: jax.Array, key: jax.Array) -> jax.Array:
        timesteps = int(ts.shape[0]) - 1
        if timesteps <= 0:
            raise ValueError(f"ts must have length >= 2, got {ts.shape[0]}")

        if driver_kind == UnconditionalDriverKind.BM:
            values = bm_driver(key, timesteps=timesteps, dim=driver_dim).path
            values = anchor_at_basepoint(values)
            return with_time(ts, values)

        if driver_kind == UnconditionalDriverKind.FBM:
            values = fractional_bm_driver(
                key, timesteps=timesteps, dim=driver_dim, hurst=float(hurst)
            ).path
            values = anchor_at_basepoint(values)
            return with_time(ts, values)

        if driver_kind == UnconditionalDriverKind.RL:
            bm_key, rl_key = jr.split(key, 2)
            bm_path = bm_driver(bm_key, timesteps=timesteps, dim=driver_dim)
            values = riemann_liouville_driver(
                rl_key, timesteps=timesteps, hurst=float(hurst), bm_path=bm_path
            ).path
            values = anchor_at_basepoint(values)
            return with_time(ts, values)

        raise ValueError(f"Unknown driver_kind: {driver_kind}")

    return sample


def create_unconditional_control_sampler_batched(
    *,
    driver_kind: UnconditionalDriverKind,
    driver_dim: int,
    hurst: float,
) -> Callable[[jax.Array, jax.Array, int], jax.Array]:
    """
    Create a batched unconditional control sampler.

    Returns a function `(ts, key, batch_size) -> control_values_batch` of shape
    (batch_size, T, driver_dim + 1), where the leading channel is `ts` and the
    remaining channels are the sampled driver values on the same grid.
    """
    import jax.random as jr

    single_sampler = create_unconditional_control_sampler(
        driver_kind=driver_kind, driver_dim=driver_dim, hurst=hurst
    )

    def sample_batch(ts: jax.Array, key: jax.Array, batch_size: int) -> jax.Array:
        keys = jr.split(key, batch_size)
        return jax.vmap(lambda k: single_sampler(ts, k))(keys)

    return sample_batch


def create_grad_batch_loss_fns(
    *,
    loss_type: LossType,
    output_path_dim: int | None = None,
) -> tuple[
    Callable[[Model, jax.Array, jax.Array], tuple[jax.Array, optax.Updates]],
    Callable[[Model, jax.Array, jax.Array], jax.Array],
]:
    """
    Create (grad_fn, batch_loss_fn) for training and evaluation.

    Both returned functions share the same call signature:
        (model, control_values_b, target_b)

    where `control_values_b` is a batch of control paths that will be fed to the model.
    """
    from taming_the_ito_lyon.training.losses import (
        mse_loss,
        rotational_geodesic_loss,
        truncated_sig_loss_time_augmented,
    )

    match loss_type:
        case LossType.MSE:
            loss_fn = mse_loss
        case LossType.RGE:
            loss_fn = rotational_geodesic_loss
        case LossType.SIGKER:
            if output_path_dim is None:
                raise ValueError(
                    "output_path_dim must be provided when loss_type is SIGKER so the "
                    "Hopf algebra can be constructed outside of jit."
                )
            # Time augmentation is important, especially for 1D outputs.
            #
            # IMPORTANT: signatures/log-signatures depend on increments (dx), so they
            # are translation-invariant in the value channels. If the absolute level
            # matters (e.g. matching initial level "h0"/v0), then we must explicitly
            # encode it. We do that via a zero-basepoint prepend, which makes x0 an
            # increment and therefore visible to signature features.
            loss_fn = truncated_sig_loss_time_augmented(
                value_dim=int(output_path_dim),
                anchor_at_start=False,
                prepend_zero_basepoint=True,
            )
        case _:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def batch_loss_fn(
        model: Model,
        control_values_b: jax.Array,
        target_b: jax.Array,
    ) -> jax.Array:
        preds = jax.vmap(model)(control_values_b)
        loss = loss_fn(preds, target_b)
        return loss

    return eqx.filter_value_and_grad(batch_loss_fn), batch_loss_fn


def configure_jax() -> None:
    """Configure global JAX settings (matmul precision and persistent compilation cache)."""
    import os

    jax.config.update("jax_default_matmul_precision", "high")
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update(
        "jax_compilation_cache_max_size",
        2048 * 1024 * 1024,  # 2GB
    )
    cache_dir = os.path.abspath("jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
