import jax
import jax.numpy as jnp
import optax
from taming_the_ito_lyon.config import (
    Optimizer,
    Config,
    NCDEConfig,
    LogNCDEConfig,
    NRDEConfig,
    MNRDEConfig,
    GRUConfig,
)
from taming_the_ito_lyon.config.config_options import (
    LossType,
    StepsizeControllerType,
    ManifoldType,
    HopfAlgebraType,
)
from stochastax.manifolds import Manifold, EuclideanSpace, SO3
from stochastax.manifolds.spd import SPDManifold
from diffrax import ConstantStepSize, PIDController
from taming_the_ito_lyon.models import (
    NeuralCDE,
    LogNCDE,
    NeuralRDE,
    MNDRE,
    GRU,
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
from taming_the_ito_lyon.training.results_gathering_fns import (
    ResultsGatheringFn,
)


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
    if not isinstance(
        config.nn_config,
        (NCDEConfig, LogNCDEConfig, NRDEConfig, MNRDEConfig, GRUConfig),
    ):
        return key, None

    scheme_enum = config.experiment_config.extrapolation_scheme
    n_recon = config.experiment_config.n_recon
    assert scheme_enum is not None
    assert n_recon is not None

    model_key, scheme_key = jax.random.split(key)
    # In extrapolation mode the model consumes controls with a prepended time
    # channel (dimension = input_channels + 1), but the extrapolation schemes are
    # fit on the raw driver channels without time. In particular, the MLP-based
    # schemes' encoder/decoder operate on value channels only.
    scheme_input_dim = int(input_path_dim) - 1
    extrapolation_scheme = create_scheme(
        scheme_enum,
        num_points=n_recon,
        input_dim=scheme_input_dim,
        key=scheme_key,
    )
    return model_key, extrapolation_scheme


def create_manifold_from_type(
    manifold_type: ManifoldType,
) -> type[Manifold]:
    match manifold_type:
        case ManifoldType.EUCLIDEAN:
            return EuclideanSpace
        case ManifoldType.SO3:
            return SO3
        case ManifoldType.SPD:
            return SPDManifold
        case _:
            raise ValueError(f"Unknown manifold: {manifold_type}")


def create_stepsize_controller(
    config: Config,
) -> ConstantStepSize | PIDController:
    match config.solver_config.stepsize_controller:
        case StepsizeControllerType.PID:
            return PIDController(
                rtol=config.solver_config.rtol,
                atol=config.solver_config.atol,
                dtmin=config.solver_config.dtmin,
            )
        case StepsizeControllerType.CONSTANT:
            return ConstantStepSize()
        case _:
            raise ValueError(
                f"Unknown stepsize controller: {config.solver_config.stepsize_controller}"
            )


def create_model(
    config: Config,
    *,
    input_path_dim: int,
    output_path_dim: int,
    key: jax.Array,
) -> Model:
    model_key, extrapolation_scheme = _maybe_create_extrapolation_scheme(
        config, input_path_dim=input_path_dim, key=key
    )

    manifold = create_manifold_from_type(config.experiment_config.manifold)
    hidden_manifold = create_manifold_from_type(
        config.experiment_config.hidden_manifold
    )
    stepsize_controller = create_stepsize_controller(config)
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
                manifold=manifold,
                stepsize_controller=stepsize_controller,
                evolving_out=config.experiment_config.evolving_out,
                extrapolation_scheme=extrapolation_scheme,
                n_recon=config.experiment_config.n_recon,
                control_interpolation=config.nn_config.control_interpolation,
            )
        case LogNCDEConfig():
            return LogNCDE(
                input_path_dim=input_path_dim,
                cde_state_dim=config.nn_config.cde_state_dim,
                init_hidden_dim=config.nn_config.init_hidden_dim,
                vf_hidden_dim=config.nn_config.vf_hidden_dim,
                initial_cond_mlp_depth=config.nn_config.initial_cond_mlp_depth,
                vf_mlp_depth=config.nn_config.vf_mlp_depth,
                output_path_dim=output_path_dim,
                signature_depth=config.nn_config.signature_depth,
                signature_window_size=config.nn_config.signature_window_size,
                stepsize_controller=stepsize_controller,
                extrapolation_scheme=extrapolation_scheme,
                n_recon=config.experiment_config.n_recon,
                key=model_key,
            )
        case NRDEConfig():
            return NeuralRDE(
                input_path_dim=input_path_dim,
                cde_state_dim=config.nn_config.cde_state_dim,
                vf_hidden_dim=config.nn_config.vf_hidden_dim,
                init_hidden_dim=config.nn_config.init_hidden_dim,
                initial_cond_mlp_depth=config.nn_config.initial_cond_mlp_depth,
                vf_mlp_depth=config.nn_config.vf_mlp_depth,
                output_path_dim=output_path_dim,
                signature_depth=config.nn_config.signature_depth,
                signature_window_size=config.nn_config.signature_window_size,
                stepsize_controller=stepsize_controller,
                extrapolation_scheme=extrapolation_scheme,
                n_recon=config.experiment_config.n_recon,
                key=model_key,
            )
        case MNRDEConfig():
            brownian_channels = config.nn_config.brownian_channels
            return MNDRE(
                input_path_dim=input_path_dim,
                cde_state_dim=config.nn_config.cde_state_dim,
                initial_hidden_dim=config.nn_config.init_hidden_dim,
                vf_hidden_dim=config.nn_config.vf_hidden_dim,
                initial_cond_mlp_depth=config.nn_config.initial_cond_mlp_depth,
                vf_mlp_depth=config.nn_config.vf_mlp_depth,
                output_path_dim=output_path_dim,
                signature_depth=config.nn_config.signature_depth,
                signature_window_size=config.nn_config.signature_window_size,
                data_manifold=manifold,
                hidden_manifold=hidden_manifold,
                hopf_algebra_type=config.nn_config.hopf_algebra,
                stepsize_controller=stepsize_controller,
                extrapolation_scheme=extrapolation_scheme,
                n_recon=config.experiment_config.n_recon,
                brownian_channels=brownian_channels,
                brownian_corr=0.0,
                key=model_key,
            )
        case GRUConfig():
            # GRU expects a manifold *instance*, while the CDE/RDE models use the
            # manifold type directly (class methods). We instantiate it here.
            return GRU(
                input_path_dim=input_path_dim,
                gru_state_dim=config.nn_config.gru_state_dim,
                output_path_dim=output_path_dim,
                mlp_hidden_dim=config.nn_config.init_hidden_dim,
                initial_cond_mlp_depth=config.nn_config.initial_cond_mlp_depth,
                key=model_key,
                manifold=manifold(),
                hidden_manifold=hidden_manifold(),
                evolving_out=config.experiment_config.evolving_out,
                extrapolation_scheme=extrapolation_scheme,
                n_recon=config.experiment_config.n_recon,
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
        case (
            Datasets.BLACK_SCHOLES
            | Datasets.BERGOMI
            | Datasets.ROUGH_BERGOMI
            | Datasets.SIMPLE_RBERGOMI
        ):
            from taming_the_ito_lyon.data.rough_volatility import RoughVolatilityDataset
            from taming_the_ito_lyon.data.simple_rough_volatility import (
                SimpleRoughVolatilityDataset,
            )

            dataset_cls: (
                type[RoughVolatilityDataset] | type[SimpleRoughVolatilityDataset]
            )
            if config.experiment_config.dataset_name == Datasets.SIMPLE_RBERGOMI:
                dataset_cls = SimpleRoughVolatilityDataset
            else:
                dataset_cls = RoughVolatilityDataset

            train = dataset_cls(config=config, split="train").make_array_source()
            val = dataset_cls(config=config, split="val").make_array_source()
            test = dataset_cls(config=config, split="test").make_array_source()
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
        case (
            Datasets.OXFORD_MULTIMOTION_STATIC
            | Datasets.OXFORD_MULTIMOTION_TRANSLATIONAL
            | Datasets.OXFORD_MULTIMOTION_UNCONSTRAINED
        ):
            from taming_the_ito_lyon.data.oxford_multimotion import (
                OxfordMultimotionDataset,
            )

            train = OxfordMultimotionDataset(
                config=config,
                split="train",
            ).make_disk_source()
            val = OxfordMultimotionDataset(
                config=config,
                split="val",
            ).make_disk_source()
            test = OxfordMultimotionDataset(
                config=config,
                split="test",
            ).make_disk_source()
        case Datasets.SPD_COVARIANCE_SOLAR:
            from taming_the_ito_lyon.data.spd_covariance import SPDCovarianceDataset

            train = SPDCovarianceDataset(
                config=config,
                split="train",
            ).make_array_source()
            val = SPDCovarianceDataset(
                config=config,
                split="val",
            ).make_array_source()
            test = SPDCovarianceDataset(
                config=config,
                split="test",
            ).make_array_source()
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
    driver_dim: int,
    anchor_at_basepoint: bool = True,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Create an unconditional control sampler.

    Returns a function `(ts, key) -> control_values` of shape (T, driver_dim + 1),
    where the leading channel is `ts` and the remaining channels are the sampled
    driver values on the same grid.
    """
    import jax.numpy as jnp
    from stochastax.controls.drivers import (
        bm_driver,
    )

    def with_time(ts: jax.Array, values: jax.Array) -> jax.Array:
        return jnp.concatenate([ts[:, None], values], axis=-1)

    def _anchor_at_basepoint(values: jax.Array) -> jax.Array:
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

        values = bm_driver(key, timesteps=timesteps, dim=int(driver_dim)).path
        if anchor_at_basepoint:
            values = _anchor_at_basepoint(values)
        return with_time(ts, values)

    return sample


def create_unconditional_control_sampler_batched(
    *,
    driver_dim: int = 1,
    anchor_at_basepoint: bool = True,
) -> Callable[[jax.Array, jax.Array, int], jax.Array]:
    """
    Create a batched unconditional control sampler.

    Returns a function `(ts, key, batch_size) -> control_values_batch` of shape
    (batch_size, T, driver_dim + 1), where the leading channel is `ts` and the
    remaining channels are the sampled driver values on the same grid.
    """
    import jax.random as jr

    single_sampler = create_unconditional_control_sampler(
        driver_dim=driver_dim,
        anchor_at_basepoint=anchor_at_basepoint,
    )

    def sample_batch(ts: jax.Array, key: jax.Array, batch_size: int) -> jax.Array:
        keys = jr.split(key, batch_size)
        return jax.vmap(lambda k: single_sampler(ts, k))(keys)

    # JIT this so unconditional mode doesn't run eager JAX work each step.
    # Compiles once per distinct (static) batch_size.
    return jax.jit(sample_batch, static_argnames=("batch_size",))


def create_grad_batch_loss_fns(
    config: Config,
    *,
    output_path_dim: int | None = None,
) -> tuple[
    Callable[[Model, jax.Array, jax.Array, jax.Array], tuple[jax.Array, optax.Updates]],
    Callable[[Model, jax.Array, jax.Array, jax.Array], jax.Array],
    Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
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
        frobenius_loss,
        ito_level2_distribution_mmd_loss,
        _maybe_unvech_spd,
    )

    match config.experiment_config.loss:
        case LossType.MSE:
            loss_fn = mse_loss
        case LossType.RGE:
            loss_fn = rotational_geodesic_loss(config)
        case LossType.FROBENIUS:
            loss_fn = frobenius_loss(config)
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
            raise ValueError(f"Unknown loss type: {config.experiment_config.loss}")

    use_spd = config.experiment_config.manifold == ManifoldType.SPD
    if use_spd and config.experiment_config.loss == LossType.SIGKER:
        raise ValueError(
            "SIGKER loss expects vector-valued paths. For SPD outputs, use MSE or "
            "FROBENIUS and enable SPD manifold output."
        )

    use_ito_mmd_loss = (
        config.experiment_config.loss == LossType.SIGKER
        and isinstance(config.nn_config, MNRDEConfig)
        and config.nn_config.hopf_algebra == HopfAlgebraType.GL
        and config.experiment_config.dataset_name == Datasets.SIMPLE_RBERGOMI
    )
    ito_weight = float(config.experiment_config.ito_level2_mmd_weight)

    def batch_loss_fn(
        model: Model,
        control_values_b: jax.Array,
        target_b: jax.Array,
        gt_driver_b: jax.Array,
    ) -> jax.Array:
        preds = jax.vmap(model)(control_values_b)
        if use_spd:
            preds = _maybe_unvech_spd(preds)
            target_b = _maybe_unvech_spd(target_b)
        base = loss_fn(preds, target_b)

        if not use_ito_mmd_loss or ito_weight <= 0.0:
            return base

        # W_model: latent Brownian channel from controls. In unconditional mode the
        # control is (t, W), so time is channel 0 and must be excluded from
        # quadratic variation.
        w_model = control_values_b[..., 1] if int(control_values_b.shape[-1]) >= 2 else control_values_b[..., 0]

        # W_gt: simulator Brownian driver from dataset (no time channel).
        w_gt = gt_driver_b[..., 0]

        # X_model/X_gt: log-price paths (channel 0).
        x_model = preds[..., 0]
        x_gt = target_b[..., 0]

        ito_mmd = ito_level2_distribution_mmd_loss(
            w_model=w_model,
            x_model=x_model,
            w_gt=w_gt,
            x_gt=x_gt,
            include_level1=True,
            max_bandwidth_points=256,
        )
        return base + jnp.asarray(ito_weight, dtype=base.dtype) * ito_mmd

    def loss_on_preds_fn(
        preds: jax.Array,
        target_b: jax.Array,
        control_values_b: jax.Array,
        gt_driver_b: jax.Array,
    ) -> jax.Array:
        if use_spd:
            preds = _maybe_unvech_spd(preds)
            target_b = _maybe_unvech_spd(target_b)
        base = loss_fn(preds, target_b)
        if not use_ito_mmd_loss or ito_weight <= 0.0:
            return base

        w_model = control_values_b[..., 1] if int(control_values_b.shape[-1]) >= 2 else control_values_b[..., 0]
        w_gt = gt_driver_b[..., 0]
        x_model = preds[..., 0]
        x_gt = target_b[..., 0]
        ito_mmd = ito_level2_distribution_mmd_loss(
            w_model=w_model,
            x_model=x_model,
            w_gt=w_gt,
            x_gt=x_gt,
            include_level1=True,
            max_bandwidth_points=256,
        )
        return base + jnp.asarray(ito_weight, dtype=base.dtype) * ito_mmd

    return (
        eqx.filter_value_and_grad(batch_loss_fn),
        batch_loss_fn,
        eqx.filter_jit(loss_on_preds_fn),
    )


def configure_jax() -> None:
    """Configure global JAX settings (matmul precision and persistent compilation cache)."""
    import os
    import lovely_jax

    lovely_jax.monkey_patch()

    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update(
        "jax_compilation_cache_max_size",
        2048 * 1024 * 1024,  # 2GB
    )
    cache_dir = os.path.abspath("jax_cache")
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)


def create_results_gathering_fn(
    config: Config,
) -> ResultsGatheringFn:
    match config.experiment_config.dataset_name:
        case (
            Datasets.BLACK_SCHOLES
            | Datasets.BERGOMI
            | Datasets.ROUGH_BERGOMI
            | Datasets.SIMPLE_RBERGOMI
        ):
            from taming_the_ito_lyon.training.results_gathering_fns import (
                get_rough_volatility_results,
            )

            return get_rough_volatility_results
        case Datasets.SG_SO3_SIMULATION:
            from taming_the_ito_lyon.training.results_gathering_fns import (
                get_sg_so3_simulation_results,
            )

            return get_sg_so3_simulation_results
        case (
            Datasets.OXFORD_MULTIMOTION_STATIC
            | Datasets.OXFORD_MULTIMOTION_TRANSLATIONAL
            | Datasets.OXFORD_MULTIMOTION_UNCONSTRAINED
        ):
            from taming_the_ito_lyon.training.results_gathering_fns import (
                get_sg_so3_simulation_results,
            )

            return get_sg_so3_simulation_results
        case Datasets.SPD_COVARIANCE_SOLAR:
            from taming_the_ito_lyon.training.results_gathering_fns import (
                get_spd_covariance_results,
            )

            return get_spd_covariance_results
        case _:
            raise ValueError(
                f"Unknown dataset name: {config.experiment_config.dataset_name}"
            )
