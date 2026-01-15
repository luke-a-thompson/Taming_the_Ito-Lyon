import jax
import jax.numpy as jnp
from typing import Callable
from taming_the_ito_lyon.config.config import Config


def mse_loss(
    pred: jax.Array,
    target: jax.Array,
) -> jax.Array:
    assert pred.shape == target.shape, (
        f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
    )
    return jnp.mean((pred - target) ** 2)


def frobenius_loss(
    config: Config,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Frobenius loss between predicted and target rotation matrices.

    If extrapolation_scheme is set, we compute the loss on the predicted and target
    rotation matrices for the reconstruction and future parts separately.

    Args:
        config: Config object

    Returns:
        Loss function
    """
    n_recon = config.experiment_config.n_recon

    def loss(
        pred: jax.Array,
        target: jax.Array,
    ) -> jax.Array:
        return jnp.mean(jnp.linalg.norm(pred - target, ord="fro", axis=(-2, -1)))

    def extrapolation_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        pred = pred[n_recon:]
        target = target[n_recon:]
        recon_loss = loss(pred, target)
        future_loss = loss(target, pred)
        return recon_loss + future_loss

    if config.experiment_config.extrapolation_scheme is not None:
        return extrapolation_loss
    else:
        return loss


def rotational_geodesic_loss(
    config: Config,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Rotational Geodesic Error (RGE) loss.
    RGE(R1, R2) = 2 * arcsin(||R2 - R1||_F / (2√2))

    Args:
        config: Config object
    """
    n_recon = config.experiment_config.n_recon

    def loss(
        pred: jax.Array,
        target: jax.Array,
    ) -> jax.Array:
        """
        Compute the Rotational Geodesic Error (RGE) loss.
        RGE(R1, R2) = 2 * arcsin(||R2 - R1||_F / (2√2))

        Args:
            pred: Predicted rotation matrices
            target: Target rotation matrices

        Returns:
            Mean RGE loss
        """
        assert pred.shape == target.shape, (
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
        assert pred.shape[-1] == pred.shape[-2], (
            "pred and target must be square matrices"
        )
        # NOTE: The closed-form RGE formula assumes both inputs are valid rotation matrices.
        # In practice, simulator outputs can drift slightly off SO(3) and floating point error
        # can push the arcsin argument marginally outside [-1, 1], producing NaNs.
        frobenius_norm = jnp.linalg.norm(pred - target, ord="fro", axis=(-2, -1))
        denom = 2.0 * jnp.sqrt(2.0)
        ratio = frobenius_norm / denom
        # Also avoid the arcsin derivative singularity at 1.0, which can create `inf`
        # gradients and quickly destabilize optimization early in training.
        eps = jnp.asarray(1e-5, dtype=ratio.dtype)
        ratio = jnp.clip(ratio, a_min=0.0, a_max=1.0 - eps)
        rge_rad = 2.0 * jnp.arcsin(ratio)
        rge_deg = rge_rad * (180.0 / jnp.pi)
        return jnp.mean(rge_deg)

    def extrapolation_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        pred = pred[n_recon:]
        target = target[n_recon:]
        recon_loss = loss(pred, target)
        future_loss = loss(target, pred)
        return recon_loss + future_loss

    if config.experiment_config.extrapolation_scheme is not None:
        return extrapolation_loss
    else:
        return loss


def truncated_sig_loss(
    depth: int = 4,
    ambient_dim: int = 1,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Create a signature(-like) loss using truncated *log signatures* as features.

    We compute a truncated log signature for each path (depth=`depth`), flatten
    it into a feature vector, then apply a dot-product kernel.

    This loss implements the **signature kernel score** (see e.g. Eq. in the
    screenshot you referenced):

        φ(P, y) := E_{x,x'~P}[k(x,x')] - 2 E_{x~P}[k(x,y)]

    """
    from stochastax.control_lifts import compute_log_signature
    from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra

    hopf = ShuffleHopfAlgebra.build(ambient_dim=int(ambient_dim), depth=int(depth))

    def loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        # following https://arxiv.org/pdf/2305.16274 remark 3.2
        assert pred.shape == target.shape, (
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
        assert int(pred.shape[-1]) == int(ambient_dim), (
            f"Expected path feature dimension {int(ambient_dim)}, got {int(pred.shape[-1])}. "
            "Pass the correct `ambient_dim` when constructing the loss."
        )

        def _phi(path: jax.Array) -> jax.Array:
            logsig = compute_log_signature(
                path=path,
                depth=depth,
                hopf=hopf,
                log_signature_type="Lyndon words",
                mode="full",
            )
            return logsig.flatten()

        phi_pred = jax.vmap(_phi)(pred)
        phi_target = jax.vmap(_phi)(target)

        # Dot-product kernel on features: k(x, y) = <phi(x), phi(y)>
        k_pp = jnp.mean(phi_pred @ phi_pred.T)
        k_pt = jnp.mean(phi_pred @ phi_target.T)

        # Expected kernel score averaged over y~target:
        # E_y[ φ(P, y) ] = E_{x,x'~P}[k(x,x')] - 2 E_{x~P, y~target}[k(x,y)]
        return k_pp - 2.0 * k_pt

    return loss


def truncated_sig_loss_time_augmented(
    *,
    depth: int = 4,
    value_dim: int = 1,
    anchor_at_start: bool = True,
    prepend_zero_basepoint: bool = False,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Signature-kernel score loss on time-augmented paths.

    This is the recommended variant for **1D outputs** where plain signatures
    largely collapse to increment-only information.

    We construct 2D paths (t, x_t) with t in [0, 1], then compute truncated
    log-signatures as features and apply the same dot-product kernel score.
    """
    from stochastax.control_lifts import compute_log_signature
    from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra

    ambient_dim = int(value_dim) + 1
    hopf = ShuffleHopfAlgebra.build(ambient_dim=int(ambient_dim), depth=int(depth))

    def loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        assert pred.shape == target.shape, (
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
        assert int(pred.shape[-1]) == int(value_dim), (
            f"Expected value dimension {int(value_dim)}, got {int(pred.shape[-1])}. "
            "Pass the correct `value_dim` when constructing the loss."
        )

        length = int(pred.shape[-2])
        ts = jnp.linspace(0.0, 1.0, length, dtype=pred.dtype)  # (T,)
        ts_col = ts[:, None]  # (T, 1)

        def _augment(path: jax.Array) -> jax.Array:
            if anchor_at_start:
                path = path - path[:1]
            aug = jnp.concatenate([ts_col, path], axis=-1)  # (T, 1+value_dim)
            if not prepend_zero_basepoint:
                return aug

            # Signatures/log-signatures depend on path *increments* (dx), hence are
            # invariant to adding a constant offset to x. If we care about absolute
            # level (e.g. matching x0 / "h0"), we must explicitly encode it.
            #
            # Prepending a zero basepoint makes the first increment equal to the
            # initial level, which then appears in the signature features.
            #
            # Note: we duplicate t=0 at the first two points; this is fine for
            # signature computation since it depends on increments, not on dt.
            zero0 = jnp.zeros((1, int(ambient_dim)), dtype=aug.dtype)
            return jnp.concatenate([zero0, aug], axis=0)  # (T+1, 1+value_dim)

        def _phi(path: jax.Array) -> jax.Array:
            aug = _augment(path)
            logsig = compute_log_signature(
                path=aug,
                depth=depth,
                hopf=hopf,
                log_signature_type="Lyndon words",
                mode="full",
            )
            return logsig.flatten()

        phi_pred = jax.vmap(_phi)(pred)
        phi_target = jax.vmap(_phi)(target)

        k_pp = jnp.mean(phi_pred @ phi_pred.T)
        k_pt = jnp.mean(phi_pred @ phi_target.T)
        return k_pp - 2.0 * k_pt

    return loss
