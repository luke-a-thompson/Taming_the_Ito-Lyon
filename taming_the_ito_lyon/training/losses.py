from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Callable
from stochastax.manifolds.spd import SPDManifold
from taming_the_ito_lyon.config.config import Config
from taming_the_ito_lyon.config.config_options import ManifoldType


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
    depth: int = 6,
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
    from stochastax.control_lifts import compute_path_signature
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
            sig = compute_path_signature(
                path=path,
                depth=depth,
                hopf=hopf,
                mode="full",
            )
            return sig.flatten()

        phi_pred = jax.vmap(_phi)(pred)
        phi_target = jax.vmap(_phi)(target)

        # Dot-product kernel on features: k(x, y) = <phi(x), phi(y)>
        k_pp = jnp.mean(phi_pred @ phi_pred.T)
        k_tt = jnp.mean(phi_target @ phi_target.T)
        k_pt = jnp.mean(phi_pred @ phi_target.T)

        # Biased MMD^2 (kernel score):
        #   E[k(X,X')] + E[k(Y,Y')] - 2 E[k(X,Y)]
        #
        # Note: k_tt is independent of model params, so adding it shifts the loss
        # but does not change gradients.
        return k_pp + k_tt - 2.0 * k_pt

    return loss


def truncated_sig_loss_time_augmented(
    *,
    depth: int = 6,
    value_dim: int = 1,
    anchor_at_start: bool = True,
    prepend_zero_basepoint: bool = True,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Signature-kernel score loss on time-augmented paths.

    This is the recommended variant for **1D outputs** where plain signatures
    largely collapse to increment-only information.

    We construct 2D paths (t, x_t) with t in [0, 1], then compute truncated
    log-signatures as features and apply the same dot-product kernel score.
    """
    from stochastax.control_lifts import compute_path_signature
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
            sig = compute_path_signature(
                path=aug,
                depth=depth,
                hopf=hopf,
                mode="full",
            )
            return sig.flatten()

        phi_pred = jax.vmap(_phi)(pred)
        phi_target = jax.vmap(_phi)(target)

        k_pp = jnp.mean(phi_pred @ phi_pred.T)
        k_tt = jnp.mean(phi_target @ phi_target.T)
        k_pt = jnp.mean(phi_pred @ phi_target.T)
        return k_pp + k_tt - 2.0 * k_pt

    return loss


def _maybe_unvech_spd(
    x: jax.Array,
) -> jax.Array:
    if x.ndim >= 2 and x.shape[-2:] == (3, 3):
        return x
    if x.shape[-1] == 6:
        return SPDManifold.unvech(x)
    return x


def _ito_level12_feature_vectors(
    *,
    w_paths: jax.Array,
    x_paths: jax.Array,
    include_level1: bool = True,
) -> jax.Array:
    """Compute per-path Itô / branched level-1/2 features from (W, X).

    Args:
        w_paths: (B, T) latent/driver BM paths.
        x_paths: (B, T) output log-price paths.
        include_level1: If True, include sum(dW) and sum(dX) as extra features.

    Returns:
        (B, D) feature matrix with columns:
            [qvW, qvX, covWX, (optional: sum_dW, sum_dX)].
    """
    if w_paths.ndim != 2 or x_paths.ndim != 2:
        raise ValueError(
            f"Expected w_paths/x_paths shaped (B, T), got {w_paths.shape} and {x_paths.shape}"
        )
    if w_paths.shape != x_paths.shape:
        raise ValueError(
            f"w_paths and x_paths must have the same shape, got {w_paths.shape} and {x_paths.shape}"
        )
    if int(w_paths.shape[1]) < 2:
        b = int(w_paths.shape[0])
        d = 5 if include_level1 else 3
        return jnp.zeros((b, d), dtype=jnp.float32)

    dW = jnp.diff(w_paths, axis=1)  # (B, T-1)
    dX = jnp.diff(x_paths, axis=1)  # (B, T-1)

    qvW = jnp.sum(dW * dW, axis=1)
    qvX = jnp.sum(dX * dX, axis=1)
    covWX = jnp.sum(dW * dX, axis=1)

    if include_level1:
        sum_dW = jnp.sum(dW, axis=1)
        sum_dX = jnp.sum(dX, axis=1)
        feat = jnp.stack([qvW, qvX, covWX, sum_dW, sum_dX], axis=1)
    else:
        feat = jnp.stack([qvW, qvX, covWX], axis=1)

    return feat.astype(jnp.float32)


def _rbf_mmd2_median_heuristic(
    x: jax.Array,
    y: jax.Array,
    *,
    eps: float = 1e-12,
    max_bandwidth_points: int = 256,
) -> jax.Array:
    """Biased RBF-kernel MMD^2 with median-heuristic bandwidth."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D feature matrices, got {x.shape} and {y.shape}")
    if int(x.shape[1]) != int(y.shape[1]):
        raise ValueError(
            f"x and y must have the same feature dimension, got {x.shape} and {y.shape}"
        )

    nx = int(x.shape[0])
    ny = int(y.shape[0])
    if nx == 0 or ny == 0:
        return jnp.asarray(0.0, dtype=jnp.float32)

    pooled = jnp.concatenate([x, y], axis=0)
    m = min(int(max_bandwidth_points), int(pooled.shape[0]))
    pooled_m = pooled[:m]

    # Pairwise squared distances on pooled subset (m, m)
    pm_norm = jnp.sum(pooled_m * pooled_m, axis=1)
    pm_dist2 = pm_norm[:, None] + pm_norm[None, :] - 2.0 * (pooled_m @ pooled_m.T)
    pm_dist2 = jnp.maximum(pm_dist2, 0.0)

    eye = jnp.eye(int(pm_dist2.shape[0]), dtype=bool)
    pm_dist2_no_diag = jnp.where(eye, jnp.nan, pm_dist2)
    med = jnp.nanmedian(pm_dist2_no_diag)
    med = jnp.where(jnp.isfinite(med) & (med > 0.0), med, 1.0)
    med = jax.lax.stop_gradient(med)
    denom = 2.0 * med + float(eps)

    def _k(a: jax.Array, b: jax.Array) -> jax.Array:
        a_norm = jnp.sum(a * a, axis=1)
        b_norm = jnp.sum(b * b, axis=1)
        dist2 = a_norm[:, None] + b_norm[None, :] - 2.0 * (a @ b.T)
        dist2 = jnp.maximum(dist2, 0.0)
        return jnp.exp(-dist2 / denom)

    k_xx = _k(x, x)
    k_yy = _k(y, y)
    k_xy = _k(x, y)
    return (jnp.mean(k_xx) + jnp.mean(k_yy) - 2.0 * jnp.mean(k_xy)).astype(
        jnp.float32
    )


def ito_level2_distribution_mmd_loss(
    *,
    w_model: jax.Array,
    x_model: jax.Array,
    w_gt: jax.Array,
    x_gt: jax.Array,
    include_level1: bool = True,
    max_bandwidth_points: int = 256,
) -> jax.Array:
    """Itô/branched level-2 distribution-matching loss for log-price paths.

    Computes per-path level-1/2 features and matches them in distribution via an
    RBF-kernel MMD^2.
    """
    feat_model = _ito_level12_feature_vectors(
        w_paths=w_model, x_paths=x_model, include_level1=include_level1
    )
    feat_gt = _ito_level12_feature_vectors(
        w_paths=w_gt, x_paths=x_gt, include_level1=include_level1
    )
    return _rbf_mmd2_median_heuristic(
        feat_model, feat_gt, max_bandwidth_points=int(max_bandwidth_points)
    )


