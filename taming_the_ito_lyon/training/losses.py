from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Callable
from stochastax.manifolds.spd import SPDManifold
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
    depth: int = 5,
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


def branched_sigker_loss(
    *,
    depth: int = 5,
    use_planar: bool,
    use_time: bool,
    use_w: bool,
    anchor_at_start: bool = True,
    prepend_zero_basepoint: bool = True,
) -> Callable[[jax.Array, jax.Array, jax.Array | None, jax.Array | None], jax.Array]:
    """Branched signature-kernel score loss (biased MMD^2 with dot-product kernel).

    This mirrors `truncated_sig_loss_time_augmented`, but uses a **branched Itô**
    signature (GL/MKW Hopf algebra) as the feature map. Quadratic covariation is
    injected via `cov_increments`, interpreted as per-step increments `Δ⟨Y⟩_k`.

    Shapes
    ------
    - x_paths: (B, T)
    - w_paths: (B, T) if `use_w=True`, otherwise ignored

    Constructed multi-channel path Y has shape (T, d):
    - if use_time and use_w: Y = (t, w, x), d=3
    - if use_time and not use_w: Y = (t, x), d=2
    - if not use_time and use_w: Y = (w, x), d=2
    - if not use_time and not use_w: Y = (x,), d=1 (usually not recommended)
    """
    from stochastax.control_lifts import (
        compute_nonplanar_branched_signature,
        compute_planar_branched_signature,
    )
    from stochastax.hopf_algebras import GLHopfAlgebra, MKWHopfAlgebra

    depth_i = int(depth)
    if depth_i <= 0:
        raise ValueError("depth must be >= 1")

    d = (1 if use_time else 0) + (1 if use_w else 0) + 1

    hopf_planar = (
        MKWHopfAlgebra.build(ambient_dim=int(d), depth=int(depth_i)) if use_planar else None
    )
    hopf_nonplanar = (
        GLHopfAlgebra.build(ambient_dim=int(d), depth=int(depth_i)) if not use_planar else None
    )

    def _ensure_2d_paths(x: jax.Array, name: str) -> jax.Array:
        if x.ndim == 3 and int(x.shape[-1]) == 1:
            x = x[..., 0]
        if x.ndim != 2:
            raise ValueError(f"Expected {name} shaped (B, T), got {x.shape}")
        return x

    def loss(
        pred_x: jax.Array,
        target_x: jax.Array,
        pred_w: jax.Array | None = None,
        target_w: jax.Array | None = None,
    ) -> jax.Array:
        pred_x_ = _ensure_2d_paths(pred_x, "pred_x")
        target_x_ = _ensure_2d_paths(target_x, "target_x")
        if pred_x_.shape != target_x_.shape:
            raise ValueError(
                f"pred_x and target_x must have the same shape, got {pred_x_.shape} and {target_x_.shape}"
            )

        if use_w:
            if pred_w is None or target_w is None:
                raise ValueError("use_w=True requires pred_w and target_w.")
            pred_w_ = _ensure_2d_paths(pred_w, "pred_w")
            target_w_ = _ensure_2d_paths(target_w, "target_w")
            if pred_w_.shape != pred_x_.shape or target_w_.shape != target_x_.shape:
                raise ValueError(
                    "pred_w/target_w must match pred_x/target_x shapes, got "
                    f"{pred_w_.shape}, {target_w_.shape} vs {pred_x_.shape}."
                )
        else:
            pred_w_ = None
            target_w_ = None

        B, T = pred_x_.shape
        if int(T) < 2 or int(B) < 1:
            return jnp.asarray(0.0, dtype=jnp.float32)

        ts = jnp.linspace(0.0, 1.0, int(T), dtype=pred_x_.dtype)  # (T,)

        def _augment_single(x_path: jax.Array, w_path: jax.Array | None) -> jax.Array:
            if anchor_at_start:
                x_path = x_path - x_path[:1]
                if w_path is not None:
                    w_path = w_path - w_path[:1]

            cols: list[jax.Array] = []
            if use_time:
                cols.append(ts)
            if use_w:
                assert w_path is not None
                cols.append(w_path)
            cols.append(x_path)
            y = jnp.stack(cols, axis=1)  # (T, d)

            if not prepend_zero_basepoint:
                return y
            zero0 = jnp.zeros((1, int(d)), dtype=y.dtype)
            return jnp.concatenate([zero0, y], axis=0)  # (T+1, d)

        def _cov_increments_from_path(path: jax.Array) -> jax.Array:
            # `cov_increments[k]` is interpreted as the per-step quadratic covariation
            # increment Δ⟨Y⟩_k for the *true* sampled path.
            #
            # If we prepend a zero basepoint, the first increment (0 -> y0) is an
            # artificial encoding of the initial level and must NOT contribute to
            # quadratic covariation. So we pad a leading zero and compute covariation
            # only from the true increments.
            if prepend_zero_basepoint:
                # path has shape (T+1, d); true increments are between path[1:] points.
                inc_true = jnp.diff(path[1:], axis=0)  # (T-1, d)
                cov_true = jnp.einsum("td,te->tde", inc_true, inc_true)  # (T-1, d, d)
                cov0 = jnp.zeros((1, int(d), int(d)), dtype=path.dtype)  # (1, d, d)
                cov = jnp.concatenate([cov0, cov_true], axis=0)  # (T, d, d)
            else:
                inc = jnp.diff(path, axis=0)  # (T-1, d)
                cov = jnp.einsum("td,te->tde", inc, inc)  # (T-1, d, d)
            if use_time:
                cov = cov.at[:, 0, :].set(0.0)
                cov = cov.at[:, :, 0].set(0.0)
            # Outer products are symmetric; time-zeroing preserves symmetry.
            return cov

        def _phi_single(x_path: jax.Array, w_path: jax.Array | None) -> jax.Array:
            path = _augment_single(x_path, w_path)
            cov_inc = _cov_increments_from_path(path)
            if use_planar:
                assert hopf_planar is not None
                sig = compute_planar_branched_signature(
                    path,
                    depth_i,
                    hopf_planar,
                    "full",
                    cov_increments=cov_inc,
                )
            else:
                assert hopf_nonplanar is not None
                sig = compute_nonplanar_branched_signature(
                    path,
                    depth_i,
                    hopf_nonplanar,
                    "full",
                    cov_increments=cov_inc,
                )
            return sig.log().flatten()

        if use_w:
            assert pred_w_ is not None and target_w_ is not None
            phi_pred = jax.vmap(_phi_single)(pred_x_, pred_w_)
            phi_target = jax.vmap(_phi_single)(target_x_, target_w_)
        else:
            phi_pred = jax.vmap(lambda x: _phi_single(x, None))(pred_x_)
            phi_target = jax.vmap(lambda x: _phi_single(x, None))(target_x_)

        k_pp = jnp.mean(phi_pred @ phi_pred.T)
        k_tt = jnp.mean(phi_target @ phi_target.T)
        k_pt = jnp.mean(phi_pred @ phi_target.T)
        return (k_pp + k_tt - 2.0 * k_pt).astype(jnp.float32)

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


