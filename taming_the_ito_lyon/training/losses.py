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

    We construct augmented paths (t, x_t) with t in [0, 1], then compute truncated
    log-signatures as features and apply the same dot-product kernel score.

    This loss accepts either:
    - vector-valued paths shaped (B, T, C), or
    - 3x3 matrix-valued paths shaped (B, T, 3, 3). These are interpreted as either
      SO(3) rotations (mapped via a log-map to (T, 3)) or SPD matrices (mapped via
      `SPDManifold.vech` to (T, 6)), depending on which structure the matrices
      satisfy.
    """
    from stochastax.control_lifts import compute_path_signature
    from stochastax.hopf_algebras.hopf_algebras import ShuffleHopfAlgebra
    from taming_the_ito_lyon.utils.so3 import log_map

    ambient_dim = int(value_dim) + 1
    hopf = ShuffleHopfAlgebra.build(ambient_dim=int(ambient_dim), depth=int(depth))
    # Special-case: if the provided paths are 3x3 matrices, convert them to a
    # Euclidean path inside the loss:
    # - SO(3): log-map of the relative rotation (T,3)
    # - SPD: vech(X) (T,6)
    so3_value_dim = 3
    hopf_so3 = ShuffleHopfAlgebra.build(
        ambient_dim=int(so3_value_dim) + 1, depth=int(depth)
    )

    def loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        assert pred.shape == target.shape, (
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
        is_matrix_3x3 = pred.ndim == 4 and pred.shape[-2:] == (3, 3)
        if not is_matrix_3x3:
            assert int(pred.shape[-1]) == int(value_dim), (
                f"Expected value dimension {int(value_dim)}, got {int(pred.shape[-1])}. "
                "Pass the correct `value_dim` when constructing the loss."
            )

        # `pred` is batched as (B, T, ...) for all datasets; for matrix-valued paths
        # the last two axes are (3,3), so we must NOT read time from `shape[-2]`.
        length = int(pred.shape[1])
        ts = jnp.linspace(0.0, 1.0, length, dtype=pred.dtype)  # (T,)
        ts_col = ts[:, None]  # (T, 1)

        def _to_euclidean_path(path: jax.Array) -> jax.Array:
            # path: (T, C) or (T, 3, 3)
            if path.ndim == 3 and path.shape[-2:] == (3, 3):
                # Disambiguate SO(3) vs SPD **statically** based on the configured
                # `value_dim`. This avoids `jax.lax.cond` shape constraints.
                #
                # - SO(3): value_dim=3, map via log-map to (T,3)
                # - SPD(3): value_dim=6, map via vech to (T,6)
                if int(value_dim) == 3:
                    r0 = path[:1]  # (1,3,3)
                    r0_t = jnp.swapaxes(r0, -1, -2)  # (1,3,3)
                    rel = r0_t @ path  # (T,3,3), broadcasts over T
                    return log_map(rel)  # (T,3)
                if int(value_dim) == 6:
                    return SPDManifold.vech(path)  # (T,6)
                raise ValueError(
                    "Matrix-valued paths require value_dim=3 (SO3) or value_dim=6 (SPD vech). "
                    f"Got value_dim={int(value_dim)}."
                )
            return path

        def _augment(path: jax.Array) -> jax.Array:
            path = _to_euclidean_path(path)
            if anchor_at_start:
                path = path - path[:1]
            aug = jnp.concatenate([ts_col, path], axis=-1)  # (T, 1+this_value_dim)
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
            zero0 = jnp.zeros((1, int(aug.shape[-1])), dtype=aug.dtype)
            return jnp.concatenate([zero0, aug], axis=0)  # (T+1, 1+value_dim)

        def _phi(path: jax.Array) -> jax.Array:
            aug = _augment(path)
            this_hopf = hopf_so3 if int(aug.shape[-1]) == (so3_value_dim + 1) else hopf
            sig = compute_path_signature(
                path=aug,
                depth=depth,
                hopf=this_hopf,
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
    depth: int = 4,
    use_planar: bool,
    use_time: bool,
    use_w: bool,
    x_dim: int = 1,
    anchor_at_start: bool = True,
    prepend_zero_basepoint: bool = True,
) -> Callable[[jax.Array, jax.Array, jax.Array | None, jax.Array | None], jax.Array]:
    """Branched signature-kernel score loss (biased MMD^2 with dot-product kernel).

    This mirrors `truncated_sig_loss_time_augmented`, but uses a **branched Itô**
    signature (GL/MKW Hopf algebra) as the feature map. Quadratic covariation is
    injected via `cov_increments`, interpreted as per-step increments `Δ⟨Y⟩_k`.

    Shapes
    ------
    - x_paths: (B, T, C) or (B, T) (treated as C=1). Also accepts SPD matrices
      shaped (B, T, 3, 3) and converts them to vech(X) internally.
    - w_paths: (B, T) if `use_w=True`, otherwise ignored.
    - If `use_w=False`, the optional `target_w` argument may be used as a
      per-step bracket *density* side-channel for x_paths, e.g. (B, T, 36)
      representing (6x6) densities for vech(X). This is used for Wishart/SPD
      datasets where quadratic variation is provided by the simulator.

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

    from stochastax.manifolds.spd import SPDManifold

    x_dim_i = int(x_dim)
    if x_dim_i <= 0:
        raise ValueError("x_dim must be >= 1")

    d_static = (1 if use_time else 0) + (1 if use_w else 0) + int(x_dim_i)

    hopf_planar = (
        MKWHopfAlgebra.build(ambient_dim=int(d_static), depth=int(depth_i))
        if use_planar
        else None
    )
    hopf_nonplanar = (
        GLHopfAlgebra.build(ambient_dim=int(d_static), depth=int(depth_i))
        if not use_planar
        else None
    )

    def _ensure_btc_paths(x: jax.Array, name: str) -> jax.Array:
        # Accept (B,T), (B,T,1), (B,T,C), or (B,T,3,3) (SPD matrices).
        if x.ndim == 2:
            return x[..., None]
        if x.ndim == 3:
            return x
        if x.ndim == 4 and x.shape[-2:] == (3, 3):
            # Convert SPD matrix path to vech path (B,T,6).
            b = int(x.shape[0])
            t = int(x.shape[1])
            x_flat = x.reshape((b * t,) + (3, 3))
            vech_flat = SPDManifold.vech(x_flat)  # (B*T, 6)
            return vech_flat.reshape((b, t, 6))
        raise ValueError(f"Expected {name} shaped (B,T), (B,T,C), or (B,T,3,3); got {x.shape}")

    def _maybe_parse_cov_density(cov: jax.Array | None, *, B: int, T: int) -> jax.Array | None:
        # Wishart dataset stores instantaneous bracket density for vech(X) as (B,T,36).
        if cov is None:
            return None
        if cov.ndim == 3 and int(cov.shape[0]) == int(B) and int(cov.shape[1]) == int(T):
            if int(cov.shape[2]) == int(x_dim_i * x_dim_i):
                return cov.reshape((B, T, int(x_dim_i), int(x_dim_i)))
        if cov.ndim == 4 and int(cov.shape[0]) == int(B) and int(cov.shape[1]) == int(T):
            if cov.shape[-2:] == (int(x_dim_i), int(x_dim_i)):
                return cov
        return None

    def loss(
        pred_x: jax.Array,
        target_x: jax.Array,
        pred_w: jax.Array | None = None,
        target_w: jax.Array | None = None,
    ) -> jax.Array:
        pred_x_btc = _ensure_btc_paths(pred_x, "pred_x")  # (B,T,C)
        target_x_btc = _ensure_btc_paths(target_x, "target_x")  # (B,T,C)
        if pred_x_btc.shape != target_x_btc.shape:
            raise ValueError(
                f"pred_x and target_x must have the same shape, got {pred_x_btc.shape} and {target_x_btc.shape}"
            )
        if int(pred_x_btc.shape[2]) != int(x_dim_i):
            raise ValueError(
                f"Expected x_dim={int(x_dim_i)} channels after conversion, got {int(pred_x_btc.shape[2])}."
            )

        if use_w:
            if pred_w is None or target_w is None:
                raise ValueError("use_w=True requires pred_w and target_w.")
            # W is always (B,T) or (B,T,1); squeeze to (B,T)
            pred_w_btc = _ensure_btc_paths(pred_w, "pred_w")
            target_w_btc = _ensure_btc_paths(target_w, "target_w")
            if int(pred_w_btc.shape[2]) != 1 or int(target_w_btc.shape[2]) != 1:
                raise ValueError(f"Expected pred_w/target_w to be scalar paths, got {pred_w_btc.shape} and {target_w_btc.shape}")
            pred_w_ = pred_w_btc[..., 0]
            target_w_ = target_w_btc[..., 0]
            if pred_w_.shape[:2] != pred_x_btc.shape[:2] or target_w_.shape[:2] != target_x_btc.shape[:2]:
                raise ValueError(
                    "pred_w/target_w must match pred_x/target_x shapes, got "
                    f"{pred_w_.shape}, {target_w_.shape} vs {pred_x_btc.shape}."
                )
        else:
            pred_w_ = None
            # In use_w=False mode, we optionally interpret `target_w` as a bracket density.
            target_w_ = None

        B, T, C = pred_x_btc.shape
        if int(T) < 2 or int(B) < 1:
            return jnp.asarray(0.0, dtype=jnp.float32)

        ts = jnp.linspace(0.0, 1.0, int(T), dtype=pred_x_btc.dtype)  # (T,)
        dt = ts[1:] - ts[:-1]  # (T-1,)

        cov_density_target_b = (
            _maybe_parse_cov_density(target_w, B=int(B), T=int(T)) if not use_w else None
        )
        has_density_target = cov_density_target_b is not None
        if cov_density_target_b is None:
            cov_density_target_b = jnp.zeros(
                (int(B), int(T), int(C), int(C)), dtype=pred_x_btc.dtype
            )

        d = int(d_static)

        def _augment_single(x_path: jax.Array, w_path: jax.Array | None) -> jax.Array:
            # x_path: (T,C)
            if anchor_at_start:
                x_path = x_path - x_path[:1]
                if w_path is not None:
                    w_path = w_path - w_path[:1]

            cols: list[jax.Array] = []
            if use_time:
                cols.append(ts[:, None])
            if use_w:
                assert w_path is not None
                cols.append(w_path[:, None])
            cols.append(x_path)
            y = jnp.concatenate(cols, axis=1)  # (T, d)

            if not prepend_zero_basepoint:
                return y
            zero0 = jnp.zeros((1, int(d)), dtype=y.dtype)
            return jnp.concatenate([zero0, y], axis=0)  # (T+1, d)

        def _cov_increments_from_x(
            x_path: jax.Array, cov_density: jax.Array, has_density: bool
        ) -> jax.Array:
            # x_path: (T,C), cov_density: (T,C,C) instantaneous bracket density d/dt <x>_t.
            # Returns per-step increments Δ<x>_k as (T-1,C,C).
            if not has_density:
                inc = jnp.diff(x_path, axis=0)  # (T-1,C)
                return jnp.einsum("tc,td->tcd", inc, inc)
            return cov_density[:-1] * dt[:, None, None]

        def _embed_cov_increments(dqv_x: jax.Array, *, dtype: jnp.dtype) -> jax.Array:
            # dqv_x: (T-1,C,C) -> (T-1,d,d) (with time/w cross terms zero)
            cov = jnp.zeros((int(dqv_x.shape[0]), int(d), int(d)), dtype=dtype)
            start = (1 if use_time else 0) + (1 if use_w else 0)
            cov = cov.at[:, start:, start:].set(dqv_x)
            if prepend_zero_basepoint:
                cov0 = jnp.zeros((1, int(d), int(d)), dtype=dtype)
                cov = jnp.concatenate([cov0, cov], axis=0)  # (T,d,d)
            return cov

        def _phi_single(
            x_path: jax.Array, w_path: jax.Array | None, cov_density: jax.Array, has_density: bool
        ) -> jax.Array:
            # Build cov increments using x-path only, then embed into (t,w,x) coordinates.
            dqv_x = _cov_increments_from_x(
                x_path, cov_density, has_density
            )
            cov_inc = _embed_cov_increments(dqv_x, dtype=x_path.dtype)
            path = _augment_single(x_path, w_path)
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
            cov_zeros = jnp.zeros((int(T), int(C), int(C)), dtype=pred_x_btc.dtype)
            phi_pred = jax.vmap(lambda x, w: _phi_single(x, w, cov_zeros, False))(pred_x_btc, pred_w_)
            phi_target = jax.vmap(lambda x, w: _phi_single(x, w, cov_zeros, False))(target_x_btc, target_w_)
        else:
            cov_zeros_b = jnp.zeros((int(B), int(T), int(C), int(C)), dtype=pred_x_btc.dtype)
            phi_pred = jax.vmap(lambda x, cov: _phi_single(x, None, cov, False))(pred_x_btc, cov_zeros_b)
            phi_target = jax.vmap(lambda x, cov: _phi_single(x, None, cov, has_density_target))(
                target_x_btc, cov_density_target_b
            )

        # For dot-product kernels, the MMD^2 reduces to the squared distance
        # between the mean feature embeddings. This avoids the O(B^2) Gram matrix.
        phi_pred_mean = jnp.mean(phi_pred, axis=0)
        phi_target_mean = jnp.mean(phi_target, axis=0)
        diff = phi_pred_mean - phi_target_mean
        return jnp.sum(diff * diff).astype(jnp.float32)

    return loss


def _maybe_unvech_spd(
    x: jax.Array,
) -> jax.Array:
    if x.ndim >= 2 and x.shape[-2:] == (3, 3):
        return x
    if x.shape[-1] == 6:
        return SPDManifold.unvech(x)
    return x


