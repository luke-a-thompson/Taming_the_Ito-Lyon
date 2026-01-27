from __future__ import annotations

import jax
import jax.numpy as jnp

from stochastax.control_lifts import (
    compute_log_signature,
    compute_nonplanar_branched_signature,
    compute_planar_branched_signature,
)
from stochastax.hopf_algebras import ShuffleHopfAlgebra, GLHopfAlgebra, MKWHopfAlgebra
from stochastax.types import HopfAlgebra


def compute_disjoint_signature_times(
    ts: jax.Array, signature_window_size: int
) -> jax.Array:
    """Return the subset of times corresponding to disjoint signature windows.

    If `signature_window_size = w`, then signatures are computed on windows starting at
    indices `0, w, 2w, ...` (in *points*), with the final time `ts[-1]` always included.

    Returns
    -------
    A 1D array `ts_sig` of shape `(num_windows + 1,)`, suitable for use as the
    timestamps for a `diffrax.LinearInterpolation` over cumulative signature values.
    """
    step = int(signature_window_size)
    if step <= 0:
        raise ValueError("signature_window_size must be a positive integer.")

    num_points = int(ts.shape[0])
    if num_points < 2:
        raise ValueError("ts must contain at least two time points.")

    # Take window starts: 0, step, 2*step, ... strictly before the last point index.
    #
    # Note: `jnp.arange(0, num_points - 1, step)` never includes `num_points - 1`
    # because the `stop` is exclusive, so it is always safe to append `ts[-1]`
    # without any data-dependent branching (important for JIT compatibility).
    start_indices = jnp.arange(0, num_points - 1, step, dtype=jnp.int32)
    return jnp.concatenate([ts[start_indices], ts[-1:]], axis=0)


def compute_windowed_logsignatures_from_values(
    values: jax.Array,
    hopf_algebra: HopfAlgebra,
    signature_depth: int,
    signature_window_size: int,
    *,
    brownian_channels: list[int] | None = None,
    brownian_corr: float | None = None,
) -> jax.Array:
    """Compute disjoint windowed log-signatures (flattened) from sampled control values.

    This is the main entry point used by the logsignature-driven models. It avoids
    constructing a Diffrax control and avoids `vmap(control.evaluate)(ts)` when the
    control values are already available (e.g. training data sampled on `ts`).

    Parameters
    ----------
    values:
        Shape (T, C). Control values sampled at the solver timestamps `ts`.
    hopf_algebra:
        A Hopf algebra: ShuffleHopfAlgebra, GLHopfAlgebra, or MKWHopfAlgebra.
    signature_depth:
        Truncation depth for the log-signature.
    signature_window_size:
        Number of *intervals* per (disjoint) window. Each window is represented using
        `signature_window_size + 1` points, and windows start at indices
        `0, signature_window_size, 2*signature_window_size, ...`.
    brownian_channels:
        Indices of Brownian channels in the full control (time is index 0). If None,
        all non-time channels are treated as Brownian for Itô covariation.
    brownian_corr:
        Optional correlation to apply between the first two Brownian channels.

    Returns
    -------
    A JAX array of shape (num_windows, logsig_size).
    """
    step = int(signature_window_size)
    if step <= 0:
        raise ValueError("signature_window_size must be a positive integer.")

    window_len = step + 1

    # For unconditional Brownian drivers we want an Itô branched lift when using
    # branched Hopf algebras (GL/MKW). In that case we must provide the quadratic
    # covariation increments. In this codebase, unconditional controls include a
    # leading time channel, followed by Brownian channels.
    #
    # We assume the sampling grid is uniform (ts = linspace(0,1,T)), which is true
    # throughout the training runtime.
    dt = jnp.asarray(1.0 / float(int(values.shape[0]) - 1), dtype=values.dtype)

    def _brownian_cov_increments(num_increments: int, dim: int) -> jax.Array:
        """Covariance for increments of (t, W) with optional correlation.

        `brownian_channels` indexes channels in the full control (time at index 0).
        If None, all non-time channels are treated as Brownian.
        """
        if dim < 2:
            raise ValueError(
                "Branched (Itô) signature requires at least 2 channels: time + Brownian."
            )
        if brownian_channels is None:
            brownian_idxs = list(range(1, dim))
        else:
            brownian_idxs = [int(i) for i in brownian_channels]
            if any(i <= 0 or i >= dim for i in brownian_idxs):
                raise ValueError(
                    f"brownian_channels must be in [1, {dim - 1}], got {brownian_idxs}"
                )
        cov1 = jnp.zeros((dim, dim), dtype=values.dtype)
        for idx in brownian_idxs:
            cov1 = cov1.at[idx, idx].set(dt)
        if brownian_corr is not None and len(brownian_idxs) >= 2:
            rho = jnp.asarray(brownian_corr, dtype=values.dtype)
            i0 = brownian_idxs[0]
            i1 = brownian_idxs[1]
            cov1 = cov1.at[i0, i1].set(rho * dt)
            cov1 = cov1.at[i1, i0].set(rho * dt)
        return jnp.broadcast_to(cov1, (num_increments, dim, dim))

    def window_logsig(i: jax.Array) -> jax.Array:
        seg = jax.lax.dynamic_slice_in_dim(values, i, window_len, axis=0)

        match hopf_algebra:
            case ShuffleHopfAlgebra():
                logsig = compute_log_signature(
                    seg,
                    signature_depth,
                    hopf_algebra,
                    "Lyndon words",
                    "full",
                )
            case GLHopfAlgebra():
                cov_increments = _brownian_cov_increments(
                    num_increments=window_len - 1, dim=int(seg.shape[-1])
                )
                sig = compute_nonplanar_branched_signature(
                    seg,
                    signature_depth,
                    hopf_algebra,
                    "full",
                    cov_increments=cov_increments,
                )
                logsig = sig.log()
            case MKWHopfAlgebra():
                cov_increments = _brownian_cov_increments(
                    num_increments=window_len - 1, dim=int(seg.shape[-1])
                )
                sig = compute_planar_branched_signature(
                    seg,
                    signature_depth,
                    hopf_algebra,
                    "full",
                    cov_increments=cov_increments,
                )
                logsig = sig.log()
            case _:
                raise ValueError(f"Unsupported Hopf algebra: {type(hopf_algebra)}")

        return logsig.flatten()

    num_points = int(values.shape[0])
    remainder = (num_points - 1) % step
    if remainder != 0:
        raise ValueError(
            "Disjoint signature windows must cover the full series with no padding. "
            f"Got num_points={num_points}, signature_window_size={step} "
            f"(remainder={remainder})."
        )

    indices = jnp.arange(0, num_points - 1, step, dtype=jnp.int32)
    return jax.vmap(window_logsig)(indices)
