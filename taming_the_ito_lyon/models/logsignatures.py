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

    Returns
    -------
    A JAX array of shape (num_windows, logsig_size).
    """
    step = int(signature_window_size)
    if step <= 0:
        raise ValueError("signature_window_size must be a positive integer.")

    window_len = step + 1

    pad_len = window_len - 1
    if pad_len > 0:
        pad_tail = jnp.repeat(values[-1][None, :], repeats=pad_len, axis=0)
    else:
        pad_tail = values[:0]
    values_padded = jnp.concatenate([values, pad_tail], axis=0)

    def window_logsig(i: jax.Array) -> jax.Array:
        seg = jax.lax.dynamic_slice_in_dim(values_padded, i, window_len, axis=0)

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
                sig = compute_nonplanar_branched_signature(
                    seg,
                    signature_depth,
                    hopf_algebra,
                    "full",
                )
                logsig = sig.log()
            case MKWHopfAlgebra():
                sig = compute_planar_branched_signature(
                    seg,
                    signature_depth,
                    hopf_algebra,
                    "full",
                )
                logsig = sig.log()
            case _:
                raise ValueError(f"Unsupported Hopf algebra: {type(hopf_algebra)}")

        return logsig.flatten()

    num_points = int(values.shape[0])
    num_windows = (num_points - 1 + step - 1) // step
    indices = (jnp.arange(num_windows, dtype=jnp.int32) * step).astype(jnp.int32)
    return jax.vmap(window_logsig)(indices)
