from __future__ import annotations

from typing import Literal, overload

import diffrax
import jax
import jax.numpy as jnp

from stochastax.control_lifts import compute_log_signature
from stochastax.control_lifts.signature_types import LogSignature


@overload
def compute_windowed_logsignatures(
    ts: jax.Array,
    control: diffrax.AbstractPath,
    *,
    signature_depth: int,
    signature_window_size: int,
    flatten: Literal[True],
) -> jax.Array: ...


@overload
def compute_windowed_logsignatures(
    ts: jax.Array,
    control: diffrax.AbstractPath,
    *,
    signature_depth: int,
    signature_window_size: int,
    flatten: Literal[False],
) -> LogSignature: ...


def compute_windowed_logsignatures(
    ts: jax.Array,
    control: diffrax.AbstractPath,
    *,
    signature_depth: int,
    signature_window_size: int,
    flatten: bool,
) -> jax.Array | object:
    """Compute per-interval windowed log-signatures along a control path.

    Parameters
    ----------
    ts:
        Shape (T,). Monotonically increasing timestamps.
    control:
        A `diffrax.AbstractPath` compatible object (e.g. `CubicInterpolation`).
    signature_depth:
        Truncation depth for the log-signature.
    signature_window_size:
        Number of *additional* points to include in each window, so that
        each window has length `signature_window_size + 1`.
    flatten:
        If True, each per-window log-signature is flattened to a 1D array,
        returning an array of shape (T-1, logsig_size). If False, returns
        a pytree whose leaves have leading dimension T-1.
    """
    eval_fn = jax.vmap(control.evaluate)
    values = eval_fn(ts)  # (T, C)
    window_len = int(signature_window_size) + 1

    pad_len = window_len - 1
    if pad_len > 0:
        pad_tail = jnp.repeat(values[-1][None, :], repeats=pad_len, axis=0)
    else:
        pad_tail = values[:0]
    values_padded = jnp.concatenate([values, pad_tail], axis=0)

    def window_logsig(i: jax.Array) -> jax.Array | object:
        seg = jax.lax.dynamic_slice_in_dim(values_padded, i, window_len, axis=0)
        logsig = compute_log_signature(
            seg,
            depth=signature_depth,
            log_signature_type="Lyndon words",
            mode="full",
        )
        if flatten:
            return logsig.flatten()
        return logsig

    num_intervals = int(values.shape[0]) - 1
    indices = jnp.arange(num_intervals, dtype=jnp.int32)
    return jax.vmap(window_logsig)(indices)
