from __future__ import annotations

from typing import Literal, overload

import diffrax
import jax
import jax.numpy as jnp

from stochastax.control_lifts import (
    compute_log_signature,
    compute_nonplanar_branched_signature,
    compute_planar_branched_signature,
)
from stochastax.control_lifts.signature_types import (
    LogSignature,
    BCKLogSignature,
    MKWLogSignature,
)
from stochastax.hopf_algebras import ShuffleHopfAlgebra, GLHopfAlgebra, MKWHopfAlgebra
from stochastax.types import HopfAlgebra, PrimitiveSignature


# Overloads for flatten=True (all return jax.Array)
@overload
def compute_windowed_logsignatures(
    ts: jax.Array,
    control: diffrax.AbstractPath,
    hopf_algebra: HopfAlgebra,
    signature_depth: int,
    signature_window_size: int,
    flatten: Literal[True],
) -> jax.Array: ...


# Overloads for flatten=False with specific Hopf algebra types
@overload
def compute_windowed_logsignatures(
    ts: jax.Array,
    control: diffrax.AbstractPath,
    hopf_algebra: ShuffleHopfAlgebra,
    signature_depth: int,
    signature_window_size: int,
    flatten: Literal[False],
) -> LogSignature: ...


@overload
def compute_windowed_logsignatures(
    ts: jax.Array,
    control: diffrax.AbstractPath,
    hopf_algebra: GLHopfAlgebra,
    signature_depth: int,
    signature_window_size: int,
    flatten: Literal[False],
) -> BCKLogSignature: ...


@overload
def compute_windowed_logsignatures(
    ts: jax.Array,
    control: diffrax.AbstractPath,
    hopf_algebra: MKWHopfAlgebra,
    signature_depth: int,
    signature_window_size: int,
    flatten: Literal[False],
) -> MKWLogSignature: ...


# Fallback overload for generic HopfAlgebra
@overload
def compute_windowed_logsignatures(
    ts: jax.Array,
    control: diffrax.AbstractPath,
    hopf_algebra: HopfAlgebra,
    signature_depth: int,
    signature_window_size: int,
    flatten: Literal[False],
) -> PrimitiveSignature: ...


def compute_windowed_logsignatures(
    ts: jax.Array,
    control: diffrax.AbstractPath,
    hopf_algebra: HopfAlgebra,
    signature_depth: int,
    signature_window_size: int,
    flatten: bool,
) -> jax.Array | PrimitiveSignature:
    """Compute per-interval windowed log-signatures along a control path.

    Parameters
    ----------
    ts:
        Shape (T,). Monotonically increasing timestamps.
    control:
        A `diffrax.AbstractPath` compatible object (e.g. `CubicInterpolation`).
    hopf_algebra:
        A Hopf algebra: ShuffleHopfAlgebra, GLHopfAlgebra, or MKWHopfAlgebra.
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

    def window_logsig(i: jax.Array) -> jax.Array | PrimitiveSignature:
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

        if flatten:
            return logsig.flatten()
        return logsig

    num_intervals = int(values.shape[0]) - 1
    indices = jnp.arange(num_intervals, dtype=jnp.int32)
    return jax.vmap(window_logsig)(indices)
