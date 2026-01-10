from __future__ import annotations

from collections.abc import Callable

import diffrax
import jax
import jax.numpy as jnp

from .logsignatures import compute_disjoint_signature_times


def solve_cde_from_windowed_logsigs(
    ts: jax.Array,
    windowed_logsigs: jax.Array,
    *,
    signature_window_size: int,
    cde_func: Callable[[jax.typing.ArrayLike, jax.Array, None], jax.Array],
    y0: jax.Array,
    solver: diffrax.AbstractAdaptiveSolver,
    stepsize_controller: diffrax.AbstractStepSizeController,
) -> jax.Array:
    """Solve a CDE/ODE driven by the cumulative sum of disjoint window log-signatures.

    Parameters
    ----------
    ts:
        Solver timestamps, shape (T,).
    windowed_logsigs:
        Disjoint window log-signatures, shape (num_windows, logsig_size).
    signature_window_size:
        Window size in number of intervals (step), consistent with how `windowed_logsigs`
        was computed.
    cde_func:
        Vector field for `diffrax.ControlTerm`.
    y0:
        Initial state at `ts[0]`.
    solver:
        Diffrax ODE solver.
    stepsize_controller:
        Diffrax stepsize controller.
    """
    ts_sig = compute_disjoint_signature_times(ts, int(signature_window_size))
    logsig_size = int(windowed_logsigs.shape[-1])
    z0 = jnp.zeros((1, logsig_size), dtype=windowed_logsigs.dtype)
    z = jnp.concatenate([z0, jnp.cumsum(windowed_logsigs, axis=0)], axis=0)
    logsig_control = diffrax.LinearInterpolation(ts=ts_sig, ys=z)

    term = diffrax.ControlTerm(cde_func, logsig_control).to_ode()
    saveat = diffrax.SaveAt(ts=ts)

    solution = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=0.1 if isinstance(stepsize_controller, diffrax.ConstantStepSize) else None,
        y0=y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )

    assert solution.ys is not None
    return solution.ys
