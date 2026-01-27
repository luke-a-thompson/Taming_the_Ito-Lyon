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
    dt0: float | None = None,
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

    # This will evaluate the vector field many times over the course of the solve.
    if dt0 is None and isinstance(stepsize_controller, diffrax.ConstantStepSize):
        dt0 = 0.01

    solution = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0,
        y0=y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )

    assert solution.ys is not None
    return solution.ys


def solve_cde_from_windowed_logsigs_piecewise(
    ts: jax.Array,
    windowed_logsigs: jax.Array,
    *,
    signature_window_size: int,
    cde_func: Callable[[jax.typing.ArrayLike, jax.Array, None], jax.Array],
    y0: jax.Array,
    solver: diffrax.AbstractAdaptiveSolver,
    stepsize_controller: diffrax.AbstractStepSizeController,
    dt0: float | None = None,
) -> jax.Array:
    """Solve a CDE/ODE by integrating each window with constant dz/dt (no interpolation)."""
    step = int(signature_window_size)
    ts_sig = compute_disjoint_signature_times(ts, step)
    num_windows = int(windowed_logsigs.shape[0])

    if num_windows != int(ts_sig.shape[0]) - 1:
        raise ValueError(
            "windowed_logsigs and signature times are inconsistent in length."
        )

    state_dim = int(y0.shape[0])
    outputs = jnp.zeros((num_windows, step + 1, state_dim), dtype=y0.dtype)

    def body(
        i: jax.Array,
        carry: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        y, out = carry
        start_index = i * step
        ts_window = jax.lax.dynamic_slice(ts, (start_index,), (step + 1,))
        t0 = ts_window[0]
        t1 = ts_window[-1]
        dt = t1 - t0
        dzdt = windowed_logsigs[i] / dt

        def ode_func(t: jax.typing.ArrayLike, y: jax.Array, args: None) -> jax.Array:
            del args
            return cde_func(t, y, None) @ dzdt

        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(ode_func),
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=0.01
            if isinstance(stepsize_controller, diffrax.ConstantStepSize)
            else None,
            y0=y,
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(ts=ts_window),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
        )
        assert solution.ys is not None
        out = out.at[i].set(solution.ys)
        return solution.ys[-1], out

    _, outputs = jax.lax.fori_loop(0, num_windows, body, (y0, outputs))

    first = outputs[0]
    if num_windows == 1:
        return first
    rest = outputs[1:, 1:, :].reshape(((num_windows - 1) * step, state_dim))
    return jnp.concatenate([first, rest], axis=0)
