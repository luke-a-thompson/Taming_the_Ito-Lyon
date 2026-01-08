"""
A reimplementation of Neural Rough Differential Equations (NRDE) using Jax and Stochastax (https://arxiv.org/abs/2009.08295).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
from collections.abc import Callable
import diffrax

from stochastax.hopf_algebras import ShuffleHopfAlgebra
from .logsignatures import (
    compute_windowed_logsignatures_from_values,
)
from .logsig_cde_solve import solve_cde_from_windowed_logsigs


class NRDEFunc(eqx.Module):
    """
    Vector field for a Neural RDE in log-ODE form.

    Given hidden state y in R^{cde_state_dim}, returns matrix in
    R^{cde_state_dim x logsig_size} which multiplies the log-signature vector
    on each interval.
    """

    vf_mlp: eqx.nn.MLP
    cde_state_dim: int
    shuffle_hopf_algebra: ShuffleHopfAlgebra = eqx.field(static=True)
    logsig_size: int

    def __init__(
        self,
        *,
        input_path_dim: int,
        cde_state_dim: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        shuffle_hopf_algebra: ShuffleHopfAlgebra,
        key: jax.Array,
    ) -> None:
        self.cde_state_dim = cde_state_dim
        self.shuffle_hopf_algebra = shuffle_hopf_algebra
        self.logsig_size = shuffle_hopf_algebra.basis_size()
        self.vf_mlp = eqx.nn.MLP(
            in_size=cde_state_dim,
            out_size=cde_state_dim
            * self.logsig_size,  # NRDE outputs one element per log-signature coefficient
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=jnn.softplus,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, t: jax.typing.ArrayLike, y: jax.Array, args: None) -> jax.Array:
        del t, args
        out = self.vf_mlp(y)
        return out.reshape(self.cde_state_dim, self.logsig_size)


class NeuralRDE(eqx.Module):
    """
    Neural Rough Differential Equation (log-ODE) model.

    Usage
    - Provide `ts` and either a `diffrax` control path or cubic interpolation coeffs.
    - The model computes per-interval log-signatures and applies discrete
      log-ODE updates with a readout on the hidden state.
    """

    # Modules
    initial: eqx.nn.MLP
    cde_func: NRDEFunc
    readout: eqx.nn.Linear

    # Static configuration
    shuffle_hopf_algebra: ShuffleHopfAlgebra = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)
    signature_window_size: int = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)

    # Solver configuration (matches NeuralCDE pattern)
    solver: diffrax.AbstractAdaptiveSolver = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        cde_state_dim: int,
        output_path_dim: int,
        vf_hidden_dim: int,
        initial_cond_mlp_depth: int,
        vf_mlp_depth: int,
        signature_depth: int,
        signature_window_size: int,
        key: jax.Array,
        readout_activation: Callable[[jax.Array], jax.Array] | None = None,
        solver: diffrax.AbstractAdaptiveSolver = diffrax.Bosh3(),
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            rtol=1e-2, atol=1e-3, dtmin=1e-6
        ),
        evolving_out: bool = True,
    ) -> None:
        k1, k2, k3 = jr.split(key, 3)
        self.shuffle_hopf_algebra = ShuffleHopfAlgebra.build(
            input_path_dim, signature_depth
        )
        # Initial state from initial control value (matches NCDE style)
        self.initial = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=cde_state_dim,
            width_size=vf_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=jnn.softplus,
            key=k1,
        )
        self.cde_func = NRDEFunc(
            input_path_dim=input_path_dim,
            cde_state_dim=cde_state_dim,
            vf_hidden_dim=vf_hidden_dim,
            vf_mlp_depth=vf_mlp_depth,
            shuffle_hopf_algebra=self.shuffle_hopf_algebra,
            key=k2,
        )
        self.readout = eqx.nn.Linear(
            in_features=cde_state_dim,
            out_features=output_path_dim,
            use_bias=True,
            key=k3,
        )
        self.readout_activation = (
            readout_activation if readout_activation is not None else (lambda x: x)
        )
        self.signature_depth = signature_depth
        self.signature_window_size = signature_window_size
        self.evolving_out = evolving_out

        self.solver = solver
        self.stepsize_controller = stepsize_controller

    def _forward_with_values(
        self,
        ts: jax.Array,
        control_values: jax.Array,
    ) -> jax.Array:
        """Core forward pass given sampled control values.

        This avoids constructing a control path and avoids evaluating it at `ts`.
        """
        x0 = control_values[0]
        h0 = self.initial(x0)

        logsigs = compute_windowed_logsignatures_from_values(
            control_values,
            self.shuffle_hopf_algebra,
            self.signature_depth,
            self.signature_window_size,
        )  # (num_windows, logsig_size)
        return solve_cde_from_windowed_logsigs(
            ts,
            logsigs,
            signature_window_size=int(self.signature_window_size),
            cde_func=self.cde_func,
            y0=h0,
            solver=self.solver,
            stepsize_controller=self.stepsize_controller,
        )

    def __call__(
        self,
        control_values: jax.Array,
    ) -> jax.Array:
        """
        Forward pass.

        Parameters
        - control_values: shape (T, C). Control values.
        - control_or_coeffs: either a diffrax control path (e.g. CubicInterpolation)
          or a tuple of cubic coefficients compatible with diffrax.CubicInterpolation.

        Returns
        - If self.evolving_out is False: shape (out_size,)
        - If self.evolving_out is True: shape (T, out_size)
        """
        length = control_values.shape[0]
        ts = jnp.linspace(0.0, 1.0, length, dtype=control_values.dtype)  # (T,)
        hidden_over_time = self._forward_with_values(ts, control_values)

        if self.evolving_out:

            def apply_readout(y: jax.Array) -> jax.Array:
                return self.readout_activation(self.readout(y))

            return jax.vmap(apply_readout)(hidden_over_time)
        else:
            return self.readout_activation(self.readout(hidden_over_time[-1]))
