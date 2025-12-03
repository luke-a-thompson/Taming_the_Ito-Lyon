"""
A reimplementation of Neural Rough Differential Equations (NRDE) using Jax and Stochastax (https://arxiv.org/abs/2009.08295).
"""

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable
import diffrax

from stochastax.analytics import get_log_signature_dim

from .logsignatures import compute_windowed_logsignatures


class NRDEFunc(eqx.Module):
    """
    Vector field for a Neural RDE in log-ODE form.

    Given hidden state y in R^{cde_state_dim}, returns matrix in
    R^{cde_state_dim x logsig_size} which multiplies the log-signature vector
    on each interval.
    """

    mlp: eqx.nn.MLP
    logsig_size: int
    cde_state_dim: int

    def __init__(
        self,
        *,
        cde_state_dim: int,
        logsig_size: int,
        vf_hidden_dim: int,
        depth: int,
        key: jax.Array,
    ) -> None:
        self.logsig_size = logsig_size
        self.cde_state_dim = cde_state_dim
        self.mlp = eqx.nn.MLP(
            in_size=cde_state_dim,
            out_size=cde_state_dim * logsig_size,
            width_size=vf_hidden_dim,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, y: jax.Array) -> jax.Array:
        out = self.mlp(y)
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
    func: NRDEFunc
    readout: eqx.nn.Linear

    # Static configuration
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)
    signature_window_size: int = eqx.field(static=True)
    evolving_out: bool

    def __init__(
        self,
        input_path_dim: int,
        cde_state_dim: int,
        output_path_dim: int,
        vf_hidden_dim: int,
        depth: int,
        *,
        signature_depth: int,
        signature_window_size: int = 1,
        key: jax.Array,
        readout_activation: Callable[[jax.Array], jax.Array] | None = None,
        evolving_out: bool = True,
    ) -> None:
        k1, k2, k3 = jr.split(key, 3)
        # Initial state from initial control value (matches NCDE style)
        self.initial = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=cde_state_dim,
            width_size=vf_hidden_dim,
            depth=depth,
            activation=jnn.softplus,
            key=k1,
        )

        logsig_size = int(
            get_log_signature_dim(depth=signature_depth, dim=input_path_dim)
        )
        self.func = NRDEFunc(
            cde_state_dim=cde_state_dim,
            logsig_size=logsig_size,
            vf_hidden_dim=vf_hidden_dim,
            depth=depth,
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
        self.signature_depth = int(signature_depth)
        self.signature_window_size = int(signature_window_size)
        self.evolving_out = bool(evolving_out)

    def _rollout_hidden(self, h0: jax.Array, logsigs: jax.Array) -> jax.Array:
        """
        Discrete log-ODE rollout across intervals using a scan.

        logsigs: (T-1, logsig_size)
        Returns hidden states at each t: (T, cde_state_dim)
        """

        def step(carry: jax.Array, logsig_i: jax.Array) -> tuple[jax.Array, jax.Array]:
            h = carry
            mat = self.func(h)  # (cde_state_dim, logsig_size)
            h_next = h + mat @ logsig_i
            return h_next, h_next

        _, h_history = jax.lax.scan(step, h0, logsigs)
        return jnp.concatenate([h0[None, :], h_history], axis=0)

    def __call__(
        self,
        ts: jax.Array,
        coeffs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        """
        Forward pass.

        Parameters
        - ts: shape (T,). Monotonically increasing timestamps.
        - control_or_coeffs: either a diffrax control path (e.g. CubicInterpolation)
          or a tuple of cubic coefficients compatible with diffrax.CubicInterpolation.

        Returns
        - If self.evolving_out is False: shape (out_size,)
        - If self.evolving_out is True: shape (T, out_size)
        """
        control = diffrax.CubicInterpolation(ts, coeffs)

        x0 = control.evaluate(ts[0])
        h0 = self.initial(x0)

        logsigs = compute_windowed_logsignatures(
            ts,
            control,
            signature_depth=int(self.signature_depth),
            signature_window_size=int(self.signature_window_size),
            flatten=True,
        )  # (T-1 or T-stride, L)
        hidden_over_time = self._rollout_hidden(h0, logsigs)  # (T, H)

        if self.evolving_out:

            def apply_readout(y: jax.Array) -> jax.Array:
                return self.readout_activation(self.readout(y))

            return jax.vmap(apply_readout)(hidden_over_time)
        else:
            return self.readout_activation(self.readout(hidden_over_time[-1]))
