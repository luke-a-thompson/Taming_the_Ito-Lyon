"""
Neural Rough Differential Equation (log-ODE/NRDE) model in JAX.

This mirrors the style and call signature of `ncde.py` but performs a
discrete log-ODE update using per-interval log-signatures computed via
`quicksig`.
"""

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable
import diffrax

from quicksig.signatures import compute_log_signature
from quicksig.signatures.get_signature_dim import get_log_signature_dim


class NRDEFunc(eqx.Module):
    """
    Vector field for a Neural RDE in log-ODE form.

    Given hidden state y in R^{hidden_size}, returns matrix in
    R^{hidden_size x logsig_size} which multiplies the log-signature vector
    on each interval.
    """

    mlp: eqx.nn.MLP
    logsig_size: int
    hidden_size: int

    def __init__(
        self,
        *,
        hidden_size: int,
        logsig_size: int,
        width_size: int,
        depth: int,
        key: jax.Array,
    ) -> None:
        self.logsig_size = logsig_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * logsig_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, y: jax.Array) -> jax.Array:
        out = self.mlp(y)
        return out.reshape(self.hidden_size, self.logsig_size)


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
        hidden_size: int,
        output_path_dim: int,
        width_size: int,
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
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=k1,
        )

        logsig_size = int(
            get_log_signature_dim(
                depth=signature_depth, dim=input_path_dim, flatten=True
            )
        )
        self.func = NRDEFunc(
            hidden_size=hidden_size,
            logsig_size=logsig_size,
            width_size=width_size,
            depth=depth,
            key=k2,
        )
        self.readout = eqx.nn.Linear(
            in_features=hidden_size,
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

    def _compute_logsignatures(
        self, ts: jax.Array, control: diffrax.AbstractPath
    ) -> jax.Array:
        """
        Compute per-interval log-signatures for the (piecewise cubic) control.

        Returns an array of shape (T-1, logsig_size).
        """
        # Sample the control at knot times
        eval_fn = jax.vmap(control.evaluate)
        values = eval_fn(ts)  # (T, C)
        window_len = int(self.signature_window_size) + 1

        # Pad the end so we can take fixed-length windows at all starts 0..T-2
        pad_len = window_len - 1
        pad_tail = (
            jnp.repeat(values[-1][None, :], repeats=pad_len, axis=0)
            if pad_len > 0
            else values[:0]
        )
        values_padded = jnp.concatenate([values, pad_tail], axis=0)  # (T+pad_len, C)

        def window_logsig(i: jax.Array) -> jax.Array:
            # Take a fixed-size dynamic slice along the time axis
            seg = jax.lax.dynamic_slice_in_dim(values_padded, i, window_len, axis=0)
            return compute_log_signature(
                seg,
                depth=int(self.signature_depth),
                log_signature_type="Lyndon words",
                mode="full",
            ).flatten()

        # Produce exactly T-1 windows to match the number of updates
        indices = jnp.arange(values.shape[0] - 1)
        return jax.vmap(window_logsig)(indices)

    def _rollout_hidden(self, h0: jax.Array, logsigs: jax.Array, T: int) -> jax.Array:
        """
        Discrete log-ODE rollout across intervals using a scan.

        logsigs: (T-1, logsig_size)
        Returns hidden states at each t: (T, hidden_size)
        """

        def step(carry: jax.Array, logsig_i: jax.Array) -> tuple[jax.Array, jax.Array]:
            h = carry
            mat = self.func(h)  # (hidden_size, logsig_size)
            h_next = h + mat @ logsig_i
            return h_next, h_next

        final_h, h_hist = jax.lax.scan(step, h0, logsigs)
        return jnp.concatenate([h0[None, :], h_hist], axis=0)

    def __call__(
        self,
        ts: jax.Array,
        control_or_coeffs: diffrax.AbstractPath | tuple[jax.Array, ...],
        *,
        evolving_out: bool = True,
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
        if isinstance(control_or_coeffs, diffrax.AbstractPath):
            control = control_or_coeffs
        else:
            control = diffrax.CubicInterpolation(ts, control_or_coeffs)

        x0 = control.evaluate(ts[0])
        h0 = self.initial(x0)

        logsigs = self._compute_logsignatures(ts, control)  # (T-1 or T-stride, L)
        hidden_over_time = self._rollout_hidden(h0, logsigs, T=ts.shape[0])  # (T, H)

        use_evolving = bool(evolving_out)
        if use_evolving:

            def apply_readout(y: jax.Array) -> jax.Array:
                return self.readout_activation(self.readout(y))

            return jax.vmap(apply_readout)(hidden_over_time)
        else:
            return self.readout_activation(self.readout(hidden_over_time[-1]))
