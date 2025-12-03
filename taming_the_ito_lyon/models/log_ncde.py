from __future__ import annotations

from collections.abc import Callable

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from stochastax.analytics import get_log_signature_dim
from stochastax.integrators import log_ode
from stochastax.vector_field_lifts import form_lyndon_lift
from stochastax.vector_field_lifts.split_vector_fields import split_multi_vector_field
from stochastax.hopf_algebras.hopf_algebra_types import ShuffleHopfAlgebra

from .logsignatures import compute_windowed_logsignatures


class LogNCDEFunc(eqx.Module):
    """Vector field for a log-NCDE built directly from Lyndon brackets.

    Given hidden state h in R^{cde_state_dim}, this builds a list of vector
    fields V_i: R^{cde_state_dim} -> R^{cde_state_dim} (one per input channel),
    lifts them to nonlinear Lyndon brackets, and then uses `log_ode` with the
    primitive log-signature on each interval.
    """

    base_mlp: eqx.nn.MLP
    signature_depth: int
    input_path_dim: int
    cde_state_dim: int
    shuffle_hopf_algebra: ShuffleHopfAlgebra = eqx.field(static=True)

    def __init__(
        self,
        *,
        cde_state_dim: int,
        input_path_dim: int,
        vf_hidden_dim: int,
        depth: int,
        signature_depth: int,
        key: jax.Array,
    ) -> None:
        self.cde_state_dim = cde_state_dim
        self.input_path_dim = input_path_dim
        self.signature_depth = signature_depth
        self.shuffle_hopf_algebra = ShuffleHopfAlgebra.build(
            input_path_dim, signature_depth, True
        )
        self.base_mlp = eqx.nn.MLP(
            in_size=cde_state_dim,
            out_size=input_path_dim * cde_state_dim,
            width_size=vf_hidden_dim,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, h: jax.Array, primitive_signature) -> jax.Array:
        # multi_vf(y): R^{cde_state_dim} -> R^{input_path_dim * cde_state_dim}
        vector_fields = split_multi_vector_field(
            self.base_mlp,
            self.input_path_dim,
            self.cde_state_dim,
        )
        brackets = form_lyndon_lift(vector_fields, h, self.shuffle_hopf_algebra)
        return log_ode(brackets, primitive_signature, h)


class LogNCDE(eqx.Module):
    """Discrete log-ODE version of NCDE that enforces Lyndon Lie polynomials."""

    initial: eqx.nn.MLP
    cde_func: LogNCDEFunc
    readout: eqx.nn.Linear
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)
    signature_window_size: int = eqx.field(static=True)
    evolving_out: bool
    logsig_size: int

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
        self.initial = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=cde_state_dim,
            width_size=vf_hidden_dim,
            depth=depth,
            activation=jnn.softplus,
            key=k1,
        )
        self.cde_func = LogNCDEFunc(
            cde_state_dim=cde_state_dim,
            input_path_dim=input_path_dim,
            vf_hidden_dim=vf_hidden_dim,
            depth=depth,
            signature_depth=int(signature_depth),
            key=k2,
        )
        self.readout = eqx.nn.Linear(
            in_features=cde_state_dim,
            out_features=output_path_dim,
            use_bias=True,
            key=k3,
        )
        self.readout_activation = readout_activation or (lambda x: x)
        self.signature_depth = int(signature_depth)
        self.signature_window_size = int(signature_window_size)
        self.evolving_out = bool(evolving_out)
        self.logsig_size = int(
            get_log_signature_dim(depth=self.signature_depth, dim=input_path_dim)
        )

    def _rollout_hidden(self, h0: jax.Array, logsigs: object) -> jax.Array:
        """Roll out hidden state over time using a scanned loop."""

        def step(
            h: jax.Array,
            primitive_signature: object,
        ) -> tuple[jax.Array, jax.Array]:
            new_h = self.cde_func(h, primitive_signature)
            return new_h, new_h

        _, h_history = jax.lax.scan(step, h0, logsigs)
        history = jnp.concatenate([h0[None, :], h_history], axis=0)
        return history

    def __call__(
        self,
        ts: jax.Array,
        coeffs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        control = diffrax.CubicInterpolation(ts, coeffs)
        x0 = control.evaluate(ts[0])
        h0 = self.initial(x0)
        logsigs = compute_windowed_logsignatures(
            ts,
            control,
            signature_depth=int(self.signature_depth),
            signature_window_size=int(self.signature_window_size),
            flatten=False,
        )
        hidden_over_time = self._rollout_hidden(h0, logsigs)

        if self.evolving_out:

            def apply_readout(y: jax.Array) -> jax.Array:
                return self.readout_activation(self.readout(y))

            return jax.vmap(apply_readout)(hidden_over_time)

        return self.readout_activation(self.readout(hidden_over_time[-1]))
