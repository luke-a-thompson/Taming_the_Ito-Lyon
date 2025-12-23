"""
A reimplementation of Log-NCDE using Jax and Stochastax (https://arxiv.org/abs/2402.18512).
"""

from collections.abc import Callable

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from stochastax.integrators import log_ode
from stochastax.vector_field_lifts import form_lyndon_lift
from stochastax.vector_field_lifts.split_vector_fields import split_multi_vector_field
from stochastax.hopf_algebras import ShuffleHopfAlgebra
from stochastax.control_lifts import LogSignature
from .logsignatures import compute_windowed_logsignatures


class LogNCDEFunc(eqx.Module):
    """Vector field for a log-NCDE built directly from Lyndon brackets.

    Given hidden state h in R^{cde_state_dim}, this builds a list of vector
    fields V_i: R^{cde_state_dim} -> R^{cde_state_dim} (one per input channel),
    lifts them to nonlinear Lyndon brackets, and then uses `log_ode` with the
    primitive log-signature on each interval.
    """

    input_path_dim: int
    base_mlp: eqx.nn.MLP
    shuffle_hopf_algebra: ShuffleHopfAlgebra = eqx.field(static=True)
    cde_state_dim: int

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
        self.input_path_dim = input_path_dim
        self.shuffle_hopf_algebra = shuffle_hopf_algebra
        self.base_mlp = eqx.nn.MLP(
            in_size=cde_state_dim,
            out_size=input_path_dim * cde_state_dim,
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(
        self,
        h: jax.Array,
        log_signature: LogSignature,
    ) -> jax.Array:
        # multi_vf(y): R^{cde_state_dim} -> R^{input_path_dim * cde_state_dim}
        vector_fields = split_multi_vector_field(
            self.base_mlp,
            self.input_path_dim,
            self.cde_state_dim,
        )
        brackets = form_lyndon_lift(vector_fields, h, self.shuffle_hopf_algebra)
        return log_ode(brackets, log_signature, h)


class LogNCDE(eqx.Module):
    """Discrete log-ODE version of NCDE that enforces Lyndon Lie polynomials."""

    initial: eqx.nn.MLP
    shuffle_hopf_algebra: ShuffleHopfAlgebra = eqx.field(static=True)
    cde_func: LogNCDEFunc
    readout: eqx.nn.Linear
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
        initial_cond_mlp_depth: int,
        vf_mlp_depth: int,
        *,
        signature_depth: int,
        signature_window_size: int,
        key: jax.Array,
        readout_activation: Callable[[jax.Array], jax.Array] | None = None,
        evolving_out: bool = True,
    ) -> None:
        k1, k2, k3 = jr.split(key, 3)
        self.shuffle_hopf_algebra = ShuffleHopfAlgebra.build(
            input_path_dim, signature_depth
        )
        self.initial = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=cde_state_dim,
            width_size=vf_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=jnn.softplus,
            key=k1,
        )
        self.cde_func = LogNCDEFunc(
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

    def __call__(
        self,
        ts: jax.Array,
        coeffs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        control = diffrax.CubicInterpolation(ts, coeffs)
        x0 = control.evaluate(ts[0])
        h0 = self.initial(x0)
        log_signatures = compute_windowed_logsignatures(
            ts,
            control,
            self.shuffle_hopf_algebra,
            int(self.signature_depth),
            int(self.signature_window_size),
            False,
        )

        def step(
            h: jax.Array,
            log_signature: LogSignature,
        ) -> tuple[jax.Array, jax.Array]:
            new_h = self.cde_func(h, log_signature)
            return new_h, new_h

        _, h_history = jax.lax.scan(step, h0, log_signatures)
        hidden_over_time = jnp.concatenate([h0[None, :], h_history], axis=0)

        if self.evolving_out:

            def apply_readout(y: jax.Array) -> jax.Array:
                return self.readout_activation(self.readout(y))

            return jax.vmap(apply_readout)(hidden_over_time)

        return self.readout_activation(self.readout(hidden_over_time[-1]))
