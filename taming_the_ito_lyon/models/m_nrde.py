"""
A reimplementation of Log-NCDE using Jax and Stochastax (https://arxiv.org/abs/2402.18512).
"""

from collections.abc import Callable
from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from stochastax.integrators import log_ode
from stochastax.vector_field_lifts.split_vector_fields import split_multi_vector_field
from stochastax.types import HopfAlgebra, PrimitiveSignature, VectorFieldLift

from stochastax.hopf_algebras import ShuffleHopfAlgebra, GLHopfAlgebra, MKWHopfAlgebra
from stochastax.vector_field_lifts import form_lyndon_lift, form_bck_lift, form_mkw_lift

from .logsignatures import compute_windowed_logsignatures


class MNRDEFunc(eqx.Module):
    """Vector field for a MNDRE built directly from Lyndon brackets.

    Given hidden state h in R^{cde_state_dim}, this builds a list of vector
    fields V_i: R^{cde_state_dim} -> R^{cde_state_dim} (one per input channel),
    lifts them to nonlinear multi-nomial brackets, and then uses `m_ode` with the
    primitive multi-nomial signature on each interval.
    """

    input_path_dim: int
    base_mlp: eqx.nn.MLP
    signature_depth: int
    cde_state_dim: int
    hopf_algebra: HopfAlgebra = eqx.field(static=True)
    vf_lift: VectorFieldLift = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_path_dim: int,
        cde_state_dim: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        signature_depth: int,
        hopf_algebra: HopfAlgebra,
        tangent_projector: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        key: jax.Array,
    ) -> None:
        self.cde_state_dim = cde_state_dim
        self.input_path_dim = input_path_dim
        self.signature_depth = signature_depth
        self.hopf_algebra = hopf_algebra

        # Default projector for Euclidean case: identity
        projector = (
            tangent_projector if tangent_projector is not None else (lambda _y, v: v)
        )

        # Select and configure the appropriate vector field lift
        match hopf_algebra:
            case ShuffleHopfAlgebra():
                self.vf_lift = partial(form_lyndon_lift, project_to_tangent=projector)
            case GLHopfAlgebra():
                self.vf_lift = form_bck_lift
            case MKWHopfAlgebra():
                self.vf_lift = partial(form_mkw_lift, project_to_tangent=projector)
            case _:
                raise ValueError(f"Unsupported Hopf algebra: {type(hopf_algebra)}")

        self.base_mlp = eqx.nn.MLP(
            in_size=cde_state_dim,
            out_size=input_path_dim * cde_state_dim,  # Following log-NCDE, MNRDE outputs the first level
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(
        self, h: jax.Array, primitive_signature: PrimitiveSignature
    ) -> jax.Array:
        vector_fields = split_multi_vector_field(
            self.base_mlp,
            self.input_path_dim,
            self.cde_state_dim,
        )
        brackets = self.vf_lift(vector_fields, h, self.hopf_algebra)
        return log_ode(brackets, primitive_signature, h)  # type: ignore[call-overload]


class MNDRE(eqx.Module):
    """Discrete log-ODE version of NCDE that enforces Lyndon Lie polynomials."""

    initial: eqx.nn.MLP
    cde_func: MNRDEFunc
    readout: eqx.nn.Linear

    # Static configuration
    hopf_algebra: HopfAlgebra = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)
    signature_window_size: int = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)

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
        hopf_algebra: HopfAlgebra,
        key: jax.Array,
        tangent_projector: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        readout_activation: Callable[[jax.Array], jax.Array] | None = None,
        evolving_out: bool = True,
    ) -> None:
        k1, k2, k3 = jr.split(key, 3)
        self.hopf_algebra = hopf_algebra.build(input_path_dim, signature_depth)
        self.initial = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=cde_state_dim,
            width_size=vf_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=jnn.softplus,
            key=k1,
        )
        self.cde_func = MNRDEFunc(
            input_path_dim=input_path_dim,
            cde_state_dim=cde_state_dim,
            vf_hidden_dim=vf_hidden_dim,
            vf_mlp_depth=vf_mlp_depth,
            signature_depth=signature_depth,
            hopf_algebra=self.hopf_algebra,
            tangent_projector=tangent_projector,
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
        logsigs = compute_windowed_logsignatures(
            ts,
            control,
            self.hopf_algebra,
            signature_depth=self.signature_depth,
            signature_window_size=self.signature_window_size,
            flatten=False,
        )

        def step(
            h: jax.Array,
            primitive_signature: PrimitiveSignature,
        ) -> tuple[jax.Array, jax.Array]:
            new_h = self.cde_func(h, primitive_signature)
            return new_h, new_h

        _, h_history = jax.lax.scan(step, h0, logsigs)
        hidden_over_time = jnp.concatenate([h0[None, :], h_history], axis=0)

        if self.evolving_out:

            def apply_readout(y: jax.Array) -> jax.Array:
                return self.readout_activation(self.readout(y))

            return jax.vmap(apply_readout)(hidden_over_time)

        return self.readout_activation(self.readout(hidden_over_time[-1]))
