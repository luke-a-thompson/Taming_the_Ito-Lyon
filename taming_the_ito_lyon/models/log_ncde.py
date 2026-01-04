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

from stochastax.vector_field_lifts.lie_lift import form_lyndon_bracket_functions
from stochastax.vector_field_lifts.split_vector_fields import split_multi_vector_field
from stochastax.hopf_algebras import ShuffleHopfAlgebra
from .extrapolation import ExtrapolationScheme
from .logsignatures import (
    compute_windowed_logsignatures_from_values,
)
from .logsig_cde_solve import solve_cde_from_windowed_logsigs


class LogNCDEFunc(eqx.Module):
    """Vector field for a log-NCDE, expressed as a CDE over (flattened) log-signatures.

    This returns a matrix of shape (cde_state_dim, logsig_size) which multiplies the
    time-derivative of a cumulative log-signature control path.
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

    def __call__(self, t: jax.typing.ArrayLike, y: jax.Array, args: None) -> jax.Array:
        del t, args

        vector_fields = split_multi_vector_field(
            self.base_mlp,
            self.input_path_dim,
            self.cde_state_dim,
        )
        bracket_functions = form_lyndon_bracket_functions(
            vector_fields, self.shuffle_hopf_algebra
        )

        flat_bracket_functions = [
            bf
            for level in bracket_functions
            for bf in level  # type: ignore[union-attr]
        ]
        cols = [bf(y) for bf in flat_bracket_functions]
        return jnp.stack(cols, axis=1)


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

    # Extrapolation scheme
    extrapolation_scheme: ExtrapolationScheme | None = eqx.field(static=True)
    n_recon: int | None = eqx.field(static=True)

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
        *,
        signature_depth: int,
        signature_window_size: int,
        key: jax.Array,
        readout_activation: Callable[[jax.Array], jax.Array] | None = None,
        solver: diffrax.AbstractAdaptiveSolver = diffrax.Bosh3(),
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            rtol=1e-2, atol=1e-3, dtmin=1e-6
        ),
        evolving_out: bool = True,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
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
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon
        self.solver = solver
        self.stepsize_controller = stepsize_controller

    def _forward_from_logsigs(
        self, ts: jax.Array, x0: jax.Array, log_signatures: jax.Array
    ) -> jax.Array:
        """Solve the induced CDE given initial input `x0` and disjoint window log-signatures."""
        h0 = self.initial(x0)
        return solve_cde_from_windowed_logsigs(
            ts,
            log_signatures,
            signature_window_size=int(self.signature_window_size),
            cde_func=self.cde_func,
            y0=h0,
            solver=self.solver,
            stepsize_controller=self.stepsize_controller,
        )

    def _forward_with_control(
        self, ts: jax.Array, control: diffrax.AbstractPath
    ) -> jax.Array:
        x0 = control.evaluate(ts[0])
        control_values = jax.vmap(control.evaluate)(ts)
        log_signatures = compute_windowed_logsignatures_from_values(
            control_values,
            self.shuffle_hopf_algebra,
            int(self.signature_depth),
            int(self.signature_window_size),
        )
        return self._forward_from_logsigs(ts, x0, log_signatures)

    def _forward_with_values(
        self, ts: jax.Array, control_values: jax.Array
    ) -> jax.Array:
        x0 = control_values[0]
        log_signatures = compute_windowed_logsignatures_from_values(
            control_values,
            self.shuffle_hopf_algebra,
            int(self.signature_depth),
            int(self.signature_window_size),
        )
        return self._forward_from_logsigs(ts, x0, log_signatures)

    def __call__(
        self,
        ts: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        if self.extrapolation_scheme is not None:
            assert self.n_recon is not None, (
                "n_recon must be set when using extrapolation_scheme"
            )
            control, _ = self.extrapolation_scheme.create_control(ts, x, self.n_recon)
            hidden_over_time = self._forward_with_control(ts, control)
        else:
            hidden_over_time = self._forward_with_values(ts, x)

        if self.evolving_out:

            def apply_readout(y: jax.Array) -> jax.Array:
                return self.readout_activation(self.readout(y))

            return jax.vmap(apply_readout)(hidden_over_time)

        return self.readout_activation(self.readout(hidden_over_time[-1]))
