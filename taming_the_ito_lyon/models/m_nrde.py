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

from stochastax.vector_field_lifts.split_vector_fields import split_multi_vector_field

from stochastax.hopf_algebras import (
    HopfAlgebra,
    ShuffleHopfAlgebra,
    GLHopfAlgebra,
    MKWHopfAlgebra,
)
from stochastax.vector_field_lifts.bck_lift import form_bck_bracket_functions
from stochastax.vector_field_lifts.lie_lift import form_lyndon_bracket_functions
from stochastax.vector_field_lifts.mkw_lift import form_mkw_bracket_functions
from stochastax.vector_field_lifts.vector_field_lift_types import (
    VectorFieldBracketFunctionLift,
)
from stochastax.manifolds import Manifold
from .logsignatures import (
    compute_windowed_logsignatures_from_values,
)
from .extrapolation import ExtrapolationScheme
from .logsig_cde_solve import (
    solve_cde_from_multiwindow_logsigs,
    solve_cde_from_windowed_logsigs_piecewise,
)
from taming_the_ito_lyon.config.config_options import HopfAlgebraType


def lipswish(x: jax.Array) -> jax.Array:
    return 0.909 * jnn.silu(x)


class MNRDEFunc(eqx.Module):
    """Vector field for a MNDRE, expressed as a CDE over (flattened) log-signatures.

    This returns a matrix of shape (cde_state_dim, logsig_size) which multiplies the
    time-derivative of a cumulative log-signature control path.
    """

    input_path_dim: int
    vf_mlp: eqx.nn.MLP
    cde_state_dim: int
    hidden_manifold: type[Manifold] = eqx.field(static=True)
    hopf_algebra: HopfAlgebra = eqx.field(static=True)
    vf_lift: VectorFieldBracketFunctionLift

    def __init__(
        self,
        *,
        input_path_dim: int,
        cde_state_dim: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        hopf_algebra: HopfAlgebra,
        vf_lift: VectorFieldBracketFunctionLift,
        hidden_manifold: type[Manifold],
        key: jax.Array,
    ) -> None:
        # Vector field
        self.cde_state_dim = cde_state_dim
        self.input_path_dim = input_path_dim
        self.vf_mlp = eqx.nn.MLP(
            in_size=cde_state_dim,
            out_size=input_path_dim * cde_state_dim,
            width_size=vf_hidden_dim,
            depth=vf_mlp_depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=key,
        )

        # Rough paths
        self.hidden_manifold = hidden_manifold
        self.hopf_algebra = hopf_algebra
        self.vf_lift = vf_lift

    def __call__(self, t: jax.typing.ArrayLike, y: jax.Array, args: None) -> jax.Array:
        del t, args

        y_retracted = self.hidden_manifold.retract(y)
        # One driving channel per vector field
        vector_fields = split_multi_vector_field(
            self.vf_mlp, self.input_path_dim, self.cde_state_dim
        )
        # Form the bracket functions for the vector fields
        bracket_functions = self.vf_lift(
            vector_fields,
            self.hopf_algebra,
            self.hidden_manifold(),
        )

        # Flatten the bracket functions to a single list of length m (logsig_size)
        flat_bracket_functions = [bf for level in bracket_functions for bf in level]
        # Back to tangent space. THIS EVALUATES vf_mlp(y)
        cols = [
            self.hidden_manifold.project_to_tangent(y_retracted, bf(y_retracted))
            for bf in flat_bracket_functions
        ]
        return jnp.stack(cols, axis=1)


class MNDRE(eqx.Module):
    """
    Discrete log-ODE version of NCDE that enforces Lyndon Lie polynomials.

    Usage:
    - Standard mode (n_recon=None): model(ts, coeffs) -> outputs
    - Extrapolation mode (n_recon set): model(ts, x) -> outputs
      where ts covers both reconstruction and future times
    """

    initial_cond_mlp: eqx.nn.MLP
    cde_func: MNRDEFunc
    cde_funcs_multi: tuple[MNRDEFunc, ...] | None
    readout_layer: eqx.nn.Linear
    output_shift: jax.Array

    # Extrapolation scheme
    extrapolation_scheme: ExtrapolationScheme | None
    n_recon: int | None = eqx.field(static=True)

    # Static configuration
    hopf_algebra: HopfAlgebra = eqx.field(static=True)
    vf_lift: VectorFieldBracketFunctionLift = eqx.field(static=True)
    data_manifold: type[Manifold] = eqx.field(static=True)
    hidden_manifold: type[Manifold] = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)
    signature_window_size: int = eqx.field(static=True)
    signature_window_sizes: tuple[int, ...] | None = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)

    # Solver configuration (matches NeuralCDE/NeuralRDE pattern)
    solver: diffrax.AbstractAdaptiveSolver = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(static=True)

    def __init__(
        self,
        input_path_dim: int,
        cde_state_dim: int,
        output_path_dim: int,
        initial_hidden_dim: int,
        initial_cond_mlp_depth: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        signature_depth: int,
        signature_window_size: int,
        signature_window_sizes: list[int] | None = None,
        *,
        key: jax.Array,
        data_manifold: type[Manifold],
        hidden_manifold: type[Manifold],
        hopf_algebra_type: HopfAlgebraType,
        solver: diffrax.AbstractAdaptiveSolver = diffrax.Tsit5(),
        stepsize_controller: diffrax.AbstractStepSizeController,
        # IMPORTANT: default to identity to avoid artificially symmetrising/skew-clipping
        # the output distribution (e.g. tanh produces symmetric outputs about 0).
        # This matches the NCDE/LogNCDE defaults in this repo.
        readout_activation: Callable[[jax.Array], jax.Array] = lambda x: x,
        evolving_out: bool = True,
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
    ) -> None:
        # If using multi-window, we need one vector field per window size.
        window_sizes_tuple: tuple[int, ...] | None = None
        if signature_window_sizes is not None:
            if len(signature_window_sizes) == 0:
                raise ValueError(
                    "signature_window_sizes must be non-empty when provided."
                )
            # Keep stable ordering; remove duplicates.
            uniq = sorted({int(x) for x in signature_window_sizes})
            window_sizes_tuple = tuple(uniq)

        num_keys = 3 + (
            len(window_sizes_tuple) if window_sizes_tuple is not None else 1
        )
        keys = jr.split(key, num_keys)
        k1 = keys[0]
        k_readout = keys[1]
        k_shift = keys[2]
        vf_keys = list(keys[3:])

        # Rough paths
        self.data_manifold = data_manifold
        self.hidden_manifold = hidden_manifold
        self.signature_depth = signature_depth
        self.signature_window_size = signature_window_size
        self.signature_window_sizes = window_sizes_tuple
        match hopf_algebra_type:
            case HopfAlgebraType.SHUFFLE:
                self.hopf_algebra = ShuffleHopfAlgebra.build(
                    input_path_dim, signature_depth
                )
                self.vf_lift = form_lyndon_bracket_functions
            case HopfAlgebraType.GL:
                self.hopf_algebra = GLHopfAlgebra.build(input_path_dim, signature_depth)
                self.vf_lift = form_bck_bracket_functions
            case HopfAlgebraType.MKW:
                self.hopf_algebra = MKWHopfAlgebra.build(
                    input_path_dim, signature_depth
                )
                self.vf_lift = form_mkw_bracket_functions
            case _:
                raise ValueError(f"Unsupported Hopf algebra type: {hopf_algebra_type}")

        # Module
        self.initial_cond_mlp = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=cde_state_dim,
            width_size=initial_hidden_dim,
            depth=initial_cond_mlp_depth,
            activation=lipswish,
            key=k1,
        )
        # Single-window default vector field
        self.cde_func = MNRDEFunc(
            input_path_dim=input_path_dim,
            cde_state_dim=cde_state_dim,
            vf_hidden_dim=vf_hidden_dim,
            vf_mlp_depth=vf_mlp_depth,
            hopf_algebra=self.hopf_algebra,
            vf_lift=self.vf_lift,
            hidden_manifold=self.hidden_manifold,
            key=vf_keys[0],
        )
        # Multi-window vector fields (one per scale). For quick A/B, we keep one
        # separate parameter set per scale and sum their induced terms.
        if window_sizes_tuple is None:
            self.cde_funcs_multi = None
        else:
            if len(vf_keys) != len(window_sizes_tuple):
                raise ValueError(
                    "Internal error: unexpected key split for multi-window MNRDE."
                )
            self.cde_funcs_multi = tuple(
                MNRDEFunc(
                    input_path_dim=input_path_dim,
                    cde_state_dim=cde_state_dim,
                    vf_hidden_dim=vf_hidden_dim,
                    vf_mlp_depth=vf_mlp_depth,
                    hopf_algebra=self.hopf_algebra,
                    vf_lift=self.vf_lift,
                    hidden_manifold=self.hidden_manifold,
                    key=k,
                )
                for k in vf_keys
            )
        self.readout_layer = eqx.nn.Linear(
            in_features=cde_state_dim,
            out_features=output_path_dim,
            use_bias=True,
            key=k_readout,
        )
        self.output_shift = jr.uniform(k_shift, (), minval=0.9, maxval=1.1)
        self.readout_activation = readout_activation

        # Static configuration
        self.extrapolation_scheme = extrapolation_scheme
        self.n_recon = n_recon
        self.evolving_out = evolving_out
        self.solver = solver
        self.stepsize_controller = stepsize_controller

    def _apply_readout(self, hidden_states: jax.Array) -> jax.Array:
        """Apply readout to hidden states."""

        def apply_single(y: jax.Array) -> jax.Array:
            activation = self.readout_activation(self.readout_layer(y))
            activation = activation + self.output_shift
            retracted = self.data_manifold.retract(activation)
            return retracted

        return jax.vmap(apply_single)(hidden_states)

    def _forward_with_values(
        self,
        ts: jax.Array,
        control_values: jax.Array,
    ) -> jax.Array:
        """Core forward pass given sampled control values (standard mode fast path)."""
        x0 = control_values[0]
        h0 = self.initial_cond_mlp(x0)

        # Multi-window mode: sum multiple logsignature-driven CDE terms, one per window size.
        if self.signature_window_sizes is not None:
            assert self.cde_funcs_multi is not None
            logsigs_list = [
                compute_windowed_logsignatures_from_values(
                    control_values,
                    self.hopf_algebra,
                    self.signature_depth,
                    int(w),
                )
                for w in self.signature_window_sizes
            ]
            return solve_cde_from_multiwindow_logsigs(
                ts,
                logsigs_list,
                signature_window_sizes=[int(w) for w in self.signature_window_sizes],
                cde_funcs=[f for f in self.cde_funcs_multi],
                y0=h0,
                solver=self.solver,
                stepsize_controller=self.stepsize_controller,
            )

        # Single-window mode (existing behavior)
        logsigs = compute_windowed_logsignatures_from_values(
            control_values,
            self.hopf_algebra,
            self.signature_depth,
            self.signature_window_size,
        )
        return solve_cde_from_windowed_logsigs_piecewise(
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

        Standard mode (self.extrapolation_scheme=None):
            model(control_values) -> outputs

        Extrapolation mode (self.extrapolation_scheme is set):
            model(control_values) -> outputs
        """
        length = control_values.shape[0]
        ts = jnp.linspace(0.0, 1.0, length, dtype=control_values.dtype)  # (T,)
        if self.extrapolation_scheme is not None:
            assert self.n_recon is not None, (
                "n_recon must be set when using extrapolation_scheme"
            )
            control, _ = self.extrapolation_scheme.create_control(
                ts, control_values, self.n_recon
            )
            control_values = jax.vmap(control.evaluate)(ts)
            hidden = self._forward_with_values(ts, control_values)
            outputs = self._apply_readout(hidden)

            return outputs
        else:
            # Standard mode
            hidden = self._forward_with_values(ts, control_values)

            if self.evolving_out:
                return self._apply_readout(hidden)

            # Single output case: also convert from 6D to 3x3 (matches `ncde.py`).
            final_output = self.readout_activation(self.readout_layer(hidden[-1]))
            final_output = final_output + self.output_shift
            return self.data_manifold.retract(final_output)
