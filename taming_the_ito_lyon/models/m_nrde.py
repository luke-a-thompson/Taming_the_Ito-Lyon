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
    VectorFieldBracketFunctions,
    VectorFieldBracketFunctionLift,
)
from stochastax.manifolds import Manifold, EuclideanSpace
from .logsignatures import (
    compute_disjoint_signature_times,
    compute_windowed_logsignatures_from_control,
    compute_windowed_logsignatures_from_values,
)
from .extrapolation import ExtrapolationScheme
from taming_the_ito_lyon.config.config_options import HopfAlgebraType


class MNRDEFunc(eqx.Module):
    """Vector field for a MNDRE, expressed as a CDE over (flattened) log-signatures.

    This returns a matrix of shape (cde_state_dim, logsig_size) which multiplies the
    time-derivative of a cumulative log-signature control path.
    """

    input_path_dim: int
    vf_mlp: eqx.nn.MLP
    cde_state_dim: int
    manifold: Manifold = eqx.field(static=True)
    hopf_algebra: HopfAlgebra = eqx.field(static=True)
    vf_lift: VectorFieldBracketFunctionLift = eqx.field(static=True)

    def __init__(
        self,
        *,
        input_path_dim: int,
        cde_state_dim: int,
        vf_hidden_dim: int,
        vf_mlp_depth: int,
        hopf_algebra: HopfAlgebra,
        vf_lift: VectorFieldBracketFunctionLift,
        manifold: Manifold = EuclideanSpace(),
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
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

        # Rough paths
        self.manifold = manifold
        self.hopf_algebra = hopf_algebra
        self.vf_lift = vf_lift

    def __call__(self, t: jax.typing.ArrayLike, y: jax.Array, args: None) -> jax.Array:
        del t, args

        y_proj = self.manifold.retract(y)
        vector_fields = split_multi_vector_field(
            self.vf_mlp, self.input_path_dim, self.cde_state_dim
        )
        bracket_functions: VectorFieldBracketFunctions = self.vf_lift(
            vector_fields,
            self.hopf_algebra,
            self.manifold,
        )

        flat_bracket_functions = [
            bf
            for level in bracket_functions
            for bf in level  # type: ignore[union-attr]
        ]
        cols = [
            self.manifold.project_to_tangent(y_proj, bf(y_proj))
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
    readout_layer: eqx.nn.Linear

    # Extrapolation scheme
    extrapolation_scheme: ExtrapolationScheme | None
    n_recon: int | None = eqx.field(static=True)

    # Static configuration
    hopf_algebra: HopfAlgebra = eqx.field(static=True)
    vf_lift: VectorFieldBracketFunctionLift = eqx.field(static=True)
    manifold: Manifold = eqx.field(static=True)
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    signature_depth: int = eqx.field(static=True)
    signature_window_size: int = eqx.field(static=True)
    evolving_out: bool = eqx.field(static=True)

    # Solver configuration (matches NeuralCDE/NeuralRDE pattern)
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
        hopf_algebra_type: HopfAlgebraType,
        key: jax.Array,
        manifold: Manifold = EuclideanSpace(),
        readout_activation: Callable[[jax.Array], jax.Array] = jnn.tanh,
        evolving_out: bool = True,
        solver: diffrax.AbstractAdaptiveSolver = diffrax.Bosh3(),
        stepsize_controller: diffrax.AbstractStepSizeController = diffrax.PIDController(
            rtol=1e-2, atol=1e-3, dtmin=1e-6
        ),
        extrapolation_scheme: ExtrapolationScheme | None = None,
        n_recon: int | None = None,
    ) -> None:
        k1, k2, k3 = jr.split(key, 3)

        # Rough paths
        self.manifold = manifold
        self.signature_depth = signature_depth
        self.signature_window_size = signature_window_size
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
            hopf_algebra=self.hopf_algebra,
            vf_lift=self.vf_lift,
            manifold=self.manifold,
            key=k2,
        )
        self.readout_layer = eqx.nn.Linear(
            in_features=cde_state_dim,
            out_features=output_path_dim,
            use_bias=True,
            key=k3,
        )
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
            retracted = self.manifold.retract(activation)
            return retracted

        return jax.vmap(apply_single)(hidden_states)

    def _forward_with_control(
        self,
        ts: jax.Array,
        control: diffrax.AbstractPath,
    ) -> jax.Array:
        """Core forward pass given control path, solved via diffrax.diffeqsolve.

        We compute disjoint-window log-signatures (flattened), build their cumulative
        sum as a piecewise-linear control path in log-signature space, and solve the
        induced controlled differential equation with Diffrax.
        """
        x0 = control.evaluate(ts[0])
        h0 = self.manifold.retract(self.initial_cond_mlp(x0))
        logsigs = compute_windowed_logsignatures_from_control(
            ts,
            control,
            self.hopf_algebra,
            self.signature_depth,
            self.signature_window_size,
        )
        ts_sig = compute_disjoint_signature_times(ts, int(self.signature_window_size))
        logsig_size = int(logsigs.shape[-1])
        z0 = jnp.zeros((1, logsig_size), dtype=logsigs.dtype)
        z = jnp.concatenate([z0, jnp.cumsum(logsigs, axis=0)], axis=0)
        logsig_control = diffrax.LinearInterpolation(ts=ts_sig, ys=z)

        term = diffrax.ControlTerm(self.cde_func, logsig_control).to_ode()
        saveat = diffrax.SaveAt(ts=ts)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=h0,
            stepsize_controller=self.stepsize_controller,
            saveat=saveat,
        )
        assert solution.ys is not None
        return jax.vmap(self.manifold.retract)(solution.ys)

    def _forward_with_values(
        self, ts: jax.Array, control_values: jax.Array
    ) -> jax.Array:
        """Core forward pass given sampled control values (standard mode fast path)."""
        x0 = control_values[0]
        h0 = self.manifold.retract(self.initial_cond_mlp(x0))

        logsigs = compute_windowed_logsignatures_from_values(
            control_values,
            self.hopf_algebra,
            self.signature_depth,
            self.signature_window_size,
        )
        ts_sig = compute_disjoint_signature_times(ts, int(self.signature_window_size))
        logsig_size = int(logsigs.shape[-1])
        z0 = jnp.zeros((1, logsig_size), dtype=logsigs.dtype)
        z = jnp.concatenate([z0, jnp.cumsum(logsigs, axis=0)], axis=0)
        logsig_control = diffrax.LinearInterpolation(ts=ts_sig, ys=z)

        term = diffrax.ControlTerm(self.cde_func, logsig_control).to_ode()
        saveat = diffrax.SaveAt(ts=ts)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=h0,
            stepsize_controller=self.stepsize_controller,
            saveat=saveat,
        )
        assert solution.ys is not None
        return jax.vmap(self.manifold.retract)(solution.ys)

    def __call__(
        self,
        ts: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        """
        Forward pass.

        Standard mode (self.extrapolation_scheme=None):
            model(ts, coeffs) -> outputs

        Extrapolation mode (self.extrapolation_scheme is set):
            model(ts, x) -> outputs
        """
        if self.extrapolation_scheme is not None:
            assert self.n_recon is not None, (
                "n_recon must be set when using extrapolation_scheme"
            )
            control, _ = self.extrapolation_scheme.create_control(ts, x, self.n_recon)
            hidden = self._forward_with_control(ts, control)
            outputs = self._apply_readout(hidden)

            return outputs
        else:
            # Standard mode
            hidden = self._forward_with_values(ts, x)

            if self.evolving_out:
                return self._apply_readout(hidden)

            return self.readout_activation(self.readout_layer(hidden[-1]))
