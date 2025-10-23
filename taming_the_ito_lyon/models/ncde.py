"""
Neural Controlled Differential Equation model. Taken from the Diffrax documentation.

https://docs.kidger.site/diffrax/examples/neural_cde/
"""

import equinox as eqx
import jax
import jax.nn as jnn
import jax.random as jr
import diffrax
from collections.abc import Callable


class CDEFunc(eqx.Module):
    """
    Vector field for a Neural CDE.

    Given hidden state y in R^{hidden_size}, returns matrix in R^{hidden_size x data_size}
    which is multiplied against dx/dt.
    """

    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        *,
        key: jax.Array,
    ) -> None:
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t: float, y: jax.Array, args: None) -> jax.Array:
        del t, args
        out = self.mlp(y)
        return out.reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):
    """
    Neural Controlled Differential Equation model.

    Usage
    - Provide `ts` and either a `diffrax` control path or cubic interpolation coeffs.
    - The model solves the induced ODE and applies a readout on the hidden state.
    """

    # Modules
    initial: eqx.nn.MLP
    func: CDEFunc
    readout: eqx.nn.Linear

    # Static configuration
    readout_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    rtol: float
    atol: float
    dtmin: float
    evolving_out: bool

    def __init__(
        self,
        input_path_dim: int,
        hidden_size: int,
        output_path_dim: int,
        width_size: int,
        depth: int,
        *,
        key: jax.Array,
        readout_activation: Callable[[jax.Array], jax.Array] | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        dtmin: float = 1e-6,
        evolving_out: bool = True,
    ) -> None:
        k1, k2, k3 = jr.split(key, 3)
        self.initial = eqx.nn.MLP(
            in_size=input_path_dim,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=k1,
        )
        self.func = CDEFunc(
            data_size=input_path_dim,
            hidden_size=hidden_size,
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
        self.rtol = rtol
        self.atol = atol
        self.dtmin = dtmin
        self.evolving_out = bool(evolving_out)

    def _solve(
        self,
        ts: jax.Array,
        control: diffrax.AbstractPath,
    ) -> jax.Array:
        term = diffrax.ControlTerm(self.func, control).to_ode()
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(
            rtol=self.rtol, atol=self.atol, dtmin=self.dtmin
        )
        y0 = self.initial(control.evaluate(ts[0]))
        saveat = diffrax.SaveAt(ts=ts) if self.evolving_out else diffrax.SaveAt(t1=True)
        solution = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
        )
        if self.evolving_out:
            return solution.ys
        return solution.ys[-1]

    def __call__(
        self,
        ts: jax.Array,
        control_or_coeffs: diffrax.AbstractPath | tuple[jax.Array, ...],
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

        hidden_states = self._solve(ts, control)

        if self.evolving_out:
            # Map readout across time
            def apply_readout(y: jax.Array) -> jax.Array:
                return self.readout_activation(self.readout(y))

            return jax.vmap(apply_readout)(hidden_states)
        else:
            return self.readout_activation(self.readout(hidden_states))
