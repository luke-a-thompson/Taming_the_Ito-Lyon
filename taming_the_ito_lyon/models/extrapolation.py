"""
Pluggable extrapolation schemes for Neural CDE models.

These schemes take reconstruction data and future time points,
and return a control path that covers the full time range
(reconstruction + future), with the future portion extrapolated.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import diffrax
from diffrax._custom_types import RealScalarLike
from taming_the_ito_lyon.config.config_options import ExtrapolationSchemeType


class ExtrapolationScheme(Protocol):
    """Protocol for extrapolation schemes."""

    @abstractmethod
    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[diffrax.AbstractPath, jax.Array]:
        """Create a control path that covers both reconstruction and future times.

        Fits the control on reconstruction data, then extrapolates to future times.

        Args:
            t_all: All timestamps of shape (T_total,) covering reconstruction + future
            x_all: Input data of shape (T_total, D), but only first n_recon are used
            n_recon: Number of reconstruction points to fit on (split index)

        Returns:
            Tuple of:
            - control: A diffrax.AbstractPath covering [t_all[0], t_all[-1]]
            - t_all: The full time array passed in
        """
        ...


class LinearScheme(eqx.Module):
    """Linear interpolation on reconstruction, linear extrapolation for future.

    - Within t_recon: linear interpolation between data points
    - Beyond t_recon: continues linearly based on last segment's slope
    """

    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[diffrax.LinearInterpolation, jax.Array]:
        if t_all.shape[0] < n_recon or x_all.shape[0] < n_recon:
            raise ValueError(
                f"t_all and x_all must have length >= n_recon, got {t_all.shape[0]} and {x_all.shape[0]}"
            )
        t_recon = t_all[:n_recon]
        x_recon = x_all[:n_recon]
        # Prepend time channel: (n_recon, D) -> (n_recon, 1+D)
        ys_with_time = jnp.concatenate([t_recon[:, None], x_recon], axis=1)
        # Fit only on reconstruction data - diffrax will extrapolate linearly
        control = diffrax.LinearInterpolation(ts=t_recon, ys=ys_with_time)
        return control, t_all


class HermiteScheme(eqx.Module):
    """Cubic spline on reconstruction, polynomial extrapolation for future.

    - Within t_recon: cubic spline interpolation
    - Beyond t_recon: continues using polynomial from last segment (can be unstable)
    """

    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[diffrax.CubicInterpolation, jax.Array]:
        if t_all.shape[0] < n_recon or x_all.shape[0] < n_recon:
            raise ValueError(
                f"t_all and x_all must have length >= n_recon, got {t_all.shape[0]} and {x_all.shape[0]}"
            )
        t_recon = t_all[:n_recon]
        x_recon = x_all[:n_recon]
        # Prepend time channel: (n_recon, D) -> (n_recon, 1+D)
        ys_with_time = jnp.concatenate([t_recon[:, None], x_recon], axis=1)
        # Fit only on reconstruction data - diffrax will extrapolate
        coeffs = diffrax.backward_hermite_coefficients(ts=t_recon, ys=ys_with_time)
        control = diffrax.CubicInterpolation(ts=t_recon, coeffs=coeffs)
        return control, t_all


class WeightedSGScheme(eqx.Module):
    """Savitzky-Golay polynomial fit on reconstruction, polynomial extrapolation.

    Fits a single polynomial to the reconstruction data, providing smooth
    extrapolation to future times.

    Uses the same numerical stability features as the SO3 implementation:
    - Center-based polynomial expansion
    - Normalized polynomial basis (t^k / max(1, k))
    - Raw weights with softplus for positivity (no softmax)
    """

    weights: jax.Array
    poly_order: int = eqx.field(static=True)

    def __init__(self, num_points: int, poly_order: int = 3, *, key: jax.Array):
        self.weights = (
            jax.random.normal(key, (num_points,)) * 0.1
        )  # Raw weights, learnable
        self.poly_order = poly_order

    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[_PolynomialPath, jax.Array]:
        """Create polynomial path that extrapolates smoothly to future."""
        if t_all.shape[0] < n_recon or x_all.shape[0] < n_recon:
            raise ValueError(
                f"t_all (shape {t_all.shape}) and x_all (shape {x_all.shape}) must have length >= n_recon ({n_recon})"
            )
        t_recon = t_all[:n_recon]
        x_recon = x_all[:n_recon]
        seq_len = n_recon

        # Resize weights if needed
        if self.weights.shape[0] != seq_len:
            weights = jnp.interp(
                jnp.linspace(0, 1, seq_len),
                jnp.linspace(0, 1, self.weights.shape[0]),
                self.weights,
            )
        else:
            weights = self.weights

        # Ensure positive weights
        weights = jax.nn.softplus(weights)

        # Center-based polynomial expansion (like SO3 version)
        center_idx = seq_len // 2
        t_center = t_recon[center_idx]

        # Normalized polynomial basis: t^k / max(1, k)
        normalizers = jnp.maximum(1.0, jnp.arange(self.poly_order + 1))

        def fit_channel(x_channel: jax.Array) -> jax.Array:
            """Fit polynomial and return coefficients."""
            t_rel = t_recon - t_center
            # Vandermonde with normalized basis
            V = jnp.vander(t_rel, N=self.poly_order + 1, increasing=True)
            V = V / normalizers[None, :]

            # Weighted least squares
            sqrt_W = jnp.sqrt(weights)
            V_w = V * sqrt_W[:, None]
            b_w = x_channel * sqrt_W

            coeffs = jnp.linalg.lstsq(V_w, b_w)[0]
            return coeffs

        # Fit polynomial for each channel
        poly_coeffs = jax.vmap(fit_channel, in_axes=1, out_axes=0)(x_recon)

        control = _PolynomialPath(
            poly_coeffs=poly_coeffs,
            t_center=t_center,
            normalizers=normalizers,
            t0=t_all[0],
            t1=t_all[-1],  # Extends to future!
        )
        return control, t_all


class _PolynomialPath(diffrax.AbstractPath):
    """Diffrax-compatible path backed by a polynomial."""

    poly_coeffs: jax.Array  # (D, poly_order+1)
    t_center: jax.Array
    normalizers: jax.Array
    powers_phi: jax.Array  # (poly_order+1,)
    powers_first: jax.Array  # (poly_order,)
    t0: RealScalarLike  # type: ignore
    t1: RealScalarLike  # type: ignore

    def __init__(
        self,
        poly_coeffs: jax.Array,
        t_center: jax.Array,
        normalizers: jax.Array,
        t0: RealScalarLike,
        t1: RealScalarLike,
    ) -> None:
        object.__setattr__(self, "poly_coeffs", poly_coeffs)
        object.__setattr__(self, "t_center", t_center)
        object.__setattr__(self, "normalizers", normalizers)
        poly_order = poly_coeffs.shape[1] - 1
        # Pre-compute power arrays for efficient derivative computation
        object.__setattr__(
            self, "powers_phi", jnp.arange(poly_order + 1, dtype=jnp.float32)
        )
        object.__setattr__(
            self, "powers_first", jnp.arange(poly_order, dtype=jnp.float32)
        )
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "t1", t1)

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is None:
            t_rel = t0 - self.t_center
            t_powers = (t_rel**self.powers_phi) / self.normalizers
            x_t = self.poly_coeffs @ t_powers
            # Prepend time channel: (D,) -> (1+D,)
            return jnp.concatenate([jnp.array([t0]), x_t])
        else:
            return self.evaluate(t1) - self.evaluate(t0)

    def derivative(self, t, left=True, order=1):
        """Compute time derivative using analytic polynomial derivative.

        Args:
            t: Time point for derivative evaluation
            left: Not used for deterministic paths
            order: Derivative order (only 1 is currently supported)

        Returns:
            First derivative vector of shape (1+D,) = [1, dx/dt]
        """
        del left
        if order != 1:
            raise ValueError(f"Only derivative order 1 is supported, got {order}")

        t_rel = t - self.t_center
        # For derivative: d/dt [t^k / max(1,k)] =
        #   - k=0: 0 (constant term)
        #   - k>=1: t^(k-1) (no extra normalizer needed for derivative)
        # So we use coeffs[:, 1:] with powers_first = arange(poly_order)
        if self.poly_coeffs.shape[1] > 1:
            # Derivative coefficients: skip constant term
            deriv_coeffs = self.poly_coeffs[:, 1:]  # (D, poly_order)
            t_powers_deriv = t_rel**self.powers_first  # (poly_order,)
            x_dot = deriv_coeffs @ t_powers_deriv  # (D,)
        else:
            # Constant polynomial (order 0) has zero derivative
            x_dot = jnp.zeros(self.poly_coeffs.shape[0])

        # Time derivative is constant 1.0
        # Return shape (1+D,) = [1, dx/dt]
        return jnp.concatenate([jnp.array([1.0]), x_dot])


class MLPScheme(eqx.Module):
    """MLP-based extrapolation with data conditioning.

    The MLP takes both time and a context vector (encoding of reconstruction data)
    to produce outputs. This allows it to adapt to individual sequences while
    still learning a general extrapolation function.
    """

    encoder: eqx.nn.MLP
    decoder: eqx.nn.MLP
    input_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        context_dim: int = 16,
        hidden_dim: int = 32,
        depth: int = 2,
        *,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)
        self.input_dim = input_dim
        self.context_dim = context_dim

        # Encoder: maps reconstruction data to context vector
        self.encoder = eqx.nn.MLP(
            in_size=input_dim,
            out_size=context_dim,
            width_size=hidden_dim,
            depth=depth,
            key=k1,
        )

        # Decoder: maps (time, context) to output
        self.decoder = eqx.nn.MLP(
            in_size=1 + context_dim,  # time + context
            out_size=input_dim,
            width_size=hidden_dim,
            depth=depth,
            key=k2,
        )

    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[_MLPPath, jax.Array]:
        if t_all.shape[0] < n_recon or x_all.shape[0] < n_recon:
            raise ValueError(
                f"t_all and x_all must have length >= n_recon, got {t_all.shape[0]} and {x_all.shape[0]}"
            )
        x_recon = x_all[:n_recon]

        # Encode reconstruction data into context (mean pooling)
        contexts = jax.vmap(self.encoder)(x_recon)  # (n_recon, context_dim)
        context = jnp.mean(contexts, axis=0)  # (context_dim,)

        control = _MLPPath(
            decoder=self.decoder,
            context=context,
            t0=float(t_all[0]),
            t1=float(t_all[-1]),
        )
        return control, t_all


class _MLPPath(diffrax.AbstractPath):
    """Diffrax-compatible path backed by a conditioned MLP."""

    decoder: eqx.nn.MLP
    context: jax.Array
    # t0 and t1 inherited from AbstractPath

    def __init__(self, decoder: eqx.nn.MLP, context: jax.Array, t0: float, t1: float):
        self.decoder = decoder
        self.context = context
        self.t0 = t0  # type: ignore[misc]
        self.t1 = t1  # type: ignore[misc]

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is None:
            # Normalize time to [0, 1] based on full range
            t_norm = (t0 - self.t0) / (self.t1 - self.t0 + 1e-8)
            # Concatenate time and context
            mlp_input = jnp.concatenate([jnp.atleast_1d(t_norm), self.context])
            x_t = self.decoder(mlp_input)
            # Prepend time channel: (D,) -> (1+D,)
            return jnp.concatenate([jnp.array([t0]), x_t])
        else:
            return self.evaluate(t1) - self.evaluate(t0)


class SO3SGScheme(eqx.Module):
    """SO(3) Savitzky-Golay scheme for rotation matrix data.

    Uses manifold-aware polynomial fitting in the tangent space (Lie algebra).
    Ensures outputs remain on SO(3) through exponential map and orthogonalization.

    Handles flattened (T, 9) driver data from SO(3) datasets, which is the standard
    format used by the SO3DynamicsSim dataloader.
    """

    poly_order: int = eqx.field(static=True)
    weights: jax.Array | None = None

    def __init__(
        self,
        poly_order: int = 3,
        num_points: int | None = None,
        *,
        key: jax.Array | None = None,
    ):
        self.poly_order = poly_order
        # Optional: learn weights (like WeightedSGScheme)
        if num_points is not None and key is not None:
            self.weights = jax.random.normal(key, (num_points,)) * 0.1
        else:
            self.weights = None

    def create_control(
        self,
        t_all: jax.Array,
        x_all: jax.Array,
        n_recon: int,
    ) -> tuple[diffrax.AbstractPath, jax.Array]:
        from taming_the_ito_lyon.utils.savitzky_golay_so3 import SO3PolynomialPath
        
        if t_all.shape[0] < n_recon:
            raise ValueError(f"t_all must have length >= n_recon")
        
        # Use FULL time array for polynomial, not just reconstruction
        # The polynomial will fit on first n_recon points but cover full range
        x_recon = x_all[:n_recon]
        t_recon = t_all[:n_recon]
        
        # Reshape to (n_recon, 3, 3)
        if x_recon.ndim == 2 and x_recon.shape[-1] == 9:
            x_recon = x_recon.reshape(x_recon.shape[0], 3, 3)
        
        # Add batch dimension
        R = x_recon[None, ...]  # (1, n_recon, 3, 3)
        
        # Process weights - must be batched!
        weights = None
        if self.weights is not None:
            if self.weights.shape[0] != n_recon:
                weights_1d = jnp.interp(
                    jnp.linspace(0, 1, n_recon),
                    jnp.linspace(0, 1, self.weights.shape[0]),
                    jax.nn.softplus(self.weights),
                )
            else:
                weights_1d = jax.nn.softplus(self.weights)
            # Broadcast to batch dimension
            weights = weights_1d[None, :]  # (1, n_recon)
        
        # Create path with FULL time range for extrapolation
        # The key: pass t_all (not t_recon) so path covers full range
        control = SO3PolynomialPath(
            R=R,
            t=t_recon,  # CHANGED: Full time array
            p=self.poly_order,
            weight=weights,  # Now (1, n_recon) or None
        )
        
        return control, t_all


def create_scheme(
    name: ExtrapolationSchemeType,
    *,
    input_dim: int | None = None,
    num_points: int | None = None,
    poly_order: int = 3,
    hidden_dim: int = 32,
    mlp_depth: int = 2,
    key: jax.Array | None = None,
) -> ExtrapolationScheme:
    """Factory function for extrapolation schemes.

    Args:
        name: One of ExtrapolationSchemeType
        input_dim: Required for 'mlp' scheme
        num_points: Required for 'sg' scheme (reconstruction sequence length)
        poly_order: Polynomial order for 'sg' scheme (default: 3)
        hidden_dim: Hidden dimension for 'mlp' scheme
        mlp_depth: Depth for 'mlp' scheme
        key: Random key (required for 'sg' and 'mlp')

    Returns:
        An ExtrapolationScheme instance

    Extrapolation behavior:
        - 'linear': Linear continuation from last segment
        - 'hermite': Polynomial from last cubic segment (can be unstable)
        - 'sg': Smooth polynomial extrapolation (fitted to all recon data)
        - 'so3_sg': SO(3) manifold-aware SG for rotation data (flattened as 9D)
        - 'mlp': Learned extrapolation (MLP outputs for any time)
    """
    match name:
        case "linear":
            return LinearScheme()
        case "hermite":
            return HermiteScheme()
        case "sg":
            if num_points is None or key is None:
                raise ValueError("'sg' scheme requires num_points and key")
            return WeightedSGScheme(
                num_points=num_points, poly_order=poly_order, key=key
            )
        case "so3_sg":
            return SO3SGScheme(poly_order=poly_order, num_points=num_points, key=key)
        case "mlp":
            if input_dim is None or key is None:
                raise ValueError("'mlp' scheme requires input_dim and key")
            return MLPScheme(
                input_dim=input_dim,
                context_dim=hidden_dim // 2,  # Use half hidden_dim for context
                hidden_dim=hidden_dim,
                depth=mlp_depth,
                key=key,
            )
        case _:
            raise ValueError(
                f"Unknown scheme: {name}. Use 'linear', 'hermite', 'sg', 'so3_sg', or 'mlp'"
            )
