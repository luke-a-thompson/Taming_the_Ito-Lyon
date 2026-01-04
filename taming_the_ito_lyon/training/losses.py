import jax
import jax.numpy as jnp
from typing import Callable


def mse_loss(
    pred: jax.Array,
    target: jax.Array,
) -> jax.Array:
    return jnp.mean((pred - target) ** 2)


def rotational_geodesic_loss(
    pred: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """
    Compute the Rotational Geodesic Error (RGE) loss.
    RGE(R1, R2) = 2 * arcsin(||R2 - R1||_F / (2√2))

    Args:
        pred: Predicted rotation matrices
        target: Target rotation matrices

    Returns:
        Mean RGE loss
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"
    assert pred.shape[-1] == pred.shape[-2], "pred and target must be square matrices"
    frobenius_norm = jnp.linalg.norm(pred - target, ord="fro", axis=(-2, -1))
    rge = 2.0 * jnp.arcsin(frobenius_norm / (2.0 * jnp.sqrt(2.0)))
    return jnp.mean(rge)


def make_sigker_loss(
    order: int = 5,
    static_kernel: str = "linear",
    solver: str = "monomial_approx",
    max_batch: int = 100,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Create a signature-kernel MMD loss that instantiates SigKernel once.

    This is the non-adversarial objective: match the *distribution* of paths
    via MMD between two batches of sample paths.
    """
    from polysigkernel import SigKernel

    signature_kernel = SigKernel(
        order=order,
        static_kernel=static_kernel,
        solver=solver,
    )

    def loss(pred: jax.Array, target: jax.Array) -> jax.Array:
        return signature_kernel.compute_mmd(pred, target, max_batch=max_batch)

    return loss
