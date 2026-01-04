import jax
import jax.numpy as jnp
import equinox as eqx

from taming_the_ito_lyon.models import Model
import optax
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


@eqx.filter_jit
def batch_loss(
    model: Model,
    ts_b: jax.Array,
    target_b: jax.Array,
    x_b: jax.Array,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    def predict_path(
        ts_i: jax.Array,
        x_i: jax.Array,
    ) -> jax.Array:
        return model(ts_i, x_i)

    preds = jax.vmap(predict_path)(ts_b, x_b)
    return loss_fn(preds, target_b)


grad_fn = eqx.filter_value_and_grad(batch_loss)


def train_epoch(
    model: Model,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    loader,
    num_batches: int,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> tuple[float, Model, optax.OptState]:
    @eqx.filter_jit(donate="all")
    def step(
        timestep_b: jax.Array,
        solution_b: jax.Array,
        drivers_b: jax.Array,
        model: Model,
        opt_state: optax.OptState,
    ) -> tuple[jax.Array, Model, optax.OptState]:
        loss_value, grads = grad_fn(model, timestep_b, solution_b, drivers_b, loss_fn)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, new_opt_state = optim.update(grads, opt_state, params)
        updated_model: Model = eqx.apply_updates(model, updates)
        return loss_value, updated_model, new_opt_state

    total_loss = 0.0
    for _ in range(num_batches):
        timestep_b, solution_b, drivers_b = next(loader)
        loss_value, model, opt_state = step(
            timestep_b,
            solution_b,
            drivers_b,
            model,
            opt_state,
        )
        total_loss += float(loss_value)

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, model, opt_state


def eval_epoch(
    model: Model,
    timestep_b: jax.Array,
    solution_b: jax.Array,
    drivers_b: jax.Array,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> float:
    return float(batch_loss(model, timestep_b, solution_b, drivers_b, loss_fn))
