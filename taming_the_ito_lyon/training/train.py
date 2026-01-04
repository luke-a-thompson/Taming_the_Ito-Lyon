import jax
import equinox as eqx

from taming_the_ito_lyon.models import Model
import optax
from typing import Callable


@eqx.filter_jit
def batch_loss_conditional(
    model: Model,
    ts_b: jax.Array,
    target_b: jax.Array,
    x_b: jax.Array,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """
    Compute the loss for a batch of data where the model is conditioned on the control values.
    """
    def predict(ts_i: jax.Array, x_i: jax.Array) -> jax.Array:
        return model(ts_i, x_i)

    preds = jax.vmap(predict)(ts_b, x_b)
    return loss_fn(preds, target_b)


grad_fn = eqx.filter_value_and_grad(batch_loss_conditional)


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
    return float(batch_loss_conditional(model, timestep_b, solution_b, drivers_b, loss_fn))
