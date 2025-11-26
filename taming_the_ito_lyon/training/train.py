import jax
import jax.numpy as jnp
import equinox as eqx

from taming_the_ito_lyon.models import Model
import optax


def batch_mse_loss(
    model: Model,
    ts_b: jax.Array,
    target_b: jax.Array,
    coeffs_b: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    def predict_path(
        t_i: jax.Array, c_i: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
    ) -> jax.Array:
        return model(t_i, c_i)

    preds = jax.vmap(predict_path)(ts_b, coeffs_b)
    return jnp.mean((preds - target_b) ** 2)


@eqx.filter_jit
def _jitted_batch_mse_loss(
    model: Model,
    ts_b: jax.Array,
    target_b: jax.Array,
    coeffs_b: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    return batch_mse_loss(model, ts_b, target_b, coeffs_b)


def eval_epoch(
    model: Model,
    timestep_b: jax.Array,
    solution_b: jax.Array,
    drivers_b: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> float:
    return float(_jitted_batch_mse_loss(model, timestep_b, solution_b, drivers_b))


def train_epoch(
    model: Model,
    optim: optax.GradientTransformation,
    opt_state: optax.OptState,
    loader,
    num_batches: int,
) -> tuple[float, Model, optax.OptState]:
    grad_fn = eqx.filter_value_and_grad(batch_mse_loss)

    @eqx.filter_jit(donate="all")
    def step(
        timestep_b: jax.Array,
        solution_b: jax.Array,
        drivers_b: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        model: Model,
        opt_state: optax.OptState,
    ) -> tuple[jax.Array, Model, optax.OptState]:
        loss_value, grads = grad_fn(model, timestep_b, solution_b, drivers_b)
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
