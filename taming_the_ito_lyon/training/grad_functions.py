import jax
import equinox as eqx
from collections.abc import Callable
from taming_the_ito_lyon.models import Model


@eqx.filter_jit
def batch_loss(
    model: Model,
    control_values_b: jax.Array,
    target_b: jax.Array,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """
    Compute the loss for a batch of data where the model is given control values.

    Args:
        model: The model to evaluate
        control_values_b: Batch of control paths, shape (batch_size, T, control_dim)
        target_b: Batch of target paths, shape (batch_size, T, target_dim)
        loss_fn: Loss function that takes (predictions, targets) and returns scalar

    Returns:
        Scalar loss value
    """
    preds = jax.vmap(model)(control_values_b)
    return loss_fn(preds, target_b)
