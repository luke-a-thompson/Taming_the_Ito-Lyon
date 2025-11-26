from .ncde import NeuralCDE
from .nrde import NeuralRDE

type Model = NeuralCDE | NeuralRDE

__all__ = ["Model", "NeuralCDE", "NeuralRDE"]
