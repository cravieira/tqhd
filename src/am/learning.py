from abc import ABC, abstractmethod
from torch import Tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .am import BaseAM

class BaseLearningStrategy(ABC):
    """docstring for BaseLearningStrategy"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, am, input, idx, retrain=False):
        """
        Update the AM with the given input vectors using a custom learning
        strategy.
        """
        pass

class Centroid(BaseLearningStrategy):
    """
    Centroid learning strategy.
    """
    def __init__(self):
        pass

    def update(self, am: 'BaseAM', input: Tensor, idx: Tensor, retrain=False):
        """docstring for update"""
        am.add(input, idx)

class OnlineHD(BaseLearningStrategy):
    """
    Centroid learning strategy. Adapted from torchhd source code.
    """
    def __init__(self, learning_rate=1.0):
        self.lr = learning_rate

    def update(self, am: 'BaseAM', input: Tensor, target: Tensor, retrain=False):
        if am.vsa != 'MAP':
            raise RuntimeError('OnlineHD is only supported with MAP AMs.')

        logit = am.search(input)
        pred = logit.argmax(1)
        is_wrong = target != pred

        # cancel update if all predictions were correct
        if is_wrong.sum().item() == 0:
            return

        # only update wrongly predicted inputs
        logit = logit[is_wrong]
        input = input[is_wrong]
        target = target[is_wrong]
        pred = pred[is_wrong]

        alpha1 = 1.0 - logit.gather(1, target.unsqueeze(1))
        alpha2 = logit.gather(1, pred.unsqueeze(1)) - 1.0

        am.add(self.lr * alpha1 * input, target)
        am.add(self.lr * alpha2 * input, pred)

        am.train_am()
