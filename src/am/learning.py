from abc import ABC, abstractmethod
import torch
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
        # TODO: This implementation might not work with batched tensors
        if retrain == False:
            am.add(input, idx)
        else:
            # Retraining algorithm is based on the one described in VoiceHD's
            # paper and used in several HDC proposals.
            logit = am.search(input)
            pred_class = torch.argmax(logit)
            if pred_class != idx:
                am.sub(input, pred_class)
                am.add(input, idx)

class CentroidOnline(Centroid):
    """
    Centroid learning strategy with online AM update during retraining. This
    class may converge faster to higher accuracies in retraining.
    """
    def __init__(self):
        pass

    def update(self, am: 'BaseAM', input: Tensor, idx: Tensor, retrain=False):
        """docstring for update"""
        super().update(am, input, idx, retrain=retrain)
        if retrain:
            # Update AM in use for the subsequent predictions in the
            # retraining loop
            am.train_am()

class OnlineHD(BaseLearningStrategy):
    """
    Centroid learning strategy. Adapted from torchhd source code.
    """
    def __init__(self, learning_rate=1.0):
        self.lr = learning_rate
        self._supported_vsas = ['MAP', 'FHRR', 'BSC']

    def update(self, am: 'BaseAM', input: Tensor, target: Tensor, retrain=False):
        if am.vsa not in self._supported_vsas:
            raise RuntimeError(f'OnlineHD is only supported with {self._supported_vsas} AMs.')

        logit = am.search(input)
        if am.vsa == 'BSC':
            # Transform logit from a hamming distance value defined in [0, D]
            # to a cosine like similarity [-1, 1]
            dim = input.shape[-1]
            logit = (2*logit - dim) / dim

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

        if am.vsa != 'BSC':
            am.add(self.lr * alpha1 * input, target)
            am.add(self.lr * alpha2 * input, pred)
        else:
            bip_input = torch.where(input.to(torch.int) > 0, 1, -1)
            am.add(self.lr * alpha1 * bip_input, target, weight=self.lr*alpha1)
            am.add(self.lr * alpha2 * bip_input, pred, weight=-self.lr*alpha2)

        am.train_am()

