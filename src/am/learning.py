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
