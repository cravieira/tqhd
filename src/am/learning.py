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
            # paper.
            logit = am.search(input)
            pred_class = torch.argmax(logit)
            if pred_class != idx:
                am.sub(input, pred_class)
                am.add(input, idx)
