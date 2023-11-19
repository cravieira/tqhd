from abc import ABC, abstractmethod
import torch
from torch import Tensor
from math import ceil

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .am import BaseAM

class BasePredictionStrategy(ABC):
    """docstring for BasePredictionStrategy"""
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def apply(self, am: 'BaseAM'):
        """
        Return an AM with the prediction strategy applied.
        """
        pass

    def __call__(self, am: 'BaseAM'):
        return self.apply(am)

class Normal(BasePredictionStrategy):
    """docstring for Normal"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply(self, am: 'BaseAM'):
        """docstring for apply"""
        return am.am

class Fault(BasePredictionStrategy):
    """
    Inflicts error in an binary AM to simulate how a faulty/innacurate hardware
    would behave when executing similarity search.
    """
    def __init__(self, fault_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.fr = fault_rate

    def _randomizer(self, input: torch.Tensor, rate: float):
        """Flip a binary tensor at random positions."""
        size = torch.numel(input)
        size_faulty = ceil(size*rate)

        indices = torch.randperm(size)[0:size_faulty]
        out = torch.clone(input)
        temp = out.flatten()
        temp[indices] = torch.logical_not(temp[indices]).type(input.dtype)
        return out

    def apply(self, am: 'BaseAM'):
        """
        Return an AM with errors inserted in the dimension according to the
        fault rate in this class.
        """
        return self._randomizer(am.am, self.fr)


