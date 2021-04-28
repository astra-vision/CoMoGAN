"""This package includes a miscellaneous collection of useful helper functions."""
from torch.nn import DataParallel

import sys

class DataParallelPassthrough(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
