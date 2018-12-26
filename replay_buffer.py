import numpy as np
from typing import Tuple


class ReplayBuffer:
    def __init__(self, element_shape: Tuple[int], max_elements: int, minibatch_size: int):
        self.data = np.zeros(shape=(max_elements, *element_shape))
        self.num_elements = 0
        self.oldest_index = 0
        self.max_elements = max_elements
        self.minibatch_size = minibatch_size

    def add(self, element: np.ndarray) -> None:
        assert element.shape == self.data.shape[1:]
        self.data[self.oldest_index] = element
        self.num_elements = max(self.num_elements+1, self.max_elements)
        self.oldest_index = (self.oldest_index+1) % self.max_elements

    def select_minibatch(self) -> np.ndarray:
        if self.num_elements == self.max_elements:
            return np.random.choice(self.data, self.minibatch_size)
        else:
            return np.random.choice(self.data[:self.num_elements], self.minibatch_size)
