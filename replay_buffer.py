import numpy as np
from typing import Tuple, List
from functools import reduce
import operator


class ReplayBuffer:
    def __init__(self, item_shapes: List[Tuple[int]], max_elements: int, minibatch_size: int):
        self.item_shapes = item_shapes
        self.item_widths = list(reduce(operator.mul, s, 1) for s in item_shapes)
        self.row_size = sum(self.item_widths)
        self.data = np.zeros(shape=(max_elements, self.row_size))

        self.num_elements = 0
        self.oldest_index = 0
        self.max_elements = max_elements
        self.minibatch_size = minibatch_size

    def add(self, *items: np.ndarray) -> None:
        self.data[self.oldest_index] = self._encode(list(items))
        self.num_elements = min(self.num_elements+1, self.max_elements)
        self.oldest_index = (self.oldest_index+1) % self.max_elements

    def select_minibatch(self) -> List[np.ndarray]:
        indices = np.random.choice(self.num_elements, self.minibatch_size)
        return self._decode(self.data[indices])

    def clear(self) -> None:
        self.num_elements = 0
        self.oldest_index = 0

    def _encode(self, items: List[np.ndarray]) -> np.ndarray:
        """
        Encode a single element from a list of numpy arrays to a contiguous row

        Args:
            items: The list of items to decode, with shapes corresponding to self.element_shapes

        Returns:
            The items formatted as a row of the replay buffer

        >>> ReplayBuffer([(1,), (3, 2)], 100, 10)._encode([np.zeros((1,)), np.zeros((3,2))]).shape == (7,)
        True

        """
        return np.concatenate([
            np.ravel(item) for item in items
        ])

    def _decode(self, minibatch: np.ndarray) -> List[np.ndarray]:
        """
        Decodes an entire minibatch of
        Args:
            minibatch: An entire minibatch of rows such that minibatch[i] is the ith row (doesn't actually have to be
                the same size as a normal minibatch)

        Returns:

        """
        items = []
        col = 0

        for item_shape, item_width in zip(self.item_shapes, self.item_widths):
            items.append(minibatch[:, col:col+item_width].reshape((minibatch.shape[0], *item_shape)))
            col += item_width
        return items


if __name__ == '__main__':
    item_shapes = [(2, 3), (4,)]
    buffer = ReplayBuffer(item_shapes, 100, 10)

    row1 = buffer._encode([
        np.array([[1, 2, 3],
                  [4, 5, 6]]),
        np.array([10, 11, 12, 13]),
    ])
    row2 = buffer._encode([
        -np.array([[1, 2, 3],
                  [4, 5, 6]]),
        -np.array([10, 11, 12, 13]),
    ])
    print('row1={}'.format(row1))
    print('row2={}'.format(row2))
    minibatch = np.asarray([row1, row2])
    print('minibatch=\n{}'.format(minibatch))
    decoded = buffer._decode(minibatch)
    print(decoded)
