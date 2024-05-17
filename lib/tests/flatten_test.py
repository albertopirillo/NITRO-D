import numpy as np
import torch.nn

from lib.layers.misc import Flatten


def test_forward():
    flatten = torch.nn.Flatten()
    my_flatten = Flatten()

    for _ in range(100):
        # Test with one grayscale image
        x = np.random.randint(-100, 100, size=(1, 32, 32))
        assert np.equal(flatten(torch.tensor(x)), my_flatten(x)).all()

        # Test with one RGB image
        x = np.random.randint(-100, 100, size=(3, 32, 32))
        assert np.equal(flatten(torch.tensor(x)), my_flatten(x)).all()

        # Test with a batch of grayscale images
        x = np.random.randint(-100, 100, size=(10, 32, 32))
        assert np.equal(flatten(torch.tensor(x)), my_flatten(x)).all()

        # Test with a batch of RGB images
        x = np.random.randint(-100, 100, size=(10, 3, 32, 32))
        assert np.equal(flatten(torch.tensor(x)), my_flatten(x)).all()
