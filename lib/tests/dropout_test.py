import numpy as np
import torch

from lib.layers.misc import Dropout


def test_rate():
    np.random.seed(0)
    for _ in range(100):
        # Generate random parameters
        batch_size = np.random.randint(1, 32)
        channels = np.random.randint(1, 100)
        height = np.random.randint(16, 100)
        width = np.random.randint(16, 100)
        dropout_rate = np.random.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                                         0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        # Instantiate the input
        x = np.random.randint(-256, 256, size=(batch_size, channels, height, width), dtype=np.int16)

        # Instantiate dropout layers
        my_dropout = Dropout(dropout_rate)
        my_dropout.train()
        torch_dropout = torch.nn.Dropout(dropout_rate)

        # Forward pass
        my_y = my_dropout(x)
        torch_y = torch_dropout(torch.tensor(x, dtype=torch.float32))

        # Assert that the number of zeros is within a certain range
        num_elements = np.prod(x.shape)
        assert np.sum(my_y == 0) - np.sum(torch_y.numpy() == 0.0) < (num_elements * 0.05)


def test_magnitude():
    np.random.seed(0)
    for _ in range(100):
        # Generate random parameters
        batch_size = np.random.randint(1, 32)
        channels = np.random.randint(1, 100)
        height = np.random.randint(16, 100)
        width = np.random.randint(16, 100)
        dropout_rate = np.random.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                                         0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        # Instantiate the input
        x = np.random.randint(-256, 256, size=(batch_size, channels, height, width), dtype=np.int16)

        # Instantiate dropout layer
        my_dropout = Dropout(dropout_rate)
        my_dropout.train()

        # Forward pass
        my_y = my_dropout(x)

        # Check that the average magnitude of the input is unchanged
        original_mean = np.mean(np.abs(x))
        assert np.abs(np.mean(np.abs(my_y)) - original_mean) < original_mean * 0.1

        # Check that the magnitude of single elements is the same
        original_mean = np.mean(np.abs(x), axis=(1, 2, 3))
        assert (np.abs(np.mean(np.abs(my_y), axis=(1, 2, 3)) - original_mean) < (original_mean * 0.1)).all()
