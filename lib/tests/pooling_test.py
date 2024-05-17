import numpy as np
import torch

from lib.layers.pooling import MaxPool2d, GlobalMaxPool2d, IntegerAvgPool2d


def test_max_forward():
    for i in range(100):
        # Generate random parameters
        batch_size = np.random.randint(1, 32)
        in_channels = np.random.randint(1, 100)
        img_height = np.random.randint(1, 50)
        img_width = np.random.randint(1, 50)
        kernel_height = np.random.choice(np.arange(1, img_height + 1, 1))
        kernel_width = np.random.choice(np.arange(1, img_width + 1, 1))
        stride_height = kernel_height
        stride_width = kernel_width
        padding_height = np.random.randint(0, (kernel_height // 2) + 1)
        padding_width = np.random.randint(0, (kernel_width // 2) + 1)
        dtype = np.random.choice(['int32', 'float32'])

        pool = torch.nn.MaxPool2d((kernel_height, kernel_width),
                                  (stride_height, stride_width),
                                  (padding_height, padding_width))
        my_pool = MaxPool2d((kernel_height, kernel_width),
                            (stride_height, stride_width),
                            (padding_height, padding_width), dtype)

        input_size = (batch_size, in_channels, img_height, img_width)

        if dtype == 'int32':
            x = np.random.randint(-100, 100, size=input_size, dtype=dtype)
        else:
            x = np.random.uniform(-1, 1, size=input_size)

        assert np.isclose(pool(torch.tensor(x)), my_pool(x)).all()


def test_avg_forward():
    for i in range(100):
        # Generate random parameters
        batch_size = np.random.randint(1, 32)
        in_channels = np.random.randint(1, 100)
        img_height = np.random.randint(1, 50)
        img_width = np.random.randint(1, 50)
        kernel_height = np.random.choice(np.arange(1, img_height + 1, 1))
        kernel_width = np.random.choice(np.arange(1, img_width + 1, 1))
        stride_height = kernel_height
        stride_width = kernel_width
        padding_height = np.random.randint(0, (kernel_height // 2) + 1)
        padding_width = np.random.randint(0, (kernel_width // 2) + 1)
        dtype = np.random.choice(['float32'])

        pool = torch.nn.AvgPool2d((kernel_height, kernel_width),
                                  (stride_height, stride_width),
                                  (padding_height, padding_width))
        my_pool = IntegerAvgPool2d((kernel_height, kernel_width),
                                   (stride_height, stride_width),
                                   (padding_height, padding_width), dtype)

        input_size = (batch_size, in_channels, img_height, img_width)

        if dtype == 'int32':
            x = np.random.randint(-100, 100, size=input_size, dtype=dtype)
        else:
            x = np.random.uniform(-1, 1, size=input_size)

        assert (np.abs(pool(torch.tensor(x)) - my_pool(x)) <= 1).all()


def test_max_backward_squared():
    batch_size = 2
    in_channels = 2
    img_size = 4
    kernel_size = 2
    stride = 2

    input_shape = (batch_size, in_channels, img_size, img_size)
    output_shape = (batch_size, in_channels, img_size // stride, img_size // stride)

    np.random.seed(42)
    x = np.random.randint(-100, 100, size=input_shape)
    delta = np.random.randint(-100, 100, size=output_shape)

    pool = MaxPool2d(kernel_size, stride)
    pool.train()
    y = pool(x)
    new_delta = pool.backward(delta, 1)

    expected_y_pred = np.array(
        [[[[79, 88], [21, 51]], [[57, 29], [91, 60]]], [[[89, 89], [30, 34]], [[66, 31], [10, 98]]]]
    )

    expected_mask = np.array([[[[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                               [[0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]],
                              [[[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]],
                               [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]]]])

    expected_new_delta = np.array([[[[0, 71, 0, 0], [0, 0, -93, 0], [0, 74, 0, 0], [0, 0, 0, -66]],
                                    [[0, 0, 0, 0], [0, -20, 0, 63], [-51, 0, 0, 3], [0, 0, 0, 0]]],
                                   [[[0, 0, 0, 0], [31, 0, 0, -99], [0, 0, 0, 0], [33, 0, -47, 0]],
                                    [[0, 5, 0, -97], [0, 0, 0, 0], [0, 0, 0, 0], [0, -47, 0, 90]]]])

    assert pool.training is True
    assert y.shape == output_shape
    assert pool.input_shape == input_shape
    assert new_delta.shape == input_shape
    assert np.equal(y, expected_y_pred).all()
    assert np.equal(pool.mask.reshape(input_shape), expected_mask).all()
    assert np.equal(new_delta, expected_new_delta).all()


def test_max_backward_rectangular():
    batch_size = 2
    in_channels = 2
    img_height = 12
    img_width = 6
    kernel_height = 4
    kernel_width = 2
    stride_height = 4
    stride_width = 2
    padding_height = 2
    padding_width = 1

    input_size = (batch_size, in_channels, img_height, img_width)
    input_size_padded = (batch_size, in_channels, img_height + 2 * padding_height, img_width + 2 * padding_width)
    output_size = (batch_size, in_channels, 4, 4)

    np.random.seed(42)
    x = np.random.randint(-25, 25, size=input_size)
    delta = np.random.randint(-25, 25, size=output_size)

    pool = MaxPool2d(kernel_size=(kernel_height, kernel_width),
                     stride=(stride_height, stride_width),
                     padding=(padding_height, padding_width), dtype='int8')
    pool.train()
    y = pool(x)
    expected_new_delta = np.array([
        [[[9, -7, 0, 22, 0, 0], [0, 0, 0, 0, 0, -10], [-23, 0, 0, 0, 0, 0], [0, -6, 0, 0, 0, 0],
          [0, 0, 0, -2, 0, 7], [0, 0, 0, 0, 0, 0], [-2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -18],
          [0, 0, 0, 0, 0, 0], [0, -15, 0, 23, 0, 0], [10, 0, 12, 14, 0, 0], [0, 0, 0, 0, 0, -6]],
         [[0, 22, 0, -1, 0, 0], [9, 0, 0, 0, 0, 9], [0, 0, 3, 0, -8, 20], [0, 0, 0, 0, 0, 0],
          [-1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -10], [-8, 0, 0, 0, 0, 0],
          [0, 0, 0, 9, 0, 0], [0, 0, -24, 0, 0, 0], [15, 0, 10, 0, 7, 0], [0, 0, 0, 0, 0, -22]]],
        [[[7, 0, 0, 0, 0, 22], [0, 0, -12, -5, 0, 0], [0, 0, 0, 0, -19, 0], [-6, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0], [0, 0, -18, 0, 0, -23], [0, 0, 0, 22, 0, 0], [0, 0, 7, 0, 0, 0],
          [0, 0, 0, 0, 0, -14], [-9, 0, 0, 0, 0, 0], [-4, -4, 0, 0, 0, 0], [0, 0, 0, 20, 0, 4]],
         [[0, 0, 12, 0, 0, -18], [12, 0, 0, 0, 19, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, -5],
          [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0], [0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 21],
          [0, 0, 0, 0, 2, 0], [0, 0, 7, 0, 0, 0], [7, 0, -21, 0, 0, -7], [0, 0, 0, 22, 0, 0]]]])

    expected_mask = np.array([
        [[[0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]]],
        [[[0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]]])

    new_delta = pool.backward(delta, 1)
    assert pool.training is True
    assert y.shape == output_size
    assert pool.input_shape == input_size
    assert pool.input_shape_pad == input_size_padded
    assert new_delta.shape == input_size
    assert np.equal(pool.mask.reshape(input_size_padded), expected_mask).all()
    assert np.equal(new_delta, expected_new_delta).all()


def test_global_max_pooling():
    batch_size = np.random.randint(2, 32)
    in_channels = np.random.randint(1, 128)
    img_height = np.random.randint(1, 64)
    img_width = np.random.randint(1, 64)

    for _ in range(100):
        x = np.random.randint(-25, 25, size=(batch_size, in_channels, img_height, img_width))
        my_gmp = GlobalMaxPool2d()
        torch_gmp = torch.nn.MaxPool2d(kernel_size=x.shape[2:])
        assert np.isclose(my_gmp(x), torch_gmp(torch.Tensor(x))).all()
