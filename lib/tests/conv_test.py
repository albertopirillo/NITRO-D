import numpy as np
import torch

from lib.layers.conv import IntegerConv2d


def test_forward():
    for _ in range(25):
        # Generate random parameters
        batch_size = np.random.randint(1, 32)
        in_channels = np.random.randint(1, 100)
        out_channels = np.random.randint(1, 100)
        img_size = np.random.randint(2, 50)
        kernel_size = np.random.randint(1, np.minimum(img_size, 15))
        padding = np.random.randint(0, 3)
        bias = np.random.choice([True, False])
        dtype = np.random.choice(['int32'])
        stride = 1

        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        input_size = (batch_size, in_channels, img_size, img_size)
        weight_size = (out_channels, in_channels, kernel_size, kernel_size)
        bias_size = (1, out_channels, 1, 1)

        # Generate random weights and set them to the layers
        my_conv = IntegerConv2d(in_channels, out_channels, kernel_size, stride, padding, bias, dtype=dtype,
                                device='cpu')
        weights = np.random.randint(-100, 100, size=weight_size, dtype=dtype)
        bias_value = np.random.randint(-100, 100, size=bias_size, dtype=dtype)
        x = np.random.randint(-100, 100, size=input_size, dtype=dtype)

        conv.weight = torch.nn.Parameter(torch.as_tensor(weights), requires_grad=False)
        my_conv.weights = weights

        if bias:
            conv.bias = torch.nn.Parameter(torch.as_tensor(bias_value.squeeze(axis=(0, 2, 3))), requires_grad=False)
            my_conv.bias = bias_value

        # Check that the output is the same
        assert np.isclose(conv(torch.as_tensor(x)), my_conv(x)).all()


def test_gradients():
    for _ in range(50):
        # Generate random parameters
        batch_size = np.random.randint(1, 8)
        in_channels = np.random.randint(1, 16)
        out_channels = np.random.randint(1, 16)
        img_size = np.random.randint(2, 25) * 2
        bias = np.random.choice([True, False])
        kernel_size = 3
        padding = 1
        stride = 1

        conv = IntegerConv2d(in_channels, out_channels, kernel_size, stride, padding, bias, debug=True)
        conv.train()

        input_size = (batch_size, in_channels, img_size, img_size)
        weights_size = (out_channels, in_channels, kernel_size, kernel_size)
        output_size = (batch_size, out_channels, img_size, img_size)
        bias_size = (1, out_channels, 1, 1)

        x = np.random.randint(-25, 25, size=input_size)
        delta = np.random.randint(-25, 25, size=output_size)
        y = conv(x)
        dX, dW, db = conv.compute_gradients(delta)

        assert y.shape == output_size
        assert dX.shape == input_size
        assert dW.shape == weights_size
        if bias:
            assert db.shape == bias_size
        else:
            assert db is None


def test_backward():
    # Test with bias
    batch_size = 1
    in_channels = 2
    out_channels = 2
    img_size = 4
    kernel_size = 3
    padding = 1
    stride = 1
    bias = True

    np.random.seed(42)
    x = np.random.randint(-100, 100, size=(batch_size, in_channels, img_size, img_size))
    delta = np.random.randint(-100, 100, size=(batch_size, out_channels, img_size, img_size))

    conv = IntegerConv2d(in_channels, out_channels, kernel_size, stride, padding, bias, debug=True)
    conv.weights = np.array([[[[1, 1, 2], [-2, -1, -1], [2, 0, -2]], [[2, -2, -2], [-2, 2, -1], [-1, -2, 0]]],
                             [[[-3, 2, -3], [0, 2, -2], [-1, -3, 1]], [[1, 0, -2], [-3, -2, -3], [0, 0, 1]]]])
    conv.bias = np.array([[[[-20]], [[36]]]])
    conv.train()

    conv(x)
    expected_x_pad = np.array([[[[0, 0, 0, 0, 0, 0], [0, 2, 79, -8, -86, 0], [0, 6, -29, 88, -80, 0],
                                 [0, 2, 21, -26, -13, 0], [0, 16, -1, 3, 51, 0], [0, 0, 0, 0, 0, 0]],
                                [[0, 0, 0, 0, 0, 0], [0, 30, 49, -48, -99, 0], [0, -13, 57, -63, 29, 0],
                                 [0, 91, 87, -80, 60, 0], [0, -43, -79, -12, -52, 0], [0, 0, 0, 0, 0, 0]]]])
    assert conv.last_input_shape == (1, 2, 4, 4)
    assert np.equal(conv.last_input_padded, expected_x_pad).all()

    new_delta = conv.backward(delta, 1024)
    expected_new_delta = np.array([[
        [[-205, 430, -372, -63], [225, -516, -255, 52], [-19, -441, 569, -48], [243, -178, 243, -166]],
        [[-685, -183, -440, 42], [-236, -312, 41, 342], [-159, 682, 483, -459], [363, -651, 117, -393]]]])
    assert np.equal(new_delta, expected_new_delta).all()
    assert conv.training is True
    assert conv.weights.shape == (out_channels, in_channels, kernel_size, kernel_size)
    assert conv.bias.shape == (1, out_channels, 1, 1)

    expected_weights = np.array(
        [[[[-4, 4, -5], [-14, -8, 9], [11, -8, 1]], [[-4, 9, 1], [-9, -9, 22], [-21, -17, -6]]],
         [[[6, -5, -3], [10, -5, -14], [-6, 6, -15]], [[-1, -4, -18], [7, 8, -6], [11, -11, 11]]]])
    expected_bias = np.array([[[[-20]], [[36]]]])
    assert np.equal(conv.weights, expected_weights).all()
    assert np.equal(conv.bias, expected_bias).all()

    # Test without bias
    bias = False
    conv = IntegerConv2d(in_channels, out_channels, kernel_size, stride, padding, bias, debug=True)
    conv.weights = np.array([[[[1, 1, 2], [-2, -1, -1], [2, 0, -2]], [[2, -2, -2], [-2, 2, -1], [-1, -2, 0]]],
                             [[[-3, 2, -3], [0, 2, -2], [-1, -3, 1]], [[1, 0, -2], [-3, -2, -3], [0, 0, 1]]]])
    assert conv.bias is None
    conv.train()

    conv(x)
    assert conv.last_input_shape == (1, 2, 4, 4)
    assert np.equal(conv.last_input_padded, expected_x_pad).all()

    new_delta = conv.backward(delta, 1024)
    assert np.equal(new_delta, expected_new_delta).all()

    assert conv.training is True
    assert conv.weights.shape == (out_channels, in_channels, kernel_size, kernel_size)
    assert conv.bias is None

    assert np.equal(conv.weights, expected_weights).all()
    assert conv.bias is None
