import numpy as np
import torch.nn

from lib.layers.linear import IntegerLinear
from lib.utils.enums import Initialization


def test_init():
    for _ in range(100):
        # Generate random parameters
        in_features = np.random.randint(1, 100)
        out_features = np.random.randint(1, 100)
        bias = np.random.choice([True, False])
        bound = 64

        # Generate a random layer and check that the weights are initialized to 0
        integer_linear = IntegerLinear(in_features, out_features, bias, init=Initialization.UNIFORM_STD)
        assert integer_linear.weights.shape == (out_features, in_features)
        assert (integer_linear.weights >= -bound).all()
        assert (integer_linear.weights <= bound).all()

        if bias:
            assert integer_linear.bias.shape == (1, out_features)
            assert np.all(integer_linear.bias >= -bound)
            assert np.all(integer_linear.bias <= bound)
        else:
            assert integer_linear.bias is None


def test_forward():
    for _ in range(100):
        # Generate random parameters
        in_features = np.random.randint(1, 100)
        out_features = np.random.randint(1, 100)
        bias = np.random.choice([False, True])
        dtype = np.random.choice(['int32'])

        linear = torch.nn.Linear(in_features, out_features, bias=bias)
        # Generate random weights and set them to the layers
        my_linear = IntegerLinear(in_features, out_features, bias=bias, dtype=dtype)
        weights = np.random.randint(-100, 100, size=(out_features, in_features), dtype=dtype)
        bias_value = np.random.randint(-100, 100, size=(1, out_features), dtype=dtype)
        x = np.random.randint(-100, 100, size=(1, in_features), dtype=dtype)

        linear.weight = torch.nn.Parameter(torch.tensor(weights), requires_grad=False)
        my_linear.weights = weights

        if bias:
            linear.bias = torch.nn.Parameter(torch.tensor(bias_value), requires_grad=False)
            my_linear.bias = bias_value

        # Check that the output is the same
        assert np.isclose(linear(torch.tensor(x)), my_linear(x)).all()


def test_backward():
    # Test with bias
    in_features = 4
    out_features = 3
    bias = True
    batch_size = 2
    lr_inv = 1024

    np.random.seed(42)
    x = np.random.randint(-100, 100, size=(batch_size, in_features))
    y = np.random.randint(0, out_features, size=batch_size)
    y = np.eye(out_features, dtype=np.int32)[y]

    linear = IntegerLinear(in_features, out_features, bias)
    linear.weights = np.array([[-1, 3, -1, -1], [1, 0, -1, 2], [1, -2, 0, 2]], dtype=np.int32)
    linear.bias = np.array([[2, -2, 0]], dtype=np.int32)
    linear.training = True

    y_pred = linear(x)
    expected_y_pred = np.array([[331, -164, -328], [-99, -244, -96]], dtype=np.int32)
    assert np.equal(y_pred, expected_y_pred).all()

    loss_grad = y_pred - y
    delta = linear.backward(loss_grad, lr_inv)
    expected_delta = np.array([[-824, 1651, -167, -1317], [-242, -105, 344, -583]], dtype=np.int32)
    assert np.equal(delta, expected_delta).all()

    assert linear.training is True
    assert linear.weights.shape == (out_features, in_features)
    assert linear.bias.shape == (1, out_features)

    expected_weights = np.array([[-1, -25, 10, 19], [2, 5, 18, -30], [2, 20, 5, -33]], dtype=np.int32)
    expected_bias = np.array([[2, -2, 0]], dtype=np.int32)
    assert np.equal(linear.weights, expected_weights).all()
    assert np.equal(linear.bias, expected_bias).all()

    # Test without bias
    bias = False
    linear = IntegerLinear(in_features, out_features, bias)
    linear.weights = np.array([[-1, 3, -1, -1], [1, 0, -1, 2], [1, -2, 0, 2]], dtype=np.int32)
    assert linear.bias is None
    linear.training = True

    y_pred = linear(x)
    expected_y_pred = np.array([[329, -162, -328], [-101, -242, -96]], dtype=np.int32)
    assert np.equal(y_pred, expected_y_pred).all()

    loss_grad = y_pred - y
    delta = linear.backward(loss_grad, lr_inv)
    expected_delta = np.array([[-820, 1645, -167, -1311], [-238, -111, 344, -577]], dtype=np.int32)
    assert np.equal(delta, expected_delta).all()

    assert linear.training is True
    assert linear.weights.shape == (out_features, in_features)
    assert linear.bias is None

    expected_weights = np.array([[-1, -25, 10, 18], [2, 5, 18, -30], [2, 20, 5, -33]], dtype=np.int32)
    assert np.equal(linear.weights, expected_weights).all()
    assert linear.bias is None
