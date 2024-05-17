import numpy as np
import torch
import torch.nn as nn

from lib.layers.activations import PocketTanh, PocketReLU, BipolarPocketReLU, PocketLeakyReLU, \
    NitroLeakyReLU, BipolarLeakyReLU
from lib.utils.misc import truncated_division
from lib.utils.nn import l2_loss_grad


def test_pocket_tanh():
    # Compare the outputs with the original PLA tanh
    def pla_tanh(act_in, out_dim):
        y_max, y_min = 127, -127
        int_max = np.iinfo(np.int32).max
        intervals = [y_max, 75, 32, -31, -74, y_min]
        slopes_inv = [int_max, 8, 2, 1, 2, 8, int_max]

        act_out, act_grad_inv = (np.full((act_in.shape[0], out_dim), y_max),
                                 np.full((act_in.shape[0], out_dim), slopes_inv[0]))

        for i in range(len(act_in)):
            for j in range(len(act_in[i].squeeze())):
                val = act_in[i].squeeze()[j]
                if val < intervals[0]:
                    act_out[i][j] = truncated_division(val, 3) + 83
                    act_grad_inv[i][j] = slopes_inv[1]
                if val < intervals[1]:
                    act_out[i][j] = val + 32
                    act_grad_inv[i][j] = slopes_inv[2]
                if val < intervals[2]:
                    act_out[i][j] = val * 2
                    act_grad_inv[i][j] = slopes_inv[3]
                if val < intervals[3]:
                    act_out[i][j] = val - 32
                    act_grad_inv[i][j] = slopes_inv[4]
                if val < intervals[4]:
                    act_out[i][j] = truncated_division(val, 3) - 83
                    act_grad_inv[i][j] = slopes_inv[5]
                if val < intervals[5]:
                    act_out[i][j] = y_min
                    act_grad_inv[i][j] = slopes_inv[6]
        return act_out.astype(int), act_grad_inv

    for _ in range(1000):
        in_features = np.random.randint(2, 100)
        batch_size = np.random.randint(1, 64)

        bound = np.iinfo(np.int8).max * 2
        x = np.random.randint(-bound, bound, size=(batch_size, in_features))

        y1, y1_prime = pla_tanh(x, in_features)
        my_tanh = PocketTanh()
        my_tanh.training = True
        y2 = my_tanh(x)
        y2_prime = my_tanh.grad_inv

        assert np.equal(y1, y2).all()
        assert np.equal(y1_prime, y2_prime).all()


def test_pocket_relu():
    # Compare the outputs with the original PLA relu
    def pla_relu(act_in, out_dim):
        y_max, y_min = 127, -128
        intervals = [y_max, 0]
        slopes_inv = [1, 0]

        act_out, act_grad = (np.full((act_in.shape[0], out_dim), y_max),
                             np.full((act_in.shape[0], out_dim), 0))

        for i in range(len(x)):
            for j in range(len(x[i])):
                val = act_in[i].squeeze()[j]
                if val <= intervals[0]:
                    act_out[i][j] = val
                    act_grad[i][j] = slopes_inv[0]
                if val <= intervals[1]:
                    act_out[i][j] = 0
                    act_grad[i][j] = slopes_inv[1]

        return act_out.astype(int), act_grad

    for _ in range(1000):
        in_features = np.random.randint(2, 100)
        batch_size = np.random.randint(1, 64)

        bound = np.iinfo(np.int8).max * 2
        x = np.random.randint(-bound, bound, size=(batch_size, in_features))

        y1, y1_prime = pla_relu(x, in_features)
        my_relu = PocketReLU()
        my_relu.training = True
        y2 = my_relu(x)
        y2_prime = my_relu.backward(np.ones_like(x), 1)

        assert np.equal(y1, y2).all()
        assert np.equal(y1_prime, y2_prime).all()


def test_pocket_relu_backward():
    def torch_pocket_relu(x_in):
        return torch.minimum(torch.nn.functional.relu(x_in), torch.full_like(x_in, 127))

    my_relu = PocketReLU()
    my_relu.train()
    rss = nn.MSELoss(reduction='sum')

    for _ in range(100):
        batch_size = np.random.randint(1, 64)
        in_features = np.random.randint(2, 100)

        x: torch.Tensor = torch.randint(-256, 256, size=(batch_size, in_features)).type(torch.float64)
        y: torch.Tensor = torch.randint(-256, 256, size=(batch_size, in_features)).type(torch.float64)
        x.requires_grad_(True)

        # Forward
        y1 = torch_pocket_relu(x)
        y1.retain_grad()
        y2 = my_relu(x.detach().numpy())
        assert np.equal(y1.detach(), y2).all()

        # Backward
        loss_torch = rss(y, y1)
        loss_torch.retain_grad()
        loss_torch.backward()
        loss_grad_np = 2 * l2_loss_grad(y.numpy(), y2, dtype='float64')
        my_relu.backward(loss_grad_np, 1)

        bw_torch = x.grad.numpy().astype(np.int32)
        bw_torch[x == 127] *= 2
        bw_np = my_relu.backward(loss_grad_np, 1).astype(np.int32)
        assert np.equal(y1.grad, loss_grad_np).all()
        assert np.equal(bw_torch, bw_np).all()


def test_saturation():
    for _ in range(100):
        batch_size = np.random.randint(1, 32)
        num_channels = np.random.randint(1, 64)
        height = np.random.randint(2, 32)
        width = np.random.randint(2, 32)
        dtype = np.random.choice([np.int8, np.int16, np.int32])
        int_min, int_max = np.iinfo(dtype).min, np.iinfo(dtype).max
        x = np.random.randint(int_min, int_max, size=(batch_size, num_channels, height, width), dtype=dtype)
        saturated_negative = x < -127
        saturated_positive = x > 127

        # Instantiate all pocket activations
        pocket_activations = [
            PocketTanh(),
            PocketReLU(),
            BipolarPocketReLU(),
            PocketLeakyReLU(),
            NitroLeakyReLU(),
            BipolarLeakyReLU()
        ]

        for f in pocket_activations:
            f.train()
            y = f(x)
            delta = f.backward(x, 1)
            assert np.all(y >= -127)
            assert np.all(y <= 127)
            assert np.all(delta[saturated_negative] == 0)
            assert np.all(delta[saturated_positive] == 0)
