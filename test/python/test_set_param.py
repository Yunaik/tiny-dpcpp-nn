import numpy as np
import torch

import tiny_dpcpp_nn as tcnn


def test_set_params():
    device = "xpu"

    torch.manual_seed(42)

    WIDTH = 16
    # avoid padding
    n_input_dims = WIDTH
    n_output_dims = WIDTH
    DTYPE = torch.bfloat16
    BATCH_SIZE = 8
    x = (torch.ones(BATCH_SIZE, WIDTH) * 0.1).to(
        device, dtype=DTYPE
    )  # torch.distributions.uniform.Uniform(-0.01,0.01).sample((1024,n_input_dims)).to(device)

    config = {
        "otype": "FullyFusedMLP",
        "activation": "None",
        "output_activation": "None",
        "n_neurons": WIDTH,
        "n_hidden_layers": 3,
        "device": device,
    }

    network = tcnn.Network(
        n_input_dims,
        n_output_dims,
        config,
        input_dtype=DTYPE,
        backend_param_dtype=DTYPE,
    )

    # tiny-cuda-nn seems to use always float32 for params
    # assert network.params.dtype == torch.float32

    val = 0.123
    param_vals = val * torch.ones(WIDTH * WIDTH * 4, 1, dtype=DTYPE).to(device)
    network.set_params(param_vals)

    # Using torch.isclose to compare param_vals with network.params and network.params.data
    is_close_params = torch.isclose(param_vals, network.params)
    is_close_params_data = torch.isclose(param_vals, network.params.data)

    print(f"is_close_params: {is_close_params.all()}")
    print(f"is_close_params_data: {is_close_params_data.all()}")

    print("Input")
    print(x[0:WIDTH, 0])
    y = network(x)
    print("Output")
    print(y[0:10, 0])


if __name__ == "__main__":
    test_set_params()
