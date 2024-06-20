import numpy as np
import torch

import tiny_dpcpp_nn as tcnn

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
    n_input_dims, n_output_dims, config, input_dtype=DTYPE, backend_param_dtype=DTYPE
)

# tiny-cuda-nn seems to use always float32 for params
assert network.params.dtype == torch.float32

val = 0.123
network.set_params(val * torch.ones(WIDTH * WIDTH * 4, 1).to(device))

print(f"Parameters 1: {network.params}")
print(f"Parameters 2: {network.params.data}")
print("Input")
print(x[0:WIDTH, 0])
y = network(x)
print("Output")
print(y[0:10, 0])
