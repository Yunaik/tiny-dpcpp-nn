import numpy as np
import torch

if torch.cuda.is_available():
    import tinycudann as tcnn

    device = "cuda"
else:
    import tiny_dpcpp_nn as tcnn

    device = "xpu"
torch.set_printoptions(precision=10)

to_numpy = lambda a: a.detach().cpu().numpy()

torch.manual_seed(42)
# DTYPE = torch.bfloat16
DTYPE = torch.float16

# avoid padding
n_input_dims = 16
n_output_dims = 16
config = {
    "otype": "FullyFusedMLP",
    "activation": "Linear",
    "output_activation": "Linear",
    "n_neurons": 16,
    "n_hidden_layers": 1,
    "device": device,
}

network = tcnn.Network(
    n_input_dims, n_output_dims, config, input_dtype=DTYPE, backend_param_dtype=DTYPE
)

# tiny-cuda-nn seems to use always float32 for params
# assert network.params.dtype == torch.float32

# make sure we have the same initialization
torch.manual_seed(42)
params = (
    torch.distributions.uniform.Uniform(-0.216, 0.216)
    .sample(network.params.shape)
    .to(device)
)

params = torch.ones(network.params.shape) * 0.01

if device == "cuda":
    network.params.data[...] = params
else:
    network.set_params(params)

# # initialization differs between the implementations
# np.testing.assert_allclose(network.params.max().item(), 0.216, rtol=1e-2)
# np.testing.assert_allclose(network.params.min().item(), -0.216, rtol=1e-2)
# np.testing.assert_allclose(network.params.mean().item(), 0, atol=1e-2)

# x = (
#     torch.distributions.uniform.Uniform(-0.0001, 0.0001)
#     .sample((1024, n_input_dims))
#     .to(device)
# )
x = torch.ones((8, n_input_dims), dtype=DTYPE).to("xpu") * 0.01

print(f"x: {x}")
y = network(x)
print(f"y: {y}")
# np.testing.assert_allclose(to_numpy(y), 0.5)
y.backward(torch.ones_like(y))

# # Loading or saving depending on the device
# file_prefix = "results/"

# if device == "cuda":
#     torch.save(network.params, file_prefix + "params.pt")
#     torch.save(y, file_prefix + "output.pt")
#     torch.save(x, file_prefix + "input.pt")
#     torch.save(network.params.grad, file_prefix + "grads.pt")
# else:
#     x_expected = torch.load(
#         file_prefix + "input.pt", map_location=torch.device("cpu")
#     ).to(device)
#     y_expected = torch.load(
#         file_prefix + "output.pt", map_location=torch.device("cpu")
#     ).to(device)
#     grads_expected = torch.load(
#         file_prefix + "grads.pt", map_location=torch.device("cpu")
#     ).to(device)
#     params_expected = torch.load(
#         file_prefix + "params.pt", map_location=torch.device("cpu")
#     )
#     # print(f"y: {y}")
#     # print(f"y expected: {y_expected}")
#     # print(to_numpy(network.params.sum().sum()))
#     # print(to_numpy(params_expected.sum().sum()))
#     # np.testing.assert_allclose(to_numpy(params), to_numpy(params_expected))
#     # np.testing.assert_allclose(
#     #     to_numpy(network.params.sum().sum()), to_numpy(params_expected.sum().sum())
#     # )
#     np.testing.assert_allclose(to_numpy(x), to_numpy(x_expected))
#     # np.testing.assert_allclose(to_numpy(y), to_numpy(y_expected))
#     # np.testing.assert_allclose(to_numpy(network.params.grad), to_numpy(grads_expected))
# print(y)
# gradient is all 0 with tiny-dpcpp-nn but not with cuda
print(network.params.grad.T)
# assert network.params.grad.abs().max().item() > 0
