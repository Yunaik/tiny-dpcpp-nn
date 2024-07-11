import numpy as np
import torch

if torch.cuda.is_available():
    import tinycudann as tcnn

    device = "cuda"
else:
    import tiny_dpcpp_nn as tcnn

    device = "xpu"
torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)

CONSTANT_CASE = True

to_numpy = lambda a: a.detach().cpu().numpy()

torch.manual_seed(42)


def elementwise_is_close(reference, value, atol=1e-8, rtol=1e-4):

    # Perform the element-wise comparison
    for i, (a, b) in enumerate(zip(reference, value)):
        abs_diff = np.abs(a - b)
        rel_diff = abs_diff / np.maximum(np.abs(a), np.abs(b))
        print(f"Element {i}:")
        print(f"  Value in cuda['y']: {a}")
        print(f"  Value in y: {b}")
        print(f"  Absolute difference: {abs_diff}")
        print(f"  Relative difference: {rel_diff}")
        print(f"  atol: {atol}")
        print(f"  rtol: {rtol}")
        print(f"  Within tolerance: {abs_diff <= atol + rtol * np.abs(b)}\n")


# avoid padding
n_input_dims = 64
n_output_dims = 64
config = {
    "otype": "FullyFusedMLP",
    "activation": "None",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 1,
    "device": device,
}
# config = {
#         "otype": "FullyFusedMLP",
#         "activation": "ReLU",
#         "output_activation": "Sigmoid",
#         "n_neurons": 64,
#         "n_hidden_layers": 3,
#         "device": device,
#     }

network = tcnn.Network(n_input_dims, n_output_dims, config)

# tiny-cuda-nn seems to use always float32 for params
assert network.params.dtype == torch.float32

# make sure we have the same initialization
torch.manual_seed(42)
params = (
    torch.distributions.uniform.Uniform(-0.216, 0.216)
    .sample(network.params.shape)
    .to(device)
)

if CONSTANT_CASE:
    params = params * 0 + 0.01

if device == "cuda":
    network.params.data[...] = params
else:
    network.set_params(params)


x = torch.distributions.uniform.Uniform(-1, 1).sample((1024, n_input_dims)).to(device)

if CONSTANT_CASE:
    x = x * 0 + 0.1
y = network(x)
y.backward(torch.ones_like(y))
filename = f"output/mlp_grad_test_tensors_{config['activation']}_{config['output_activation']}_{config['n_neurons']}_{config['n_hidden_layers']}_constant{CONSTANT_CASE}.npz"

if device == "cuda":
    np.savez_compressed(
        filename,
        x=to_numpy(x),
        y=to_numpy(y),
        params=to_numpy(network.params),
        params_grad=to_numpy(network.params.grad),
    )
elif device == "xpu":
    cuda = np.load(filename)
    elementwise_is_close(cuda["y"].flatten(), to_numpy(y).flatten())
    np.testing.assert_array_equal(cuda["x"].flatten(), to_numpy(x).flatten())
    np.testing.assert_array_equal(cuda["params"].sum(), to_numpy(network.params.sum()))
    np.testing.assert_allclose(cuda["y"].flatten(), to_numpy(y).flatten())
    np.testing.assert_allclose(
        cuda["params_grad"].flatten(),
        to_numpy(network.params.grad).flatten(),
        rtol=1e-4,
        atol=1e-4,
    )
