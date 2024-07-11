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

# avoid padding
n_input_dims = 64
n_output_dims = 64
if CONSTANT_CASE:
    config = {
        "otype": "FullyFusedMLP",
        "activation": "None",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 1,
        "device": device,
    }
else:
    config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "Sigmoid",
        "n_neurons": 64,
        "n_hidden_layers": 3,
        "device": device,
    }

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


x = torch.distributions.uniform.Uniform(-1, 1).sample((1024, n_input_dims)).to(device)*0 + 0.1
y = network(x)
y.backward(torch.ones_like(y))

constant_string = "_constant" if CONSTANT_CASE else ""
if device == "cuda":
    np.savez_compressed(
        f"output/mlp_grad_test_tensors{constant_string}.npz",
        x=to_numpy(x),
        y=to_numpy(y),
        params=to_numpy(network.params),
        params_grad=to_numpy(network.params.grad),
    )
elif device == "xpu":
    cuda = np.load(f"output/mlp_grad_test_tensors{constant_string}.npz")
    np.testing.assert_array_equal(cuda["x"].flatten(), to_numpy(x).flatten())
    np.testing.assert_array_equal(cuda["params"].sum(), to_numpy(network.params.sum()))
    np.testing.assert_allclose(cuda["y"].flatten(), to_numpy(y).flatten())
    np.testing.assert_allclose(
        cuda["params_grad"].flatten(),
        to_numpy(network.params.grad).flatten(),
        rtol=1e-4,
        atol=1e-4,
    )
