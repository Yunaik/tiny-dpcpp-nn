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


to_numpy = lambda a: a.detach().cpu().numpy()

torch.manual_seed(42)


def elementwise_is_close(reference, value, atol=1e-8, rtol=1e-4):

    # Perform the element-wise comparison
    for i, (a, b) in enumerate(zip(reference, value)):
        abs_diff = np.abs(a - b)
        rel_diff = abs_diff / np.maximum(np.abs(a), np.abs(b))
        if abs_diff > atol or rel_diff > rtol:
            print(f"Element {i}:")
            print(f"  Value in reference (cuda): {a}")
            print(f"  Value in value (dpcpp): {b}")
            print(f"  Absolute difference: {abs_diff} with atol {atol}")
            print(f"  Relative difference: {rel_diff} with rtol {rtol}")


def run_config(config, constant_weights, constant_input):
    # avoid padding
    n_input_dims = config["n_neurons"]
    n_output_dims = config["n_neurons"]

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

    if constant_weights:
        params = params * 0 + 0.01

    if device == "cuda":
        network.params.data[...] = params
    else:
        network.set_params(params)

    x = (
        torch.distributions.uniform.Uniform(-1, 1)
        .sample((1024, n_input_dims))
        .to(device)
    )

    if constant_input:
        x = x * 0 + 0.1
    y = network(x)
    y.backward(torch.ones_like(y))
    filename = f"output/mlp_grad_test_tensors_{config['activation']}_{config['output_activation']}_{config['n_neurons']}_{config['n_hidden_layers']}_constant_weights{constant_weights}_constant_input{constant_input}.npz"
    print(
        f"Running {filename} with config {config}, constant_weights: {constant_weights}, constant_input: {constant_input}"
    )
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
        print(cuda["y"])
        elementwise_is_close(cuda["y"].flatten(), to_numpy(y).flatten())
        np.testing.assert_array_equal(cuda["x"].flatten(), to_numpy(x).flatten())
        np.testing.assert_allclose(
            cuda["params"].sum(), to_numpy(network.params.sum()), rtol=1e-4, atol=1e-2
        )
        np.testing.assert_allclose(cuda["y"].flatten(), to_numpy(y).flatten())
        np.testing.assert_allclose(
            cuda["params_grad"].flatten(),
            to_numpy(network.params.grad).flatten(),
            rtol=1e-4,
            atol=1e-4,
        )


if __name__ == "__main__":
    configs = [
        {
            "otype": "FullyFusedMLP",
            "activation": "None",
            "output_activation": "None",
            "n_neurons": 16,
            "n_hidden_layers": 1,
            "device": device,
        },
        # {
        #     "otype": "FullyFusedMLP",
        #     "activation": "None",
        #     "output_activation": "None",
        #     "n_neurons": 16,
        #     "n_hidden_layers": 5,
        #     "device": device,
        # },
        # {
        #     "otype": "FullyFusedMLP",
        #     "activation": "ReLU",
        #     "output_activation": "None",
        #     "n_neurons": 16,
        #     "n_hidden_layers": 1,
        #     "device": device,
        # },
        # {
        #     "otype": "FullyFusedMLP",
        #     "activation": "ReLU",
        #     "output_activation": "Sigmoid",
        #     "n_neurons": 16,
        #     "n_hidden_layers": 1,
        #     "device": device,
        # },
        # {
        #     "otype": "FullyFusedMLP",
        #     "activation": "ReLU",
        #     "output_activation": "Sigmoid",
        #     "n_neurons": 64,
        #     "n_hidden_layers": 1,
        #     "device": device,
        # },
        # {
        #     "otype": "FullyFusedMLP",
        #     "activation": "ReLU",
        #     "output_activation": "Sigmoid",
        #     "n_neurons": 64,
        #     "n_hidden_layers": 3,
        #     "device": device,
        # },
    ]

    for config in configs:
        for constant_weights in [True, False]:
            for constant_input in [True, False]:
                run_config(config, constant_weights, constant_input)
