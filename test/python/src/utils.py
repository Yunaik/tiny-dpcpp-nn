import numpy as np

from src.mlp import MLP

from tiny_dpcpp_nn import Network, NetworkWithInputEncoding
import torch


def to_packed_layout_coord(idx, rows, cols):
    assert idx < rows * cols
    i = idx // cols
    j = idx % cols

    if i % 2 == 0:
        return i * cols + 2 * j
    else:
        return (i - 1) * cols + 2 * j + 1


def vertical_pack(matrix):
    rows, cols = matrix.shape
    packed = [0] * (rows * cols)  # Preallocate the packed array

    for idx in range(rows * cols):
        packed_idx = to_packed_layout_coord(idx, rows, cols)
        packed[packed_idx] = matrix.flat[idx]  # Use flat for 1D indexing

    return np.array(packed).reshape(rows, cols)


def vertical_unpack(packed_matrix):
    rows, cols = packed_matrix.shape
    original = np.zeros(
        (rows, cols), dtype=packed_matrix.dtype
    )  # Preallocate the original array

    for idx in range(rows * cols):
        packed_idx = to_packed_layout_coord(idx, rows, cols)
        original.flat[idx] = packed_matrix.flat[packed_idx]  # Use flat for 1D indexing

    return original


def get_reshaped_params(
    weights,
    width,
    n_hidden_layers,
    dtype,
    device,
    mode,  # reshape, pack, unpack
):

    assert (
        len(weights.shape) == 1 or weights.shape[1] == 1
    ), "Weights is assumed to be a 1-D vector"

    n_input_dims = width  # because we pad
    input_matrix = (
        weights[: width * n_input_dims]
        .reshape(width, n_input_dims)
        .to(dtype)
        .to(device)
    )

    len_input_matrix = input_matrix.shape[0] * input_matrix.shape[1]
    hidden_layer_size = width * width
    hidden_matrices = []

    for nth_hidden in range(n_hidden_layers - 1):
        hidden_matrix = (
            weights[
                len_input_matrix
                + nth_hidden * hidden_layer_size : len_input_matrix
                + (1 + nth_hidden) * hidden_layer_size
            ]
            .reshape(width, width)
            .to(dtype)
            .to(device)
        )

        hidden_matrices.append(hidden_matrix)

    n_output_dims = width  # because we pad
    output_matrix = (
        weights[-width * n_output_dims :]
        .reshape(width, n_output_dims)
        .to(dtype)
        .to(device)
    )

    all_weights = []

    all_weights.append(input_matrix)
    all_weights.extend(hidden_matrices)
    all_weights.append(output_matrix[:n_output_dims, ...])
    for layer in all_weights:
        if mode == "pack":
            layer = vertical_pack(layer)
        elif mode == "unpack":
            layer = vertical_unpack(layer)
    return all_weights

def get_grad_params(model):
    # This funciton unpacks for comparison with torch
    grads_all = []
    params_all = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            gradient = param.grad
            if len(gradient.shape) == 1 or param.data.shape[1] == 1:
                grad = model.get_reshaped_params(
                    gradient.clone(), is_packed_format=True
                )
            else:
                grad = gradient.clone()
            grads_all.append(grad)

        if len(param.data.shape) == 1 or param.data.shape[1] == 1:
            param_reshaped = model.get_reshaped_params(
                param.data.clone(), is_packed_format=True
            )
        else:
            param_reshaped = param.data.clone()

        params_all.append(param_reshaped)
    return grads_all, params_all


def compare_matrices(weights_dpcpp, weights_torch, atol=1e-1, rtol=5e-2):
    for layer, _ in enumerate(weights_dpcpp):
        assert (
            weights_dpcpp[layer].shape == weights_torch[layer].shape
        ), f"Shape different: {weights_dpcpp[layer].shape} x {weights_torch[layer].shape}"

        are_close = torch.allclose(
            weights_dpcpp[layer].to(dtype=torch.float),
            weights_torch[layer].to(dtype=torch.float),
            atol=atol,
        ) or torch.allclose(
            weights_dpcpp[layer].to(dtype=torch.float),
            weights_torch[layer].to(dtype=torch.float),
            rtol=rtol,
        )
        if not are_close:
            print(f"weights_dpcpp: {weights_dpcpp}")
            print(f"weights_torch: {weights_torch}")
            print(f"weights_dpcpp[layer] sum: {weights_dpcpp[layer].sum().sum()}")
            print(f"weights_torch[layer] sum: {weights_torch[layer].sum().sum()}")
        assert are_close


def create_models(
    input_size,
    hidden_sizes,
    output_size,
    activation_func,
    output_func,
    input_dtype,
    backend_param_dtype,
    use_nwe,
    use_weights_of_tinynn,
    constant_weight=False,
):

    # Create and test CustomMLP
    model_torch = MLP(
        input_size,
        hidden_sizes,
        output_size,
        activation_func,
        output_func,
        dtype=backend_param_dtype,
        nwe_as_ref=use_nwe,
        constant_weight=constant_weight,
    )

    network_config = {
        "activation": activation_func,
        "output_activation": output_func,
        "n_neurons": hidden_sizes[0],
        "n_hidden_layers": len(hidden_sizes),
    }

    if use_nwe:
        encoding_config = {
            "otype": "Identity",
            "n_dims_to_encode": input_size,  # assuming the input size is 2 as in other tests
            "scale": 1.0,
            "offset": 0.0,
        }

        model_dpcpp = NetworkWithInputEncoding(
            n_input_dims=input_size,
            n_output_dims=output_size,
            encoding_config=encoding_config,
            network_config=network_config,
            input_dtype=input_dtype,
            backend_param_dtype=backend_param_dtype,
        )
    else:
        model_dpcpp = Network(
            n_input_dims=input_size,
            n_output_dims=output_size,
            network_config=network_config,
            input_dtype=input_dtype,
            backend_param_dtype=backend_param_dtype,
        )

    if use_weights_of_tinynn:
        weights = model_dpcpp.get_reshaped_params()
        model_torch.set_weights(weights)
    else:
        weights = model_torch.get_all_weights()
        model_dpcpp.set_params(weights.flatten())

    model_torch.to(model_dpcpp.device)
    return model_dpcpp, model_torch
