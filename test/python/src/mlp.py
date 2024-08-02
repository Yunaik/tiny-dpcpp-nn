import torch
import torch.nn.functional as F
import copy
import numpy as np

BIAS = False


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_size,
        activation_func="relu",
        output_activation=None,
        nwe_as_ref=False,  # NetworkWithEncoding (padded input as ones) is used as ref
        dtype=torch.bfloat16,
        weight_mode="constant",
        weight_val=0.1,
    ):
        super().__init__()
        self.dtype = dtype
        self.nwe_as_ref = nwe_as_ref
        self.input_width = input_size
        self.width = hidden_sizes[0]
        self.output_width = output_size
        self.layers = torch.nn.ModuleList()
        assert isinstance(activation_func, str) or None
        self.activation_func = activation_func
        self.output_activation = output_activation

        # Input layer
        input_dim = hidden_sizes[0]  # we pad in forward
        input_layer = torch.nn.Linear(
            input_dim, hidden_sizes[0], bias=BIAS, dtype=self.dtype
        )
        self._initialize_weights(input_layer, weight_mode, weight_val)
        self.layers.append(input_layer)

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            hidden_layer = torch.nn.Linear(
                hidden_sizes[i - 1], hidden_sizes[i], bias=BIAS, dtype=self.dtype
            )
            self._initialize_weights(
                hidden_layer,
                weight_mode,
                weight_val,
            )
            self.layers.append(hidden_layer)

        # Output layer
        output_layer = torch.nn.Linear(
            hidden_sizes[-1], output_size, bias=BIAS, dtype=self.dtype
        )
        self._initialize_weights(
            output_layer,
            weight_mode,
            weight_val,
        )
        self.layers.append(output_layer)

    def _initialize_weights(
        self,
        layer,
        weight_mode,
        weight_val,
    ):
        if weight_mode == "constant":
            torch.nn.init.constant_(layer.weight, weight_val)
        elif weight_mode == "linspace":
            num_elements = layer.weight.numel()
            linspace_vals = torch.linspace(
                -weight_val, weight_val, num_elements, dtype=self.dtype
            )
            layer.weight.data = linspace_vals.view_as(layer.weight)
        else:
            torch.nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))

    def forward(self, x):
        x_changed_dtype = x.to(self.dtype)
        assert x_changed_dtype.dtype == self.dtype
        batch_size = x_changed_dtype.size(0)
        if self.nwe_as_ref:
            padded_vals = torch.ones(
                (batch_size, self.layers[0].in_features - self.input_width),
                dtype=x_changed_dtype.dtype,
                device=x_changed_dtype.device,
            )  # ones, as NWE pads with 1
        else:
            padded_vals = torch.zeros(
                (batch_size, self.layers[0].in_features - self.input_width),
                dtype=x_changed_dtype.dtype,
                device=x_changed_dtype.device,
            )  # zeros, such that the bwd pass through padded vals also equals zero
        x_changed_dtype = torch.cat((x_changed_dtype, padded_vals), dim=1)

        self.activations = []  # Store activations for gradient retention
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x_changed_dtype = self._apply_activation(
                    layer(x_changed_dtype), self.output_activation
                )
            else:
                x_changed_dtype = self._apply_activation(
                    layer(x_changed_dtype), self.activation_func
                )
            x_changed_dtype.retain_grad()  # Retain gradient for this activation
            self.activations.append(x_changed_dtype)

        return x_changed_dtype

    def _apply_activation(self, x, activation_func):
        if activation_func.lower() == "relu":
            return F.relu(x)
        elif activation_func.lower() == "leaky_relu":
            return F.leaky_relu(x)
        elif activation_func.lower() == "sigmoid":
            return torch.sigmoid(x)
        elif activation_func.lower() == "tanh":
            return torch.tanh(x)
        elif (
            (activation_func.lower() == "none")
            or (activation_func is None)
            or (activation_func.lower() == "linear")
        ):
            return x
        else:
            raise ValueError(f"Invalid activation function: {activation_func}")

    def set_weights(self, parameters):
        for i, weight in enumerate(parameters):
            assert (
                self.layers[i].weight.shape == weight.shape
            ), f"In layer {i} - self layer shape: {self.layers[i].weight.shape}, passed shape: {weight.shape}"
            self.layers[i].weight = torch.nn.Parameter(weight)

    def get_all_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, "weight"):
                weight = layer.weight.data
                # Pad the first dimension if necessary
                if weight.shape[0] != self.width:
                    padding = (
                        0,
                        0,
                        0,
                        self.width - weight.shape[0],
                    )  # pad last dim
                    weight = torch.nn.functional.pad(weight, padding, "constant", 0)
                # Pad the second dimension if necessary
                elif weight.shape[1] != self.width:
                    padding = (0, self.width - weight.shape[1])  # pad last dim
                    weight = torch.nn.functional.pad(weight, padding, "constant", 0)
                # transpose as in tiny-dpcpp-nn swiftnet, the matrices are transposed
                weights.append(weight.T)
        return torch.stack(weights)

    def get_all_grads(self):
        grads = []
        for param in self.parameters():
            grad = param.grad
            if grad is not None:
                # Pad the first dimension if necessary
                if grad.shape[0] != self.width:
                    padding = (
                        0,
                        0,
                        0,
                        self.width - grad.shape[0],
                    )  # pad last dim
                    grad = torch.nn.functional.pad(grad, padding, "constant", 0)
                # Pad the second dimension if necessary
                elif grad.shape[1] != self.width:
                    padding = (0, self.width - grad.shape[1])  # pad last dim
                    grad = torch.nn.functional.pad(grad, padding, "constant", 0)
                grads.append(grad)
        return torch.stack(grads)
