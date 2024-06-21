import torch
import torch.optim as optim
import intel_extension_for_pytorch
from tiny_dpcpp_nn_pybind_module import SimpleNN

# Need to manually set and check it aligns with pybind_module.cpp
DTYPE = torch.float16
DEVICE = "xpu"


class Module(torch.nn.Module):
    def __init__(self, device=DEVICE):
        super(Module, self).__init__()
        self.network = SimpleNN()
        self.device = device
        initial_params = self.network.get_weight()
        self.params = torch.nn.Parameter(initial_params, requires_grad=True)
        print(f"Params dtype: {self.params.dtype}")
        print(f"Params device: {self.params.device}")

    def forward(self, x):
        return self.params


def manual_mse_loss(output, target):
    """
    Computes the Mean Squared Error (MSE) loss between output and target tensors.

    Args:
    - output (torch.Tensor): Predicted output tensor from the model.
    - target (torch.Tensor): Target tensor with true values.

    Returns:
    - loss (torch.Tensor): Computed MSE loss tensor.
    """
    loss = torch.mean((output - target) ** 2)
    return loss


if __name__ == "__main__":
    # Create an instance of SimpleNN with initial weight 1.0
    net = Module()
    print(f"Net weight: {net.params.data}")
    net.params.data.copy_(torch.ones((1,), dtype=DTYPE) * 3)
    print(f"Net weight after: {net.params.data}/{net.network.get_weight()}")

    # Create an optimizer (SGD in this case)
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    # Training loop (just a few iterations for demonstration)
    for epoch in range(100):
        # Generate some dummy input and target
        input_data = torch.tensor([2.0], dtype=DTYPE, device=DEVICE)
        target = torch.tensor([4.0], dtype=DTYPE, device=DEVICE)

        # Forward pass: compute predicted y by passing x to the model
        output = net(input_data)
        assert (
            output == net.network.get_weight()
        ), "Output and underlying weight not the same"
        # Compute and print loss

        loss = manual_mse_loss(output, target)
        print(f"Epoch {epoch}: Loss = {loss.item()}")

        # Zero the gradients before running the backward pass
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

    # Print final weight after training
    print(f"Final weight: {net.params.data} (should be: {target})")
    assert torch.isclose(net.params.data, target, atol=1e-1)
