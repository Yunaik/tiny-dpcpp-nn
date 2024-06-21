import torch
import torch.optim as optim
import intel_extension_for_pytorch
from tiny_dpcpp_nn_pybind_module import SimpleNN


class Module(torch.nn.Module):
    def __init__(self, device="cpu"):
        super(Module, self).__init__()
        self.network = SimpleNN()
        self.device = device
        initial_params = self.network.get_weight()
        # Creating the torch.nn.Parameter object with the initialized tensor
        # self.params = torch.nn.Parameter(
        #     initial_params.detach().clone().to(device), requires_grad=True
        # )
        self.params = torch.nn.Parameter(initial_params.to(device), requires_grad=True)

    def forward(self, x):
        return self.params


if __name__ == "__main__":
    # Create an instance of SimpleNN with initial weight 1.0
    net = Module()
    print(f"Net weight: {net.params.data}")

    # Define a simple loss function (mean squared error)
    loss_fn = torch.nn.MSELoss()

    # Create an optimizer (SGD in this case)
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    # Training loop (just a few iterations for demonstration)
    for epoch in range(30):
        # Generate some dummy input and target
        input_data = torch.tensor([2.0], dtype=torch.float32)
        target = torch.tensor([4.0], dtype=torch.float32)

        # Forward pass: compute predicted y by passing x to the model
        output = net(input_data)

        # Compute and print loss
        loss = loss_fn(output, target)
        print(f"Epoch {epoch}: Loss = {loss.item()}")

        # Zero the gradients before running the backward pass
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

    # Print final weight after training
    print(f"Final weight: {net.params.data} (should be: {target})")
