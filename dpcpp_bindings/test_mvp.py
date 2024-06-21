import torch
import torch.optim as optim
from tiny_dpcpp_nn_pybind_module import SimpleNN

# Create an instance of SimpleNN with initial weight 1.0
net = SimpleNN(1.0)
print(f"Net weight: {net.weight}")

# Define a simple loss function (mean squared error)
loss_fn = torch.nn.MSELoss()

# Create an optimizer (SGD in this case)
optimizer = optim.SGD([net.weight], lr=0.1)

# Training loop (just a few iterations for demonstration)
for epoch in range(10):
    # Generate some dummy input and target
    input_data = torch.tensor([2.0], dtype=torch.float32)
    target = torch.tensor([4.0], dtype=torch.float32)

    # Forward pass: compute predicted y by passing x to the model
    output = net.weight

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
print(f"Final weight: {net.weight}")
