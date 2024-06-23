import torch
import intel_extension_for_pytorch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from src.utils import create_models

torch.set_printoptions(precision=10)


# USE_ADAM = True
USE_ADAM = False

DTYPE = torch.float16
USE_NWE = False
WIDTH = 16
num_epochs = 100
DEVICE = "xpu"

LR = 1e-2


class SimpleSGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, name="", lr=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SimpleSGDOptimizer, self).__init__(params, defaults)
        self.name = name

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    print("p.grad is none")
                    continue
                grad = p.grad.data

                p.data.copy_(p.data - group["lr"] * grad)

        return loss


# Define a simple linear function for the dataset
def true_function(x):
    return 0.5 * x


def test_regression():
    # Create a synthetic dataset based on the true function
    input_size = WIDTH
    output_size = 1
    num_samples = 2**3
    batch_size = 2**3

    # inputs
    # inputs_single = torch.linspace(-1, 1, steps=num_samples)
    inputs_single = torch.ones(num_samples)
    inputs_training = inputs_single.repeat(input_size, 1).T

    # Corresponding labels with some noise
    noise = torch.randn(num_samples, output_size) * 0.0

    labels_training = true_function(inputs_training) + noise
    # Create a DataLoader instance for batch processing
    dataset = TensorDataset(inputs_training, labels_training)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )  # if we shuffle, the loss is not identical

    # Instantiate the network
    model_dpcpp, model_torch = create_models(
        input_size,
        [WIDTH],
        output_size,
        "relu",
        "linear",
        use_nwe=USE_NWE,
        input_dtype=torch.float if USE_NWE else DTYPE,
        backend_param_dtype=DTYPE,
        use_weights_of_tinynn=True,
    )

    def criterion(y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()

    if USE_ADAM:
        optimizer1 = torch.optim.Adam(model_dpcpp.parameters(), lr=LR)
        optimizer2 = torch.optim.Adam(model_torch.parameters(), lr=LR)
    else:
        optimizer1 = SimpleSGDOptimizer(model_dpcpp.parameters(), name="dpcpp", lr=LR)
        optimizer2 = SimpleSGDOptimizer(model_torch.parameters(), name="torch", lr=LR)

    # Training loop
    params_updated1 = None
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            # DPCPP
            inputs1 = inputs.clone().to(DEVICE).to(DTYPE)
            labels1 = labels.clone().to(DEVICE).to(DTYPE)
            # Forward pass
            outputs1 = model_dpcpp(inputs1)
            loss1 = criterion(outputs1, labels1)
            optimizer1.zero_grad()
            loss1.backward()
            params1 = model_dpcpp.params.clone().detach()
            if params_updated1 is not None:
                assert not torch.equal(
                    params1, params_updated1
                ), "The params for model_dpcpp are the same after update, but they should be different."

            grads1 = model_dpcpp.params.grad.clone().detach()
            optimizer1.step()
            params_updated1 = model_dpcpp.params.clone().detach()
            assert not torch.equal(
                params1, params_updated1
            ), "The params for model_dpcpp are the same after update, but they should be different."

            # Torch
            inputs2 = inputs.clone().to(DEVICE).to(DTYPE)
            labels2 = labels.clone().to(DEVICE).to(DTYPE)
            outputs2 = model_torch(inputs2)
            loss2 = criterion(outputs2, labels2)
            optimizer2.zero_grad()
            loss2.backward()
            params2 = model_torch.get_all_weights()
            grads2 = model_torch.get_all_grads()
            optimizer2.step()
            params_updated2 = model_torch.get_all_weights()

            assert not torch.equal(
                params2, params_updated2
            ), "The params for model_dpcpp are the same after update, but they should be different."

        print(f"Epoch {epoch}, Losses (dpcpp/torch): { loss1.item()}/{ loss2.item()}")
        print(
            "========================================================================"
        )

    print("Finished Training")


if __name__ == "__main__":
    test_regression()
