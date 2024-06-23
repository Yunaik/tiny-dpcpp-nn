import torch
import pytest
import intel_extension_for_pytorch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from src.utils import create_models

torch.set_printoptions(precision=10)

optimisers = ["adam", "sgd"]
dtypes = [torch.bfloat16]
# dtypes = [torch.float16, torch.bfloat16]

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


@pytest.mark.parametrize(
    "dtype, optimiser",
    [(dtype, optimiser) for dtype in dtypes for optimiser in optimisers],
)
def test_regression(dtype, optimiser):
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
        input_dtype=torch.float if USE_NWE else dtype,
        backend_param_dtype=dtype,
        use_weights_of_tinynn=True,
    )

    def criterion(y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()

    if optimiser == "adam":
        optimizer1 = torch.optim.Adam(model_dpcpp.parameters(), lr=LR)
        optimizer2 = torch.optim.Adam(model_torch.parameters(), lr=LR)
    elif optimiser == "sgd":
        optimizer1 = SimpleSGDOptimizer(model_dpcpp.parameters(), name="dpcpp", lr=LR)
        optimizer2 = SimpleSGDOptimizer(model_torch.parameters(), name="torch", lr=LR)
    else:
        raise NotImplementedError(f"{optimiser} not implemented as optimisers")

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            # DPCPP
            inputs1 = inputs.clone().to(DEVICE).to(dtype)
            labels1 = labels.clone().to(DEVICE).to(dtype)
            # Forward pass
            outputs1 = model_dpcpp(inputs1)
            loss1 = criterion(outputs1, labels1)
            optimizer1.zero_grad()
            loss1.backward()
            params1 = model_dpcpp.params.clone().detach()

            grads1 = model_dpcpp.params.grad.clone().detach()
            optimizer1.step()
            params_updated1 = model_dpcpp.params.clone().detach()
            assert not torch.equal(
                params1, params_updated1
            ), "The params for model_dpcpp are the same after update, but they should be different."

            # Torch
            inputs2 = inputs.clone().to(DEVICE).to(dtype)
            labels2 = labels.clone().to(DEVICE).to(dtype)
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

            # Assertions
            assert (
                params1.dtype == params2.dtype
            ), f"Params not same dtype: {params1.dtype}, {params2.dtype}"
            assert torch.isclose(
                inputs1, inputs2
            ).all(), f"Inputs not close with sums: {abs(inputs1).sum()}, {abs(inputs2).sum()}"
            assert torch.isclose(
                outputs1, outputs2
            ).all(), f"Outputs not close with sums: {abs(outputs1).sum()}, {abs(outputs2).sum()}"
            assert torch.isclose(
                labels1, labels2
            ).all(), f"Labels not close with sums: {abs(labels1).sum()}, {abs(labels2).sum()}"
            assert torch.isclose(
                loss1, loss2
            ).all(), f"Loss not close with sums: {loss1.item()}, {loss2.item()}"
            assert torch.isclose(
                abs(params1).sum(), abs(params2).sum()
            ), f"Params before not close with sums: {abs(params1).sum()}, {abs(params2).sum()}"

            assert torch.isclose(
                abs(grads1).sum(), abs(grads2).sum()
            ), f"Grads not close with sums: {abs(grads1).sum()}, {abs(grads2).sum()}"

            assert torch.isclose(
                abs(params_updated1).sum(), abs(params_updated2).sum()
            ), f"Params after not close with sums: {abs(params_updated1).sum()}, {abs(params_updated2).sum()}"

        print(f"Epoch {epoch}, Losses (dpcpp/torch): { loss1.item()}/{ loss2.item()}")
        print(
            "========================================================================"
        )

    print("Finished Training")


if __name__ == "__main__":
    dtype = torch.bfloat16
    optimiser = "sgd"
    test_regression(dtype, optimiser)
