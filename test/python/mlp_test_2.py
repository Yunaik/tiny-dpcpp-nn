import numpy as np
import torch

if torch.cuda.is_available():
    import tinycudann as tcnn

    device = "cuda"
else:
    import tiny_dpcpp_nn as tcnn

    device = "xpu"

to_numpy = lambda a: a.detach().cpu().numpy()

torch.manual_seed(42)

# avoid padding
n_input_dims = 64
n_output_dims = 64
x = (torch.ones(1024, 64) * 0.1).to(
    device, dtype=torch.float16
)  # torch.distributions.uniform.Uniform(-0.01,0.01).sample((1024,n_input_dims)).to(device)


# config = {'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'Sigmoid', 'n_neurons': 64, 'n_hidden_layers': 3, 'device': device}
config = {
    "otype": "FullyFusedMLP",
    "activation": "None",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 3,
    "device": device,
}

DTYPE = torch.bfloat16
# DTYPE = torch.float16
networkbf16 = tcnn.Network(
    n_input_dims, n_output_dims, config, input_dtype=DTYPE, backend_param_dtype=DTYPE
)

# tiny-cuda-nn seems to use always float32 for params
assert networkbf16.params.dtype == torch.float32


# make sure we have the same initialization
# networkbf16.params.data[...] = torch.ones(64*64*4,1).to(device)#((networkbf16.params.bool().float())*0.1).to(device)#torch.distributions.uniform.Uniform(-0.216,0.216).sample(networkbf16.params.shape).to(device)
# print(f"Params: {networkbf16.params[0:10, 0]}")
networkbf16.set_params(torch.ones(64 * 64 * 4, 1).to(device))

torch.xpu.synchronize()
print("Input")
print(x[0:64, 0])
ybf16 = networkbf16(x)
torch.xpu.synchronize()
print("Output")
print(ybf16[0:10, 0])
# np.testing.assert_allclose(to_numpy(ybf16), 0.5)
# ybf16.backward(torch.ones_like(ybf16))

# redo with fp16
DTYPE = torch.float16
# config = {'otype': 'FullyFusedMLP', 'activation': 'ReLU', 'output_activation': 'Sigmoid', 'n_neurons': 64, 'n_hidden_layers': 3, 'device': device}
network = tcnn.Network(
    n_input_dims, n_output_dims, config, input_dtype=DTYPE, backend_param_dtype=DTYPE
)
network.params.data[...] = torch.ones(64 * 64 * 4, 1).to(
    device, dtype=torch.float32
)  # ((networkbf16.params.bool().float())*0.1).to(device)#torch.distributions.uniform.Uniform(-0.216,0.216).sample(network.params.shape).to(device)
networkbf16.set_params(torch.ones(64 * 64 * 4, 1, dtype=torch.float32).to(device))

assert network.params.dtype == torch.float32
print("Are params equal?", (network.params - networkbf16.params).abs().max().item())

# x = torch.distributions.uniform.Uniform(-0.01,0.01).sample((1024,44)).to(device)
print(x[0:64, 0])
y = network(x)
torch.xpu.synchronize()
print("(y-ybf16).abs().max().item()", (y - ybf16).abs().max().item())
print("y.abs().max().item()", y.abs().max().item())
print("ybf16.abs().max().item()", ybf16.abs().max().item())
print(ybf16[0:10, 0])
print(y[0:10, 0])
y.backward(torch.ones_like(y))

torch.xpu.synchronize()
print(networkbf16.params.grad)
print(network.params.grad)
print(
    "(network.params.grad-networkbf16.params.grad).abs().max().item()",
    (network.params.grad - networkbf16.params.grad).abs().max().item(),
)
print("network.params.grad.abs().max().item()", network.params.grad.abs().max().item())
print(
    "networkbf16.params.grad.abs().max().item()",
    networkbf16.params.grad.abs().max().item(),
)

# gradient is all 0 with tiny-dpcpp-nn but not with cuda
# assert network.params.grad.abs().max().item() > 0
