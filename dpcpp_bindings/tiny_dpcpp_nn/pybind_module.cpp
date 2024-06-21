#include <torch/extension.h>

// Define a simple neural network class with one weight tensor
class SimpleNN {
  public:
    explicit SimpleNN() {
        float initial_weight = 1.0;
        weight = torch::from_blob(&initial_weight, {1}, torch::kFloat32).clone();
    }

    // Expose the weight tensor directly
    torch::Tensor weight;
};

// Binding code using Pybind11
PYBIND11_MODULE(tiny_dpcpp_nn_pybind_module, m) {
    py::class_<SimpleNN>(m, "SimpleNN").def(py::init<>()).def_readwrite("weight", &SimpleNN::weight);
}
