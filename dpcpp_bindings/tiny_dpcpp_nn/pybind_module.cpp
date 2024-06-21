#include "ipex.h"
#include <torch/extension.h>

using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;
// A utility type trait to map C++ types to PyTorch data types
template <typename T> struct torch_type {};

template <> struct torch_type<int> {
    static const auto dtype = torch::kInt32;
    typedef int type;
};

template <> struct torch_type<double> {
    static const auto dtype = torch::kFloat64;
    typedef double type;
};

template <> struct torch_type<float> {
    static const auto dtype = torch::kFloat32;
    typedef float type;
};

template <> struct torch_type<fp16> {
    static const auto dtype = c10::ScalarType::Half;
    typedef at::Half type;
};

template <> struct torch_type<bf16> {
    static const auto dtype = c10::ScalarType::BFloat16;
    typedef at::BFloat16 type;
};
// Define a simple neural network class with one weight tensor
class SimpleNN {
  public:
    explicit SimpleNN() {
        float initial_weight = 1.0;
        const torch::TensorOptions &options =
            torch::TensorOptions().dtype(torch_type<float>::dtype).device(torch::kCPU);

        weight = torch::from_blob(&initial_weight, {1}, torch::kFloat32).clone();
    }
    torch::Tensor get_weight() { return weight; }

  private:
    // Expose the weight tensor directly
    torch::Tensor weight;
};

// Binding code using Pybind11
PYBIND11_MODULE(tiny_dpcpp_nn_pybind_module, m) {
    py::class_<SimpleNN>(m, "SimpleNN").def(py::init<>()).def("get_weight", &SimpleNN::get_weight);
}
