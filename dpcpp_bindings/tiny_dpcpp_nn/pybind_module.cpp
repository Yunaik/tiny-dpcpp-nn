#include "ipex.h"
#include <torch/extension.h>

using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;
// A utility type trait to map C++ types to PyTorch data types
template <typename T> struct torch_type {};

template <> struct torch_type<int> {
    static const auto dtype = torch::kInt32;
};

template <> struct torch_type<double> {
    static const auto dtype = torch::kFloat64;
};

template <> struct torch_type<float> {
    static const auto dtype = torch::kFloat32;
};

template <> struct torch_type<fp16> {
    static const auto dtype = c10::ScalarType::Half;
};

template <> struct torch_type<bf16> {
    static const auto dtype = c10::ScalarType::BFloat16;
};
// Define a simple neural network class with one weight tensor
template <typename T> class SimpleNN {
  public:
    explicit SimpleNN() {
        // T initial_weight = 1.0;
        // const torch::TensorOptions &options = torch::TensorOptions().dtype(torch_type<T>::dtype).device(torch::kCPU);

        // weight = torch::from_blob(&initial_weight, {1}, options).clone();
        // Allocate memory on the SYCL queue for an array of size 1
        sycl::queue q; // Replace with your desired SYCL device selector
        T *device_ptr = static_cast<T *>(sycl::malloc_device(sizeof(T), q));

        // Check if allocation was successful
        if (!device_ptr) {
            throw std::runtime_error("Failed to allocate memory on SYCL device.");
        }
        // Initialize the memory with the provided value
        T val = 1.0;
        q.memcpy(device_ptr, &val, sizeof(T)).wait();

        weight = xpu::dpcpp::fromUSM(device_ptr, torch_type<T>::dtype, {1});
    }
    torch::Tensor get_weight() { return weight; }

  private:
    // Expose the weight tensor directly
    torch::Tensor weight;
};

// Binding code using Pybind11
PYBIND11_MODULE(tiny_dpcpp_nn_pybind_module, m) {
    py::class_<SimpleNN<fp16>>(m, "SimpleNN").def(py::init<>()).def("get_weight", &SimpleNN<fp16>::get_weight);
}
