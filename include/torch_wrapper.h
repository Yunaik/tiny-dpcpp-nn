
#include <torch/torch.h>

// MLP class definition
struct MLP : torch::nn::Module {
    MLP(int input_width, int n_hidden_layers, int output_width) {
        for (int i = 0; i < n_hidden_layers; i++) {
            // Using `push_back` to add layers to the module
            layers->push_back(torch::nn::Linear(input_width, output_width));
            layers->push_back(torch::nn::Functional(torch::relu));
        }
        layers->push_back(torch::nn::Linear(input_width, output_width));
        register_module("layers", layers);
    }

    torch::Tensor forward(const torch::Tensor &input) { return layers->forward(input); }

    torch::nn::Sequential layers;
};

class NeuralNetwork {
  public:
    virtual torch::Tensor inference(const torch::Tensor &input) = 0; // Pure virtual function
    virtual ~NeuralNetwork() = default;                              // Virtual destructor for proper cleanup
};

template <typename T, int WIDTH> class NetworkModuleWrapper : public NeuralNetwork {
  private:
    std::unique_ptr<tnn::NetworkModule<T, WIDTH>> network;

  public:
    NetworkModuleWrapper(int input_width, int output_width, int n_hidden_layers, float weight_val)
        : network(std::make_unique<tnn::NetworkModule<T, WIDTH>>(input_width, output_width, n_hidden_layers,
                                                                 Activation::ReLU, Activation::None)) {
        torch::Tensor init_params =
            torch::ones({(int)network->n_params(), 1}).to(torch::kXPU).to(c10::ScalarType::BFloat16) * weight_val;
        torch::Tensor params = network->initialize_params(init_params);
    }

    torch::Tensor inference(const torch::Tensor &input) override { return network->inference(input); }
};

class TorchMLPWrapper : public NeuralNetwork {
  private:
    MLP network;

  public:
    TorchMLPWrapper(int input_width, int output_width, int n_hidden_layers, std::string device = "xpu")
        : network(input_width, n_hidden_layers, output_width) {
        network.to(c10::ScalarType::BFloat16);
        if (device == "cpu") {
            network.to(torch::kCPU);
        } else {
            network.to(torch::kXPU);
        }
    }

    torch::Tensor inference(const torch::Tensor &input) override {
        if (network.is_training()) {
            network.eval();
            std::cout << "Setting to eval mode" << std::endl;
        }
        return network.forward(input);
    }
};