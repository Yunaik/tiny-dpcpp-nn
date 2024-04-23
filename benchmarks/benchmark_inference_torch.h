/**
 * @file benchmark_inference.h
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Implementation of a templated inference benchmark function.
 * TODO: make this a class and derived it from a base class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "common_benchmarks.h"
#include "mpi.h"
#include "tnn_api.h"
#include "torch_wrapper.h"

/// benchmarking function with input width = width = output width
template <typename T, int WIDTH>
double benchmark_inference(const size_t batch_size, const int n_hidden_layers, const int n_iterations, bool use_torch) {

    constexpr int input_width = WIDTH;
    constexpr int output_width = WIDTH;
    constexpr float weight_val = 1.0 / WIDTH;

    torch::Tensor input = torch::ones({(int)batch_size, input_width}).to(torch::kXPU).to(c10::ScalarType::BFloat16);

    // Initialize the chosen network type based on the use_torch flag
    std::unique_ptr<NeuralNetwork> network;
    if (use_torch) {
        network = std::make_unique<TorchMLPWrapper>(input_width, output_width, n_hidden_layers);
    } else {
        network =
            std::make_unique<NetworkModuleWrapper<T, WIDTH>>(input_width, output_width, n_hidden_layers, weight_val);
    }

    tinydpcppnn::benchmarks::common::WriteBenchmarkHeader("Inference", batch_size, WIDTH, n_hidden_layers, sizeof(T),
                                                          type_to_string<T>());

    constexpr int n_iterations_warmup = n_iterations / 2;
    torch::Tensor output_net;
    // Do a warmup loop, not benched.
    for (int iter = 0; iter < n_iterations_warmup; iter++) {
        output_net = network->inference(input);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto begin_time = std::chrono::steady_clock::now();
    for (int iter = 0; iter < n_iterations; iter++) {
        output_net = network->inference(input);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end_time = std::chrono::steady_clock::now();

    double gflops = tinydpcppnn::benchmarks::common::WritePerformanceDataInference(
        begin_time, end_time, batch_size, WIDTH, n_hidden_layers, n_iterations, sizeof(T));

    const float output_ref = std::pow(weight_val * WIDTH, n_hidden_layers + 1);
    bool all_values_correct = torch::allclose(output_net, torch::full_like(output_net, output_ref), 1.0e-2);
    if (all_values_correct) {
        std::cout << "All values in the tensor are correct." << std::endl;
    } else {
        std::cout << "Not all values in the tensor are correct." << std::endl;
    }
    return gflops;
}
