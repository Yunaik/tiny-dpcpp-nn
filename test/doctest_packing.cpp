/**
 * @file doctest_swiftnet.cpp
 * @author Christoph Bauinger (christoph.bauinger@intel.com)
 * @brief Tests for the Swiftnet class.
 * @version 0.1
 * @date 2024-01-19
 *
 * Copyright (c) 2024 Intel Corporation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "doctest/doctest.h"

#include "common.h"

TEST_CASE("Testing get_packed_weights") {

    // Reference values for comparison
    std::vector<float> reference_values = {
        1,   17,  2,   18,  3,   19,  4,   20,  5,   21,  6,   22,  7,   23,  8,   24,  9,   25,  10,  26,  11,  27,
        12,  28,  13,  29,  14,  30,  15,  31,  16,  32,  33,  49,  34,  50,  35,  51,  36,  52,  37,  53,  38,  54,
        39,  55,  40,  56,  41,  57,  42,  58,  43,  59,  44,  60,  45,  61,  46,  62,  47,  63,  48,  64,  65,  81,
        66,  82,  67,  83,  68,  84,  69,  85,  70,  86,  71,  87,  72,  88,  73,  89,  74,  90,  75,  91,  76,  92,
        77,  93,  78,  94,  79,  95,  80,  96,  97,  113, 98,  114, 99,  115, 100, 116, 101, 117, 102, 118, 103, 119,
        104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127, 112, 128, 129, 145, 130, 146,
        131, 147, 132, 148, 133, 149, 134, 150, 135, 151, 136, 152, 137, 153, 138, 154, 139, 155, 140, 156, 141, 157,
        142, 158, 143, 159, 144, 160, 161, 177, 162, 178, 163, 179, 164, 180, 165, 181, 166, 182, 167, 183, 168, 184,
        169, 185, 170, 186, 171, 187, 172, 188, 173, 189, 174, 190, 175, 191, 176, 192, 193, 209, 194, 210, 195, 211,
        196, 212, 197, 213, 198, 214, 199, 215, 200, 216, 201, 217, 202, 218, 203, 219, 204, 220, 205, 221, 206, 222,
        207, 223, 208, 224, 225, 241, 226, 242, 227, 243, 228, 244, 229, 245, 230, 246, 231, 247, 232, 248, 233, 249,
        234, 250, 235, 251, 236, 252, 237, 253, 238, 254, 239, 255, 240, 256,
    };

    int input_width = 16;
    int network_width = 16;
    int output_width = 16;
    int m_n_hidden_layers = 2; // Adjust as needed for testing

    // Create a vector with values from 1 to 16 * (16 + 16) for testing
    std::vector<float> unpacked_weights;
    for (int i = 1; i <= input_width * network_width + (m_n_hidden_layers - 1) * network_width * network_width +
                             network_width * output_width;
         ++i) {
        unpacked_weights.push_back(static_cast<float>(i));
    }

    auto packed_weights =
        get_packed_weights(unpacked_weights, m_n_hidden_layers, input_width, network_width, output_width);

    // Check the size of the packed weights
    CHECK(packed_weights.size() == reference_values.size());

    // Check the shape (this part may depend on your shape representation)
    CHECK(packed_weights.size() % network_width == 0); // Should be multiple of network_width

    // Check packed values against reference
    for (size_t i = 0; i < packed_weights.size(); ++i) {
        CHECK(packed_weights[i] == reference_values[i]);
    }
}