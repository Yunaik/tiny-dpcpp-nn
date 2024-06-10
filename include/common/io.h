#pragma once

#include "encoding.h"
#include "json.hpp"
#include <string>
#include <unordered_map>

namespace io {

using json = nlohmann::json;
// Define the enum classes

// Map to convert string to enum for each enum class
const std::unordered_map<std::string, GridType> gridTypeMap{
    {"Hash", GridType::Hash}, {"Dense", GridType::Dense}, {"Tiled", GridType::Tiled}};

const std::unordered_map<std::string, HashType> hashTypeMap{{"Prime", HashType::Prime},
                                                            {"CoherentPrime", HashType::CoherentPrime},
                                                            {"ReversedPrime", HashType::ReversedPrime},
                                                            {"Rng", HashType::Rng}};

const std::unordered_map<std::string, InterpolationType> interpolationTypeMap{
    {"Nearest", InterpolationType::Nearest},
    {"Linear", InterpolationType::Linear},
    {"Smoothstep", InterpolationType::Smoothstep}};

const std::unordered_map<std::string, GradientMode> gradientModeMap{
    {"Ignore", GradientMode::Ignore}, {"Overwrite", GradientMode::Overwrite}, {"Accumulate", GradientMode::Accumulate}};

const std::unordered_map<std::string, ReductionType> reductionTypeMap{
    {"Concatenation", ReductionType::Concatenation}, {"Sum", ReductionType::Sum}, {"Product", ReductionType::Product}};

// Helper function to convert string to enum
template <typename T> T stringToEnum(const std::string &value, const std::unordered_map<std::string, T> &enumMap) {
    auto it = enumMap.find(value);
    if (it != enumMap.end()) {
        return it->second;
    }
    // Handle error or default case here
    // For example, you can throw an exception if the value is not found
    throw std::runtime_error("Invalid enum value");
}

json loadJsonConfig(const std::string &filename) {
    std::ifstream file{filename};
    if (!file) {
        throw std::runtime_error("Error: Unable to open file '" + filename + "'");
    }
    json config = json::parse(file, nullptr, true, /*skip_comments=*/true);

    if (config.contains(EncodingParams::GRID_TYPE)) {
        config[EncodingParams::GRID_TYPE] = stringToEnum(config[EncodingParams::GRID_TYPE], gridTypeMap);
    } else if (config.contains(EncodingParams::HASH)) {
        config[EncodingParams::HASH] = stringToEnum(config[EncodingParams::HASH], hashTypeMap);
    } else if (config.contains(EncodingParams::INTERPOLATION_METHOD)) {
        config[EncodingParams::INTERPOLATION_METHOD] =
            stringToEnum(config[EncodingParams::INTERPOLATION_METHOD], interpolationTypeMap);
    }

    return config;
}

// Function to validate and copy encoding_config with correct enums
json validateAndCopyEncodingConfig(const json &encodingConfig) {
    json encodingConfigCopy = encodingConfig; // Start by making a copy of the entire input JSON

    auto convertIfString = [&encodingConfigCopy](const auto &enumMap, const std::string &key) {
        if (encodingConfigCopy.contains(key) && encodingConfigCopy[key].is_string()) {
            // Convert string to enum
            encodingConfigCopy[key] = stringToEnum(encodingConfigCopy[key].get<std::string>(), enumMap);
        }
    };

    // Convert string values to enums if they are strings
    convertIfString(gridTypeMap, EncodingParams::GRID_TYPE);
    convertIfString(hashTypeMap, EncodingParams::HASH);
    convertIfString(interpolationTypeMap, EncodingParams::INTERPOLATION_METHOD);

    return encodingConfigCopy;
}

template <typename T>
std::vector<T> get_packed_weights(std::vector<T> unpacked_weights, int m_n_hidden_layers, int input_width,
                                  int network_width, int output_width) {
    std::vector<T> weights_packed(unpacked_weights.size(), 0.0);

    for (int idx = 0; idx < weights_packed.size(); idx++) {

        int i = 0;
        int j = 0;
        if (idx < input_width * network_width) {

            i = idx / network_width; // rows
            j = idx % network_width; // cols

            weights_packed[toPackedLayoutCoord(i + j * network_width, network_width, network_width)] =
                unpacked_weights[idx];
        } else if ((idx >= input_width * network_width) &&
                   (idx < input_width * network_width + (m_n_hidden_layers - 1) * network_width * network_width)) {
            int layer = (idx - input_width * network_width) / (network_width * network_width);
            int mat_offset = (idx - (input_width * network_width + layer * network_width * network_width)) %
                             (network_width * network_width);

            i = mat_offset / network_width; // rows
            j = mat_offset % network_width; // cols

            weights_packed[input_width * network_width + layer * network_width * network_width +
                           toPackedLayoutCoord(i + j * network_width, network_width, network_width)] =
                unpacked_weights[idx];
        } else {
            int mat_offset =
                (idx - input_width * network_width - (m_n_hidden_layers - 1) * network_width * network_width) %
                (network_width * output_width);
            i = mat_offset / network_width; // rows
            j = mat_offset % network_width; // cols

            weights_packed[input_width * network_width + (m_n_hidden_layers - 1) * network_width * network_width +
                           toPackedLayoutCoord(i + j * network_width, network_width, network_width)] =
                unpacked_weights[idx];
        }
    }

    return weights_packed;
}

template <typename T, int WIDTH>
std::vector<T> load_weights_as_packed_from_file(std::string filename, int m_n_hidden_layers, int input_width,
                                                int output_width) {
    // Read each value from the file and set it as a bf16 value in weights matrices
    std::vector<T> data_vec;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
    }

    std::string line;
    while (std::getline(file, line)) {
        try {
            float value = std::stod(line);
            data_vec.push_back((T)(value));
        } catch (const std::invalid_argument &e) {
            std::cerr << "Invalid argument: " << e.what() << std::endl;
        } catch (const std::out_of_range &e) {
            std::cerr << "Out of range: " << e.what() << std::endl;
        }
    }

    file.close();

    return get_packed_weights<T>(data_vec, m_n_hidden_layers, input_width, WIDTH, output_width);
}

template <typename T> std::vector<T> loadVectorFromCSV(const std::string &filename) {
    std::vector<T> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::invalid_argument("Failed to open the file for reading: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            data.push_back(static_cast<T>(std::stof(token)));
        }
    }

    return data;
}

template <typename T> void saveCSV(const std::string &filename, const std::vector<T> &data) {
    std::ofstream file(filename);

    if (file.is_open()) {
        for (const auto &value : data) {
            file << (double)value << std::endl;
        }
        file.close();
    }
}

std::vector<float> loadCSV(const std::string &filename) {
    std::vector<float> data;
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            float value;
            std::istringstream iss(line);
            iss >> value;
            data.push_back(value);
        }
        file.close();
    }
    return data;
}

// Function to read target vectors from a file with a specified delimiter
std::vector<std::vector<float>> readTargetVectorsFromFile(const std::string &filename, char delimiter) {
    std::vector<std::vector<float>> targetVectors;

    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening " << filename << std::endl;
        return targetVectors; // Return an empty vector in case of an error
    }

    std::string line;

    while (std::getline(inputFile, line)) {
        std::vector<float> vectorFromCSV;
        std::istringstream lineStream(line);
        std::string valueStr;

        while (std::getline(lineStream, valueStr, delimiter)) {
            float value = std::stod(valueStr); // Convert the string to a double
            vectorFromCSV.push_back(value);
        }

        targetVectors.push_back(vectorFromCSV);
    }

    inputFile.close();

    return targetVectors;
}

void saveImageToPGM(const std::string &filename, const int width, const int height,
                    const std::vector<unsigned char> &image) {
    // Create and open the output file
    std::ofstream outputFile(filename, std::ios::out | std::ios::binary);

    // Write PGM header
    outputFile << "P5\n" << width << " " << height << "\n255\n";

    // Write the image data to the file
    outputFile.write(reinterpret_cast<const char *>(image.data()), image.size());

    // Close the file
    outputFile.close();
}
} // namespace io
