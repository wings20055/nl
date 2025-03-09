#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "MLP.h"
int main() {
    // Seed random number
    srand(time(0));

    MLP mlp(2, 2, 1, 0.1);

    // XOR dataset
    std::vector<std::vector<double>> xor_data = { //change these parameters based on whatever function you need
        {0, 0, 1},  
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 0}};
    std::cout << "starting" << '\n';
    // Train the MLP
    for (int epoch = 0; epoch < 10000; ++epoch) {
        mlp.train(xor_data);
    }
    std::cout << "done" << '\n';
    // Test the MLP
    for (const auto &data : xor_data) {
        std::vector<double> input = {data[0], data[1]};
        int prediction = mlp.predict(input);
        std::cout << "Input: (" << input[0] << ", " << input[1]
                  << ") -> Prediction: " << prediction << std::endl;
    }

    return 0;
}