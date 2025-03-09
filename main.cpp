#include <iostream>
#include <vector>
#include "hebb.h"
int main() {
    // XOR dataset (not linearly separable)
    int dataset[4][3] = {{0, 0, 0},
                         {0, 1, 0},
                         {1, 0, 0},
                         {1, 1, 0}};

    int *data[4] = {dataset[0], dataset[1], dataset[2], dataset[3]};
    Perceptron p(data, 4, 3, 1.5, 1.0);

    // Train the perceptron
    for (int i = 0; i < 1000; i++) {
        p.train();
    }

    // Print final weights and bias
    p.printWeights();

    // Test predictions
    std::cout << "Predictions:" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "Input: (" << data[i][0] << ", " << data[i][1] << ") -> Prediction: " << p.predict(data[i]) << std::endl;
    }

    return 0;
}
