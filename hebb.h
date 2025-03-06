#include <iostream>
#include <vector>

// Dot product function for 2D weights
int dot(std::vector<std::vector<double>> &weights, int *input, int rows, int cols) {
    int res = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            res += weights[i][j] * input[j];
        }
    }
    return res;
}

class Perceptron {
public:
    int **dataset, rows, cols, bias;
    std::vector<std::vector<double>> weights; // 2D array for weights
    double learning_rate;
    double threshold;

    Perceptron(int **_dataset, int _rows, int _columns, double lr = 1.5, double th = 1.0) {
        dataset = _dataset;
        rows = _rows;
        cols = _columns;
        learning_rate = lr;
        bias = 0;
        threshold = th;
        weights[0][0] = 1;
        weights[1][0] = 1;

        // Initialize weights as a 2D array (rows x cols-1)
        weights.resize(rows, std::vector<double>(cols - 1, 0)); // Initialize all weights to 0
    }

    void train() {
        for (int i = 0; i < rows; i++) {
            double z1 = (dataset[i][0]+(dataset[i][1]+1)%2);
            double z2 = ((dataset[i][0]+1)%2+dataset[i][1]);
            double sum = weights[i][0]*z1 + weights[i][1]*z2;
            int pred = (sum>threshold) ? 1: 0;
            int error = dataset[i][cols - 1] - pred;
            weights[0][0] += learning_rate * error * z1;
            weights[0][1] += learning_rate * error * z2;
            // Update bias
            bias += learning_rate * error;
        }
    }

    int predict(int *input) {
      // Compute z1 and z2
      double z1 = input[0] * input[1]; // z1 = x1 * x2
      double z2 = input[0] + input[1]; // z2 = x1 + x2

      // Compute the weighted sum of inputs (z1 + z2)
      double weighted_sum = weights[0][0] * z1 + weights[0][1] * z2 + bias;

      // Apply activation function (step function)
      return (weighted_sum >= threshold) ? 1 : 0;
  }

    void printWeights() {
        std::cout << "Weights:" << std::endl;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols - 1; j++) {
                std::cout << weights[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Bias: " << bias << std::endl;
    }
};