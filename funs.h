#include <vector>
#include <iostream>
#include <cmath>
int dot(std::vector<std::vector<double>> &weights, int *input, int rows, int cols) {
    int res = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols - 1; j++) {
            res += weights[i][j] * input[j];
        }
    }
    return res;
}
double sigmoid(double x) {
    return 1.0/(1.0 + exp(x));
}
