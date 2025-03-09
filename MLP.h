#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include "funs.h"
class MLP {
   public:
    MLP(int inS, int hS, int oS, double l) {
        inSize = inS;
        hidSize = hS;
        outSize = oS;
        lr = l;
        initalize();
    }
    std::vector<double> forward(std::vector<double> &input) {  // forward pass
        hidden.resize(hidSize, 0.0);  // input to hidden layer
        for (int i = 0; i < hidSize; ++i) {
            double sum = 0.0;
            for (int j = 0; j < inSize; ++j) {
                sum+= input[j]*hWeightIN[j][i];
            }
            sum+=hBias[i];
            hidden[i] = sigmoid(sum);
        }
        std::vector<double> output(outSize, 0.0);
        for (int i = 0; i < outSize; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hidSize; ++j) {
                sum+=(hWeightOUT[j][i] * hidden[j]);
            }
            sum+=biasOut[i];
            output[i] = sigmoid(sum);
        }
        return output;
    }
    void backward(std::vector<double>& input, std::vector<double> &expected, std::vector<double> &predicted) {
        std::vector<double> outGrad(outSize, 0.0); //output layer gradient
        for (int i = 0; i < outSize; ++i) {
            outGrad[i] = (predicted[i] - expected[i]) * dsigmoid(predicted[i]);
        }
        std::vector<double> hidGrad(hidSize, 0.0); //hidden layer gradient
        for (int i = 0; i < hidSize; ++i) {
            double sum = 0.0;
            for (int j = 0; j < outSize; ++j) {
                sum += outGrad[j] * hWeightOUT[i][j];
            }
            hidGrad[i] = sum * dsigmoid(hidden[i]);
        }
        for (int i = 0; i < outSize; ++i) {
            for (int j = 0; j < hidSize; ++j) {
                hWeightOUT[j][i] -= lr * outGrad[i] * hidden[j];
            }
            biasOut[i] -= lr * outGrad[i];
        }
        for (int i = 0; i < hidSize; ++i) {
            for (int j = 0; j < inSize; ++j) {
                hWeightIN[j][i] -= lr * hidGrad[i] * input[j];
            }
            hBias[i] -= lr * hidGrad[i];
        }
    }

    void train(std::vector<std::vector<double>> &inputs) {
        for (int i = 0; i < inputs.size(); ++i) {
            std::vector<double> input(inSize, 0.0);
            std::vector<double> target(outSize, 0.0);
            double totalLoss = 0;
            for (int j = 0; j < inputs[i].size() - 1; ++j) {
                input[j] = inputs[i][j];
            }
            target[0] = inputs[i][inSize];
            std::vector<double> output = forward(input);
            backward(input, target, output);
            // std::cout << mse(output, target) << '\n';

        }

    }
    int predict(std::vector<double> &input) {
        std::vector<double> output = forward(input);
    
        // Debug: Print output values
        std::cout << "Output: ";
        for (double o : output) std::cout << o << " ";
        std::cout << std::endl;
    
        if (output[0] >= 0.5) {
            return 1;
        } else {
            return 0;
        }
    }

   public:
    int inSize, hidSize, outSize;
    double lr;
    std::vector<double> hidden;
    std::vector<std::vector<double>> hWeightIN;
    std::vector<std::vector<double>> hWeightOUT;
    std::vector<double> hBias;
    std::vector<double> biasOut;
    void initalize() {  // initalizes weights and biases
        srand(time(0));
        hWeightIN.resize(inSize, std::vector<double>(hidSize, 0.0));
        for (int i = 0; i < inSize; ++i) {
            for (int j = 0; j < hidSize; ++j) {
                hWeightIN[i][j] = ((double)rand() / RAND_MAX) - 1;
            }
        }
        hWeightOUT.resize(hidSize, std::vector<double>(outSize, 0.0));
        for (int i = 0; i < hidSize; ++i) {
            for (int j = 0; j < outSize; ++j) {
                hWeightOUT[i][j] = ((double)rand() / RAND_MAX) - 1;
            }
        }
        hBias.resize(hidSize, 0.0);
        biasOut.resize(outSize, 0.0);
    }
};