#include "funs.h"
#include <iostream>
#include <vector>
class MLP {
public: 
    MLP (int inS, int hS, int oS, double l) {
        inSize = inS;
        hidSize = hS;
        outSize = oS;
        lr = l;
        initalize();
    }
    std::vector<double> forward(std::vector<double>& input) { //forward pass
        std::vector<double> hidden(hidSize, 0.0);
        for(int i = 0;i<hidSize;++i) {
            for(int j = 0;j<inSize;++j) {
                hidden[i] = input[j] * hWeightIN[j][i];
            }
            hidden[i] += hBias[i];
            hidden[i] += sigmoid(hidden[i]);
        }
        std::vector<double> output(outSize, 0.0);
        for(int i = 0;i<outSize;++i) {
            for(int j = 0;j<hidSize;++j) {
                output[i] = hidden[j] * hWeightOUT[j][i];
            }
            output[i] += biasOut[i];
            output[i] += sigmoid(output[i]);
        }
        return output;
    }
    void train() {

    }
    int predict(std::vector<double>& input) {
        std::vector<double> output = forward(input);
        int pred = 0;
        double max = output[0];
        for(int i = 0;)

    }

public:
    int inSize, hidSize, outSize;
    double lr;
    std::vector<std::vector<double>> hWeightIN;
    std::vector<std::vector<double>> hWeightOUT;
    std::vector<double> hBias;
    std::vector<double> biasOut;
    void initalize() { //initalizes weights and biases
        hWeightIN.resize(inSize, std::vector<double>(hidSize, 0));
        for(int i = 0;i<inSize;++i) {
            for(int j = 0;j<hidSize;++j) {
                hWeightIN[i][j] = ((double) rand() / RAND_MAX) - 1;
            }
        }
        hWeightOUT.resize(hidSize, std::vector<double>(outSize, 0.0));
        for(int i = 0;i<hidSize;++i) {
            for(int j = 0;j<outSize;++j) {
                hWeightOUT[i][j] = ((double) rand() / RAND_MAX) - 1;
            }
        }
        hBias.resize(hidSize, 0.0);
        biasOut.resize(outSize, 0.0);
    }
};