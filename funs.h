#include <vector>
#include <iostream>
#include <cmath>
double dot(std::vector<std::vector<double>> &weights, double *input, int rows, int cols)
{
    int res = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            res += weights[i][j] * input[j];
        }
    }
    return res;
}
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}
double dsigmoid(double x)
{
    return x * (1.0 - x);
}
double mse(std::vector<double> exp, std::vector<double> act) {
    double ans = 0;
    for(int i = 0;i<exp.size();++i) {
        ans+=(exp[i]-act[i])*(exp[i]-act[i]);
    }
    return ans/exp.size();
}
std::vector<double> mseGrad(std::vector<double> exp, std::vector<double> act) {
    std::vector<double> gradient(exp.size());
    for(int i = 0;i<exp.size();++i) {
        gradient[i] = (2.0/gradient.size()) * (exp[i]-act[i]);
    }
    return gradient;
}
