#include <iostream>
using namespace std;

double sigmoid(double x = 0.0)
{
    return 1/(1+exp(-1 * x));
}

double derivative(double (*f)(double), double x, double h) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

unsigned random(unsigned start, unsigned end)
{
    return start + (rand() % (end - start + 1));
}

double simpleError(unsigned target, unsigned output)
{
    return 0.5 * pow((target - output), 2);
}