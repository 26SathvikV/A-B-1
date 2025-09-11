#include "helperFunctions.h";

//Activation Function
double activationFunction(unsigned x)
{
    return sigmoid(x);
}


//Derivative of the Activation Function
double activationFunctionDerivative(unsigned x)
{
    return derivative(sigmoid, x, 0.0001);
}

//Error Function
double errorFunction(unsigned target, unsigned output)
{
    return simpleError(target, output);
}