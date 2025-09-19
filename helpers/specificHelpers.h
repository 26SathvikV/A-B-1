#include "helperFunctions.h"

/**
 * Sets the current activation function
 * 
 * @param x    the input value to pass through the activation function
 */
double activationFunction(unsigned x)
{
    return sigmoid(x);
}


/**
 * Derivative of activationFunction()
 * 
 * @param x    the input value to pass through the derivative of the activation function
 */
double activationFunctionDerivative(unsigned x)
{
    return derivative(sigmoid, x, 0.0001);
}

/**
 * Sets the current error function
 * 
 * @param target    the target value
 * @param output    the given output
 */
double errorFunction(unsigned target, unsigned output)
{
    return simpleError(target, output);
}