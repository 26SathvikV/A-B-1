using namespace std;

/**
 * All of the appropriate libraries
 */
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>
#include <random>



/**
 * Prints an array
 * 
 * @category T       a given/placeholder data type (double, integer, etc.)
 * 
 * @param arr        the array to print
 * @param numCols    the number of columns in the array
 */
template <typename T>
void printArray(T *arr, int numCols)
{
    if (!arr) {
        cout << "Empty Array \n";
        return;
    }

    for (int col = 0; col < numCols; col++) {
        cout << arr[col] << "\t";
    }
    cout << "\n";
} //void printArray(T *arr, int numCols)


/**
 * Prints a 2d array
 * 
 * @category T       a given/placeholder data type (double, integer, etc.)
 * 
 * @param arr        the array to print
 * @param numRows    the number of rows in the array
 * @param numCols    the number of columns in the array
 */
template <typename T>
void printArray2d(T **arr, int numRows, int numCols)
{
    if (!arr) {
        cout << "Empty Array \n";
        return;
    }

    //Header
    cout << "  \t";
    for (int col = 0; col < numCols; col++) {
        cout << col << "\t";
        if (typeid(arr[0][0]).name() == "double")
            cout << "\t";
    }
    cout << "\n";


    //Rows
    for (int row = 0; row < numRows; row++) {
        cout << row << "\t";
        printArray(arr[row], numCols);
    }
} //void printArray2d(T **arr, int numRows, int numCols)


/**
 * Sigmoid Activation Function
 * 
 * @param x    the value to pass through the sigmoid
 * 
 * @return the sigmoid of x
 */
double sigmoid(double x = 0.0)
{
    return 1/(1+exp(-1.0 * x));
}

/**
 * Derives a function, based on the AP Calculus definition
 * 
 * @param f    the function to derive
 * @param x    the value to pass through the derivative
 * @param h    a small change in x
 * 
 * @return the derviative of f(x)
 */
double derivative(double (*f)(double), double x, double h) {
    return (f(x + h) - f(x)) / (h);
}

/**
 * Random in a range
 * 
 * @param start    the minimum random value
 * @param end      the maximum random value
 * 
 * @return a random value between start and end
 */
double randomRange(double start, double end)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> distr(start, end);

    return distr(gen);
}

/**
 * A simple error function
 * 
 * @param target    the target value
 * @param output    the output
 * 
 * @return the error value
 */
double simpleError(unsigned target, unsigned output)
{
    return 0.5 * (target - output) * (target - output);
}

/**
 * Makes a 2d double array from a pointer variable
 * 
 * @param arr        the 2d array pointer
 * @param rows       the number of rows in the 2d array
 * @param columns    the number of columns in the 2d array
 * 
 * @return the 2d array
 */
double** make2dDoubleArray (double **arr, int rows, int columns)
{
    arr = new double*[rows];

    for (int fIndex = 0; fIndex < rows; fIndex++)
        arr[fIndex] = new double[columns];
    
    return arr;
}

/**
 * Sets the random doubles for an array
 * 
 * @param weights      the array to fill
 * @param weightLen    the length of weights
 * @param wLow         the minimum random number
 * @param wHigh        the maximum random number
 */
void fillRandom(double* weights, int weightLen, double wLow, double wHigh)
{
    for (int index = 0; index < weightLen; index++)
    {
        weights[index] = randomRange(wLow, wHigh);
    }
}

/**
 * Sets the random doubles for a 2d array
 * 
 * @param weights        the array to fill
 * @param weightRows     the amount of rows in weights
 * @param weightCols     the amount of columns in weights
 * @param wLow           the minimum random number
 * @param wHigh          the maximum random number
 */
void fillRandom2d(double** weights, int weightRows, int weightCols, double wLow, double wHigh)
{
    for (int fIndex = 0; fIndex < weightRows; fIndex++)
    {
        fillRandom(weights[fIndex], weightCols, wLow, wHigh);
    }
}