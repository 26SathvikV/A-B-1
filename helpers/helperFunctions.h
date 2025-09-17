using namespace std;

#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>
#include <random>

//Prints an array
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
}

//Prints a 2d array as a table
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
}

//Sigmoid Activation Function
double sigmoid(double x = 0.0)
{
    return 1/(1+exp(-1.0 * x));
}

//Derives a function
double derivative(double (*f)(double), double x, double h) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

//Random in a range
double randomRange(double start, double end)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> distr(start, end);

    return distr(gen);
}

//Simple error function
double simpleError(unsigned target, unsigned output)
{
    return 0.5 * ((target - output) ^ 2);
}

//Makes a 2d double array from a pointer variable
double** make2dDoubleArray (double **arr, int rows, int columns)
{
    arr = new double*[rows];

    for (int fIndex = 0; fIndex < rows; fIndex++)
        arr[fIndex] = new double[columns];
    
    return arr;
}

//Sets the random weights for a 2d array
void fillRandom(double* weights, int weightLen, double wLow, double wHigh)
{
    for (int index = 0; index < weightLen; index++)
    {
        weights[index] = randomRange(wLow, wHigh);
    }
}

//Sets the random weights for a 2d array
void fillRandom2d(double** weights, int weightRows, int weightCols, double wLow, double wHigh)
{
    for (int fIndex = 0; fIndex < weightRows; fIndex++)
    {
        fillRandom(weights[fIndex], weightCols, wLow, wHigh);
    }
}