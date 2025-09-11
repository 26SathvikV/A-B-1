using namespace std;

//Sigmoid Activation Function
double sigmoid(double x = 0.0)
{
    return 1/(1+exp(-1 * x));
}

//Derives a function
double derivative(double (*f)(double), double x, double h) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

//Random in a range
unsigned random(unsigned start, unsigned end)
{
    return start + (rand() % (end - start + 1));
}

//Simple error function
double simpleError(unsigned target, unsigned output)
{
    return 0.5 * pow((target - output), 2);
}

//Makes a 2d double array from a pointer variable
void make2dDoubleArray (double **arr, int rows, int columns)
{
    arr = new double*[rows];

    for (int fIndex = 0; fIndex < rows; fIndex++)
        arr[fIndex] = new double[columns];
}

//Sets the random weights for a 2d array
void fillRandom(double* weights, double wLow, double wHigh)
{
    for (int fIndex = 0; fIndex < inputNodes; fIndex++)
    {
        weights[fIndex] = random(wLow, wHigh);
    }
}

//Sets the random weights for a 2d array
void fillRandom2d(double** weights, double wLow, double wHigh)
{
    for (int fIndex = 0; fIndex < inputNodes; fIndex++)
    {
        for (int sIndex = 0; sIndex < hiddenNodes; sIndex++)
        {
            weights[fIndex][sIndex] = random(wLow, wHigh);
        }
    }
}