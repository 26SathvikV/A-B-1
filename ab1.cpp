//* Trains & Runs an A-B-1 AI Network using Steepest Gradient Descent

#include "helperFunctions.h"
#include <iostream>
#include <fstream>
using namespace std;



//Variables

//Are we training?
bool inTraining = true;

//Checks
bool printActivations = true;
bool printHidden = true;
bool printOutputs = true;
bool printWeights = true;
bool printTable = true;
bool printTraining = true;

//Max Layers
int inputNodes;
int hiddenNodes;
int outputNodes;

//Layer Arrays
double *a;
double *h;
double *F;

//Weights
double **Wkj;
double **Wj0;

//Training Params
double weightHigh;
double weightLow;
int maxIterations;
int errorThreshold;
double l = 0.5;
bool truthTable[2][2] = {{0,0},{0,1}}; // AND
int expected;



//Specific Helper Functions

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



//Learning Functions

//Hidden Layer Forward Pass
void hiddenLayerForwardPass(int k, int j)
{
    double Θ = 0;
    for (int index = 0; index < inputNodes; index++)
    {
        Θ += a[index] * Wkj[index][j];
    }

    h[j] = activationFunction(Θ);
}

//Output Layer Forward Pass
void outputLayerForwardPass(int k, int j)
{
    double Θ = 0;
    for (int index = 0; index < hiddenNodes; index++)
    {
        Θ += h[index] * Wj0[index][0];
    }

    F[0] = activationFunction(Θ);
}


//Ouput Layer Backward Pass
void outputLayerBackwardPass(int k, int j, int expected, double Θ)
{
    double ω = errorFunction(expected, F[0]);

    double ψ = ω * activationFunctionDerivative(Θ);

    double weightGrad = -1 * h[j] * ψ;

    double dW = -1 * l * weightGrad;

    Wj0[j][0] += dW;
}

//Hidden Layer Backward Pass
void hiddenLayerBackwardPass(int k, int j, double Θ, double ψ)
{
    double Ω = ψ * Wj0[j][0];

    double Ψ = Ω * activationFunctionDerivative(Θ);

    double weightGrad = -1 * a[k] * Ψ;

    double dW = -1 * l * weightGrad;

    w[k][j] += dW;
}



//Process Functions

//Interprets the data in a file
void interpFile(string name)
{}

//Sets the configuration parameters
void setParams(int activationLayers, int hiddenLayers, int outputLayers, double wHigh, double wLow)
{
    inputNodes = activationLayers;
    hiddenNodes = hiddenLayers;
    outputNodes = outputLayers;

    weightHigh = wHigh;
    weightLow = wLow;

    a = new double[inputNodes];
    h = new double[hiddenNodes];
    F = new double[outputNodes];

    setWkj(wLow, wHigh);
}

//Sets the weights for the input → hidden
void setWkj(double wLow, double wHigh)
{
    Wkj = new double*[inputNodes];
    for (int fIndex = 0; fIndex < inputNodes; fIndex++)
    {
        Wkj[fIndex] = new double[hiddenNodes];
        for (int sIndex = 0; sIndex < hiddenNodes; sIndex++)
        {
            Wkj[fIndex][sIndex] = random(weightLow, weightHigh);
        }
    }
}

//Sets the weights for the hidden → output
void setWj0(double wLow, double wHigh)
{
    Wj0 = new double*[hiddenNodes];
    for (int fIndex = 0; fIndex < hiddenNodes; fIndex++)
    {
        Wj0[fIndex] = new double[outputNodes];
        for (int sIndex = 0; sIndex < outputNodes; sIndex++)
        {
            Wj0[fIndex][sIndex] = random(weightLow, weightHigh);
        }
    }
}

//Echoes the configuration parameters
void echoParams()
{}

//Allocates memory for the network arrays
void allocateMemory()
{}

//Populates the arrays
void populateArrays()
{

}

//Trains & reports training results
void train()
{
    
    if (inTraining)
    {
        cout << weightHigh;
        cout << weightLow;
    }
}

//Runs all the test cases
void runTests()
{
    if (inTraining)
    {
        cout << "Training...";
    }
}




//Main Function - runs all of the code that was built so far
int main(int argc, char* argv[])
{
    printf("Starting...");
    
    //Initialize main function variables
    string pConfigFile = "configdata-and.txt";
    time_t t;
    clock_t t1, t2;

    if (argc > 1)
        pConfigFile = argv[1];
    t1 = clock();

    //Runs all of the previous functions
    setParams(2, 2, 1, 0.0, 1.0);
    echoParams();
    allocateMemory();
    populateArrays();
    train();
    runTests();
    return 0;
}