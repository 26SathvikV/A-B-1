//* Trains & Runs an A-B-1 AI Network using Steepest Gradient Descent

#include <iostream>
#include <fstream>
using namespace std;


//Divisions of the code for organization purposes

#include "helpers/globalVariables.h" //Stores all the variables
#include "helpers/helperFunctions.h" //Mathematical functions
#include "helpers/specificHelpers.h" //Specific activation & error functions
#include "helpers/learningFunctions.h" //Functions for learning
#include "helpers/fileInterp.h" //Unused, holding space for file interpetration



//Process Functions

//Sets the configuration parameters
void setParams(int activationLayers, int hiddenLayers, int outputLayers, double wHigh, double wLow)
{
    if (printSteps)
    {
        cout << "Setting parameters...";
    }

    //Topology
    inputNodes = activationLayers;
    hiddenNodes = hiddenLayers;
    outputNodes = outputLayers;

    //Weight Range
    weightHigh = wHigh;
    weightLow = wLow;

    if (printSteps)
    {
        cout << "Parameters set!";
    }
}


//Echoes the configuration parameters
void echoParams()
{
    if (printSteps)
    {
        cout << "Echoing parameters...";
        cout << "\n\n";
    }

    //Network Configuration
    cout << "NETWORK CONFIGURATION";
    cout << "-------------------------------------------";
    cout << "Network Type: A-B-1"; //Change manually
    cout << "Inputs (A): " << inputNodes;
    cout << "Hidden Layers (B): " << hiddenNodes;
    cout << "Outputs (C): " << outputNodes;
    cout << "Activation Function: " << activationFunctionType;
    cout << "\n\n";

    //Runtime Params
    cout << "RUNTIME PARAMETERS";
    cout << "-------------------------------------------";
    cout << "Training: " << inTraining;
    cout << "Weight Source: RANDOM"; //Change manually if needed
    cout << "Weight Range: " << weightLow << " - " << weightHigh;
    cout << "Learning Rate (λ): " << l;
    cout << "Maximum Iterations: " << maxIterations;
    cout << "Error Threshold: " << errorThreshold;
    cout << "Apply Weight Changes: store then apply"; //Change manually if needed
    cout << "Truth Table \n" << truthTable;
    cout << "\n\n";

    //Memory Alloc
    cout << "Memory Allocation";
    cout << "-------------------------------------------";
    cout << "a[] size: " << inputNodes;
    cout << "h[] size: " << hiddenNodes;
    cout << "F[] size: " << outputNodes;
    cout << "\n";
    cout << "Wkj: [" << inputNodes << "][" << hiddenNodes << "]";
    cout << "Wj0: [" << hiddenNodes << "][" << outputNodes << "]";
    cout << "\n";
    cout << "Θh[] size: " << hiddenNodes;
    cout << "ΘF[] size: " << outputNodes;
    cout << "\n\n";
    cout << "Parameters echoed!";
}


//Allocates memory for the network arrays
void allocateMemory()
{
    if (printSteps)
    {
        cout << "Allocating memory...";
    }

    //Layers
    a = new double[inputNodes];
    h = new double[hiddenNodes];
    F = new double[outputNodes];

    //Weights
    make2dDoubleArray(Wkj, inputNodes, hiddenNodes);
    make2dDoubleArray(Wj0, hiddenNodes, outputNodes);
    make2dDoubleArray(dWkj, inputNodes, hiddenNodes);
    make2dDoubleArray(dWj0, hiddenNodes, outputNodes);

    //Biases
    Θh = new double[hiddenNodes];
    ΘF = new double[outputNodes];

    if (printSteps)
    {
        cout << "Memory allocated!";
    }
}


//Populates the arrays
void populateArrays()
{
    if (printSteps)
    {
        cout << "Populating arrays...";
    }

    //Fill the weights with random values
    fillRandom2d(Wkj, weightLow, weightHigh);
    fillRandom2d(Wj0, weightLow, weightHigh);

    if (printSteps)
    {
        cout << "Arrays populated!";
    }
}


//Trains & reports training results
void train()
{
    if (printSteps)
    {
        cout << "Training...";
    }

    int currIteration = 0;

    //Training Loop
    while (currIteration < maxIterations)
    {
        double sumError = 0.0;
        
        for (int index = 0; index < testInputRows; index++)
        {
            int *testCase = testInputs[index];
            a[0] = testCase[0];
            a[1] = testCase[1];
            int expected = truthTable[testCase[0]][testCase[1]];

            for (int j = 0; j < hiddenNodes; j++)
            {
                hiddenLayerForwardPass(index, j);
            }
            
            for (int j = 0; j < outputNodes; j++)
            {
                outputLayerForwardPass(index, j);
            }
            
            sumError += errorFunction(expected, F[0]);
            
            for (int j = 0; j < outputNodes; j++)
            {
                double ψ = outputLayerBackwardPass(index, j, expected);

                for (int hidLayNum = 0; hidLayNum < hiddenNodes; hidLayNum++)
                {
                    hiddenLayerBackwardPass(index, hidLayNum, ψ);
                }
            }
        }

        double avgError = sumError/testInputRows;
        currIteration++;
        bool stoppedByError = false;
        bool stoppedByIteration = false;

        if (avgError <= errorThreshold)
        {
            stoppedByError = true;
        }

        if (currIteration >= maxIterations)
        {
            stoppedByIteration = true;
        }
        
        if (printSteps)
        {
            if (stoppedByError && stoppedByIteration)
            {
                cout << "Stopped by error and maxed iterations";
            }
            else if (stoppedByError)
            {
                cout << "Stopped by error";
            }
            else
            {
                cout << "Stopped by iterations";
            }
        }

        if (stoppedByError || stoppedByIteration)
        {
            break;
        }
    }

    if (printSteps)
    {
        cout << "Trained!";
    }
}


//Runs all the test cases
double *runTests()
{
    if (printSteps)
    {
        cout << "Running tests...";
    }

    double *runOutputs = new double[outputNodes];

    for (int index = 0; index < testInputRows; index++)
    {
        for (int n = 0; n < testInputCols; n++)
        {
            a[n] = testInputs[index][n];
        }
        for (int j = 0; j < hiddenNodes; j++)
        {
            hiddenLayerForwardPass(index, j);
        }
        for (int j = 0; j < outputNodes; j++)
        {
            outputLayerForwardPass(index,j);
            runOutputs[j] = F[j];
        }
    }

    if (printSteps)
    {
        cout << "Tests finished!";
    }

    return runOutputs;
}



//Main Function - runs all of the code that was built so far
int run(int argc, char* argv[])
{
    if (printSteps)
    {
        printf("Starting...");
    }
    
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