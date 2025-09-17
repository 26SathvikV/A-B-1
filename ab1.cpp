//* Trains & Runs an A-B-1 AI Network using Steepest Gradient Descent

#include <iostream>
#include <fstream>
using namespace std;


//Divisions of the code for organization purposes

#include "helpers/learningFunctions.h" //Functions for learning
#include "helpers/fileInterp.h" //Unused, holding space for file interpetration



//Process Functions

//Sets the configuration parameters
void setParams(int **inputs, int inpRows, int inpCols, double **truth, int tRows, int tCols)
{
    if (printSteps)
    {
        cout << "Setting parameters...\n\n";
    }

    //Topology
    cout << "Activation layers (a): ";
    cin >> inputNodes;
    cout << "Hidden layers (h): ";
    cin >> hiddenNodes;
    cout << "Output layers (i): ";
    cin >> outputNodes;

    //Weight Range
    cout << "\nMinimum weight value: ";
    cin >> weightLow;
    cout << "Maximum weight value: ";
    cin >> weightHigh;

    //Learning Variables
    cout << "\nLearning rate (l): ";
    cin >> l;
    cout << "Maximum iterations: ";
    cin >> maxIterations;
    cout << "Error threshold: ";
    cin >> errorThreshold;

    //Test Inputs & Truth Table
    testInputRows = inpRows;
    testInputCols = inpCols;
    testInputs = inputs;
    
    truthTableRows = tRows;
    truthTableCols = tCols;
    truthTable = truth;

    if (printSteps)
    {
        cout << "\nParameters set!\n\n";
    }
}


//Echoes the configuration parameters
void echoParams()
{
    if (printSteps)
    {
        cout << "Echoing parameters...\n\n";
    }

    //Network Configuration
    cout << "\n\n\nNETWORK CONFIGURATION\n";
    cout << "-------------------------------------------\n";
    cout << "Network Type: " << networkType << "\n"; //Change manually
    cout << "Inputs (A): " << inputNodes << "\n";
    cout << "Hidden Layers (B): " << hiddenNodes << "\n";
    cout << "Outputs (C): " << outputNodes << "\n";
    cout << "Activation Function: " << activationFunctionType << "\n";

    if (printTruth)
    {
        cout << "Truth Table \n";
        printArray2d(truthTable, truthTableRows, truthTableCols);
        cout << "\n";
    }
    cout << "\n\n";

    //Runtime Params
    cout << "RUNTIME PARAMETERS\n";
    cout << "-------------------------------------------\n";
    cout << "Training: " << inTraining << "\n";
    cout << "Weight Source: " << weightSource << "\n";
    cout << "Weight Range: " << weightLow << " - " << weightHigh << "\n";
    cout << "Learning Rate (λ): " << l << "\n";
    cout << "Maximum Iterations: " << maxIterations << "\n";
    cout << "Error Threshold: " << errorThreshold << "\n";
    cout << "Apply Weight Changes: " << weightChangesApplication << "\n";

    if (printInputs)
    {
        cout << "Test Inputs \n";
        printArray2d(testInputs, testInputRows, testInputCols);
        cout << "\n";
    }
    cout << "\n\n";

    //Memory Alloc
    cout << "Memory Allocation\n";
    cout << "-------------------------------------------\n";
    cout << "a[] size: " << inputNodes << "\n";
    cout << "h[] size: " << hiddenNodes << "\n";
    cout << "F[] size: " << outputNodes << "\n";
    cout << "\n";
    cout << "Wkj: [" << inputNodes << "][" << hiddenNodes << "]\n";
    cout << "Wj0: [" << hiddenNodes << "][" << outputNodes << "]\n";
    cout << "\n";
    cout << "Θh[] size: " << hiddenNodes << "\n";
    cout << "ΘF[] size: " << outputNodes << "\n";
    
    if (printSteps)
        cout << "\n\nParameters echoed!\n\n\n";
}


//Allocates memory for the network arrays
void allocateMemory()
{
    if (printSteps)
    {
        cout << "Allocating memory...\n";
    }

    //Layers
    a = new double[inputNodes];
    h = new double[hiddenNodes];
    F = new double[outputNodes];

    //Weights
    Wkj = make2dDoubleArray(Wkj, inputNodes, hiddenNodes);
    Wj0 = make2dDoubleArray(Wj0, hiddenNodes, outputNodes);
    dWkj = make2dDoubleArray(dWkj, inputNodes, hiddenNodes);
    dWj0 = make2dDoubleArray(dWj0, hiddenNodes, outputNodes);

    //Biases
    Θh = new double[hiddenNodes];
    ΘF = new double[outputNodes];

    if (printSteps)
    {
        cout << "\nMemory allocated!\n\n";
    }
}


//Populates the arrays
void populateArrays()
{
    if (printSteps)
    {
        cout << "Populating arrays...\n";
    }

    //Fill the weights with random values
    fillRandom2d(Wkj, inputNodes, hiddenNodes, weightLow, weightHigh);
    fillRandom2d(Wj0, hiddenNodes, outputNodes, weightLow, weightHigh);

    if (printWeights)
    {
        cout << "Wkj:\n";
        printArray2d(Wkj, inputNodes, hiddenNodes);
        cout << "\n\nWj0:\n";
        printArray2d(Wj0, hiddenNodes, outputNodes);
    }

    if (printSteps)
    {
        cout << "\nArrays populated!\n\n";
    }
}


//Trains & reports training results
void train()
{
    if (printSteps)
    {
        cout << "Training...\n";
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
            if (printAdvancedSteps || stoppedByError || stoppedByIteration)
                cout << "Iteration " << currIteration << ": ";

            if (stoppedByError && stoppedByIteration)
            {
                cout << "Stopped by error and maxed iterations\n";
            }
            else if (stoppedByError)
            {
                cout << "Stopped by error\n";
            }
            else if (stoppedByIteration)
            {
                cout << "Stopped by iterations\n";
            }
            else
            {
                if (printAdvancedSteps)
                    cout << "No issues \n";
            }
        }

        if (stoppedByError || stoppedByIteration)
        {
            break;
        }
    }

    if (printSteps)
    {
        cout << "\nTrained!\n\n";
    }
}


//Runs all the test cases
double *runTests()
{
    if (printSteps)
    {
        cout << "Running tests...\n";
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
        cout << "\nTests finished!\n\n";
    }

    return runOutputs;
}



//Runs all of the code that was built so far
int run(int argc, char* argv[])
{
    if (printSteps)
    {
        printf("Starting...\n");
    }
    
    //Initialize main function variables
    string pConfigFile = "configdata-and.txt";
    time_t t;
    clock_t t1, t2;

    if (argc > 1)
        pConfigFile = argv[1];
    t1 = clock();

    const int testInputRows = 4;
    const int testInputCols = 2;
    int testInputsFile[testInputRows][testInputCols] = {{0,0}, {0,1}, {1,0}, {1,1}};
    int **inputs;
    inputs =  new int*[testInputRows];
    for (int index = 0; index < testInputRows; index++)
    {
        inputs[index] = new int[2];
        inputs[index] = testInputsFile[testInputCols];
    }

    const int truthRows = 2;
    const int truthCols = 2;
    double truthFile[truthRows][truthCols] = {{0.0, 0.0}, {0.0, 1.0}};
    double **truth;
    truth =  new double*[truthRows];
    for (int index = 0; index < truthRows; index++)
    {
        truth[index] = new double[truthCols];
        truth[index] = truthFile[index];
    }

    //Runs all of the previous functions
    setParams(inputs, 2, 4, truth, 2, 2);
    echoParams();
    allocateMemory();
    populateArrays();
    train();
    double* runOutputs = runTests();

    //Final prints
    if (printWeights)
    {
        cout << "\nFinal Wkj:\n";
        printArray2d(Wkj, inputNodes, hiddenNodes);

        cout << "\nFinal Wj0:\n";
        printArray2d(Wkj, hiddenNodes, outputNodes);
    }
    if (printInputs)
    {
        cout << "\nInputs:\n";
        printArray(a, inputNodes);
    }

    if (printTestInputs)
    {
        cout << "\n\n\n\nTest Inputs:\n";
        printArray2d(testInputs, testInputRows, testInputCols);
    }

    if (printTruth)
    {
        cout << "\nTruth Table:\n";
        printArray2d(truth, truthRows, truthCols);
    }
    
    cout << "\nTest Outputs:\n";
    printArray(runOutputs, outputNodes);

    cout << t1;

    return 0;
}

//Main function
int main(int argc, char* argv[])
{
    run(argc, argv);
    
    string doWeEnd = "";
    cout << "\n\nEnter \"quit\" to stop.\n";
    cin >> doWeEnd;

    if (doWeEnd == "quit")
        return 0;
}