//Are we training?
bool inTraining = true;

//Checks
bool printActivations = true;
bool printHidden = true;
bool printOutputs = true;
bool printWeights = true;
bool printTable = true;
bool printTraining = true;
bool printSteps = true;

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
double **dWkj;
double **dWj0;

//Biases
double *Θh;
double *ΘF;

//Training Params
double weightHigh;
double weightLow;
int maxIterations;
int errorThreshold;
double l = 0.5;
const int testInputRows = 4;
const int testInputCols = 2;
int testInputs[testInputRows][testInputCols] = {{0,0},{0,1},{1,0},{1,1}};
int truthTable[2][2] = {{0,0},{0,1}}; // AND
int expected;

//Variables for Echo Params
char* activationFunctionType = "sigmoid";

//Inputs
int **inputs;