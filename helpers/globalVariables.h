//Are we training?
bool inTraining = true;

//Checks
bool printSteps = false;
bool printAdvancedSteps = false;

bool printTruth = true;
bool printInputs = true;
bool printTestInputs = false;
bool printWeights = true;

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
int expected;

//Test Inputs & Truth Table
int testInputRows;
int testInputCols;
int **testInputs;

int truthTableRows;
int truthTableCols;
double **truthTable;

//Variables for Echo Params
string networkType = "A-B-1";
string activationFunctionType = "sigmoid";
string weightSource = "RANDOM";
string weightChangesApplication = "store then apply";

//Inputs
int **inputs;