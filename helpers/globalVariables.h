/**
 * Are we training?
 */
bool inTraining = true;


/**
 * Various print customizations
 */
bool printSteps = false;
bool printAdvancedSteps = false;

bool printTruth = false;
bool printInputs = false;
bool printTestInputs = false;
bool printWeights = false;


/**
 * Defines the lengths of the input, hidden, and output layers
 */
int inputNodes;
int hiddenNodes;
int outputNodes;


/**
 * Stores the values in the input, hidden, and output layers
 */
double *a;
double *h;
double *F;


/**
 * Defines the appropriate weights and their changes during training
 */
double **Wkj;
double **Wj0;
double **dWkj;
double **dWj0;


/**
 * Defines the biases
 */
double *Θh;
double *ΘF;


/**
 * Variables related to training
 */
double weightHigh;
double weightLow;
int maxIterations;
int errorThreshold;
double l = 0.5;
int expected;


/**
 * Test Inputs & Truth Table
 */
int testInputRows;
int testInputCols;
int **testInputs;

int truthTableRows;
int truthTableCols;
double **truthTable;


/**
 * Variables for echoParams()
 */
string networkType = "A-B-1";
string activationFunctionType = "sigmoid";
string weightSource = "RANDOM";
string weightChangesApplication = "store then apply";


/**
 * Inputs
 */
int **inputs;