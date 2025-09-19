#include "globalVariables.h"
#include "specificHelpers.h"



/**
 * Hidden Layer Forward Pass
 * 
 * @param k           input node index
 * @param j           hidden node index
 */
void hiddenLayerForwardPass(int k, int j)
{
    double Θ = 0;
    for (int index = 0; index < inputNodes; index++)
    {
        Θ += a[index] * Wkj[index][j];
    }

    h[j] = activationFunction(Θ);
    Θh[j] = Θ;
} //hiddenLayerForwardPass(int k, int j)


/**
 * Output Layer Forward Pass
 * 
 * @param k           input node index
 * @param j           hidden node index
 */
void outputLayerForwardPass(int k, int j)
{
    double Θ = 0;
    for (int index = 0; index < hiddenNodes; index++)
    {
        Θ += h[index] * Wj0[index][0];
    }

    F[j] = activationFunction(Θ);
    ΘF[j] = Θ;
} //outputLayerForwardPass(int k, int j)


/**
 * Ouput Layer Backward Pass
 * 
 * @param k           input node index
 * @param j           hidden node index
 * @param expected    expected value
 */
double outputLayerBackwardPass(int k, int j, int expected)
{
    double Θ = ΘF[j];

    double ω = errorFunction(expected, F[0]);

    double ψ = ω * activationFunctionDerivative(Θ);

    double weightGrad = -1 * h[j] * ψ;

    double dW = -1 * l * weightGrad;

    dWj0[k][j] = dW;

    return ψ;
} //outputLayerBackwardPass(int k, int j, int expected)


/**
 * Hidden Layer Backward Pass
 * 
 * @param k    input node index
 * @param j    hidden node index
 * @param ψ    error signal at the output node
 */
void hiddenLayerBackwardPass(int k, int j, double ψ)
{
    double Θ = Θh[j];

    double Ω = ψ * Wj0[j][0];

    double Ψ = Ω * activationFunctionDerivative(Θ);

    double weightGrad = -1 * a[k] * Ψ;

    double dW = -1 * l * weightGrad;

    dWkj[k][j] = dW;
} //hiddenLayerBackwardPass(int k, int j, double ψ)


/**
 * Applies all weight updates
 */
void applyWeightUpdates()
{
    for (int k = 0; k < inputNodes; k++)
    {
        for (int j = 0; j < hiddenNodes; j++)
        {
            Wkj[k][j] += dWkj[k][j];
        }
    }

    for (int j = 0; j < hiddenNodes; j++)
    {
        for (int i = 0; i < outputNodes; i++)
        {
            Wj0[j][i] += dWj0[j][i];
        }
    }
} //applyWeightUpdates()