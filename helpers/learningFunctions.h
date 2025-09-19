#include "globalVariables.h"
#include "specificHelpers.h"

//Hidden Layer Forward Pass
void hiddenLayerForwardPass(int k, int j)
{
    double Θ = 0;
    for (int index = 0; index < inputNodes; index++)
    {
        Θ += a[index] * Wkj[index][j];
    }

    h[j] = activationFunction(Θ);
    Θh[j] = Θ;
}

//Output Layer Forward Pass
void outputLayerForwardPass(int k, int j)
{
    double Θ = 0;
    for (int index = 0; index < hiddenNodes; index++)
    {
        Θ += h[index] * Wj0[index][0];
    }

    F[j] = activationFunction(Θ);
    ΘF[j] = Θ;
}


//Ouput Layer Backward Pass
double outputLayerBackwardPass(int k, int j, int expected)
{
    double Θ = ΘF[j];

    double ω = errorFunction(expected, F[0]);

    double ψ = ω * activationFunctionDerivative(Θ);

    double weightGrad = -1 * h[j] * ψ;

    double dW = -1 * l * weightGrad;

    dWj0[k][j] = dW;

    return ψ;
}

//Hidden Layer Backward Pass
void hiddenLayerBackwardPass(int k, int j, double ψ)
{
    double Θ = Θh[j];

    double Ω = ψ * Wj0[j][0];

    double Ψ = Ω * activationFunctionDerivative(Θ);

    double weightGrad = -1 * a[k] * Ψ;

    double dW = -1 * l * weightGrad;

    dWkj[k][j] = dW;
}

//Applies all weight updates
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
}