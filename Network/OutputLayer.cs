namespace Multi_layer_perceptron.Network;

/// <summary>
/// Represents the output layer with sigmoid activation in the neural network.
/// </summary>
public class OutputLayer
{
    private readonly int _hiddenSize;
    private readonly int _outputSize;

    /// <summary>
    /// Gets the weight matrix connecting hidden neurons to output neurons [hiddenSize, outputSize].
    /// </summary>
    public double[,] Weights { get; }

    /// <summary>
    /// Gets the bias values for the output neurons.
    /// </summary>
    public double[] Biases { get; }

    /// <summary>
    /// Gets the activations produced during the forward pass.
    /// </summary>
    public double[] Outputs { get; }

    /// <summary>
    /// Gets the error deltas computed during backpropagation.
    /// </summary>
    public double[] Deltas { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="OutputLayer"/> class with Xavier uniform initialization.
    /// </summary>
    /// <param name="hiddenSize">The number of hidden neurons.</param>
    /// <param name="outputSize">The number of output neurons.</param>
    /// <param name="random">Random number generator for Xavier initialization.</param>
    public OutputLayer(int hiddenSize, int outputSize, Random random)
    {
        _hiddenSize = hiddenSize;
        _outputSize = outputSize;
        Weights = new double[hiddenSize, outputSize];
        Biases = new double[outputSize];
        Outputs = new double[outputSize];
        Deltas = new double[outputSize];

        double limit = Math.Sqrt(6.0 / (hiddenSize + outputSize));
        for (int j = 0; j < hiddenSize; j++)
        {
            for (int k = 0; k < outputSize; k++)
            {
                Weights[j, k] = (random.NextDouble() * 2.0 - 1.0) * limit;
            }
        }
    }

    /// <summary>
    /// Computes the forward pass activation for each output neuron: out_k = sigmoid(sum_j(hiddenOut_j * w_jk) + b_k).
    /// </summary>
    /// <param name="hiddenOut">Activations from the hidden layer.</param>
    /// <returns>Output layer activation vector.</returns>
    public double[] Forward(double[] hiddenOut)
    {
        for (int k = 0; k < _outputSize; k++)
        {
            double sum = Biases[k];
            for (int j = 0; j < _hiddenSize; j++)
            {
                sum += hiddenOut[j] * Weights[j, k];
            }
            Outputs[k] = Sigmoid(sum);
        }
        return Outputs;
    }

    /// <summary>
    /// Computes error deltas for the output layer using MSE loss and sigmoid derivative:
    /// delta_k = (out_k - target_k) * out_k * (1 - out_k).
    /// </summary>
    /// <param name="target">The expected target vector.</param>
    public void ComputeDeltas(double[] target)
    {
        for (int k = 0; k < _outputSize; k++)
        {
            double outK = Outputs[k];
            Deltas[k] = (outK - target[k]) * outK * (1.0 - outK);
        }
    }

    /// <summary>
    /// Updates layer weights and biases using gradient descent: w_jk -= lr * delta_k * hiddenOut_j; b_k -= lr * delta_k.
    /// </summary>
    /// <param name="hiddenOut">The hidden layer output vector.</param>
    /// <param name="lr">The learning rate.</param>
    public void UpdateWeights(double[] hiddenOut, double lr)
    {
        for (int j = 0; j < _hiddenSize; j++)
        {
            for (int k = 0; k < _outputSize; k++)
            {
                Weights[j, k] -= lr * Deltas[k] * hiddenOut[j];
            }
        }

        for (int k = 0; k < _outputSize; k++)
        {
            Biases[k] -= lr * Deltas[k];
        }
    }

    private static double Sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }
}

