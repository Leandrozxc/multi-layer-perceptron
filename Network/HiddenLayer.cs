namespace Multi_layer_perceptron.Network;

/// <summary>
/// Represents the hidden layer with sigmoid activation in the neural network.
/// </summary>
public class HiddenLayer
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;

    /// <summary>
    /// Gets the weight matrix connecting inputs to hidden neurons [inputSize, hiddenSize].
    /// </summary>
    public double[,] Weights { get; }

    /// <summary>
    /// Gets the bias values for the hidden neurons.
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
    /// Initializes a new instance of the <see cref="HiddenLayer"/> class with Xavier uniform initialization.
    /// </summary>
    /// <param name="inputSize">The number of inputs to this layer.</param>
    /// <param name="hiddenSize">The number of neurons in this layer.</param>
    /// <param name="random">Random number generator for Xavier initialization.</param>
    public HiddenLayer(int inputSize, int hiddenSize, Random random)
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        Weights = new double[inputSize, hiddenSize];
        Biases = new double[hiddenSize];
        Outputs = new double[hiddenSize];
        Deltas = new double[hiddenSize];

        double limit = Math.Sqrt(6.0 / (inputSize + hiddenSize));
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                Weights[i, j] = (random.NextDouble() * 2.0 - 1.0) * limit;
            }
        }
    }

    /// <summary>
    /// Computes the forward pass activation for each hidden neuron: out_j = sigmoid(sum_i(in_i * w_ij) + b_j).
    /// </summary>
    /// <param name="input">Input values vector.</param>
    /// <returns>Hidden layer activation outputs.</returns>
    public double[] Forward(double[] input)
    {
        for (int j = 0; j < _hiddenSize; j++)
        {
            double sum = Biases[j];
            for (int i = 0; i < _inputSize; i++)
            {
                sum += input[i] * Weights[i, j];
            }
            Outputs[j] = Sigmoid(sum);
        }
        return Outputs;
    }

    /// <summary>
    /// Computes error deltas for the hidden layer using next layer's weights and deltas:
    /// delta_j = out_j * (1 - out_j) * sum_k(next.Deltas[k] * next.Weights[j,k]).
    /// </summary>
    /// <param name="input">Input values vector (unused in delta formula, preserved for API signature).</param>
    /// <param name="next">The subsequent output layer.</param>
    public void Backward(double[] input, OutputLayer next)
    {
        int outputSize = next.Outputs.Length;
        for (int j = 0; j < _hiddenSize; j++)
        {
            double errorSum = 0.0;
            for (int k = 0; k < outputSize; k++)
            {
                errorSum += next.Deltas[k] * next.Weights[j, k];
            }
            double outJ = Outputs[j];
            Deltas[j] = outJ * (1.0 - outJ) * errorSum;
        }
    }

    /// <summary>
    /// Updates layer weights and biases using gradient descent: w_ij -= lr * delta_j * in_i; b_j -= lr * delta_j.
    /// </summary>
    /// <param name="input">The layer input vector.</param>
    /// <param name="lr">The learning rate.</param>
    public void UpdateWeights(double[] input, double lr)
    {
        for (int i = 0; i < _inputSize; i++)
        {
            for (int j = 0; j < _hiddenSize; j++)
            {
                Weights[i, j] -= lr * Deltas[j] * input[i];
            }
        }

        for (int j = 0; j < _hiddenSize; j++)
        {
            Biases[j] -= lr * Deltas[j];
        }
    }

    private static double Sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }
}

