namespace Multi_layer_perceptron.Network;

/// <summary>
/// Represents a 3-layer Multilayer Perceptron (Input, Hidden, Output) with backpropagation.
/// </summary>
public class NeuralNetwork
{
    /// <summary>
    /// Gets the input layer.
    /// </summary>
    public InputLayer Input { get; }

    /// <summary>
    /// Gets the hidden layer.
    /// </summary>
    public HiddenLayer Hidden { get; }

    /// <summary>
    /// Gets the output layer.
    /// </summary>
    public OutputLayer Output { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="NeuralNetwork"/> class.
    /// Weights are initialized using Xavier uniform distribution with the specified seed.
    /// </summary>
    /// <param name="inputSize">The number of input features.</param>
    /// <param name="hiddenSize">The number of hidden neurons.</param>
    /// <param name="outputSize">The number of output neurons.</param>
    /// <param name="seed">Random seed for reproducible initialization.</param>
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int seed = 42)
    {
        var random = new Random(seed);
        Input = new InputLayer(inputSize);
        Hidden = new HiddenLayer(inputSize, hiddenSize, random);
        Output = new OutputLayer(hiddenSize, outputSize, random);
    }

    /// <summary>
    /// Executes a forward pass and returns the network's output activations.
    /// </summary>
    /// <param name="x">The input feature vector.</param>
    /// <returns>The output predictions.</returns>
    public double[] Predict(double[] x)
    {
        Input.Values = x;
        var hiddenOut = Hidden.Forward(x);
        return Output.Forward(hiddenOut);
    }

    /// <summary>
    /// Performs one training step using stochastic gradient descent on a single sample:
    /// forward pass, compute deltas, update weights, and return sample mean squared error.
    /// </summary>
    /// <param name="x">The input feature vector.</param>
    /// <param name="target">The expected target vector.</param>
    /// <param name="lr">The learning rate.</param>
    /// <returns>The mean squared error (MSE) for this sample.</returns>
    public double TrainStep(double[] x, double[] target, double lr)
    {
        // 1. Forward pass
        Predict(x);

        // 2. Output deltas
        Output.ComputeDeltas(target);

        // 3. Hidden deltas using the old output weights
        Hidden.Backward(x, Output);

        // 4. Update weights for both layers
        Output.UpdateWeights(Hidden.Outputs, lr);
        Hidden.UpdateWeights(x, lr);

        // 5. Compute and return sample MSE
        double sumSqErr = 0.0;
        int kLen = Output.Outputs.Length;
        for (int k = 0; k < kLen; k++)
        {
            double diff = Output.Outputs[k] - target[k];
            sumSqErr += diff * diff;
        }

        return sumSqErr / kLen;
    }

    /// <summary>
    /// Predicts the class index with the highest activation (argmax).
    /// </summary>
    /// <param name="x">The input feature vector.</param>
    /// <returns>The zero-based index of the predicted class.</returns>
    public int PredictClass(double[] x)
    {
        var outputs = Predict(x);
        int bestIdx = 0;
        double maxVal = outputs[0];
        for (int i = 1; i < outputs.Length; i++)
        {
            if (outputs[i] > maxVal)
            {
                maxVal = outputs[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}

