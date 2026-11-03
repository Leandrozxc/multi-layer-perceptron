using Backprop;
using System.Globalization;

namespace Multi_layer_perceptron.Network;

/// <summary>Adapts the unchanged supplied NeuralNet coordinator to the application.</summary>
public class NeuralNetwork
{
    private readonly NeuralNet _engine;
    private readonly int _inputSize;
    private readonly int _outputSize;

    /// <summary>Creates a supplied network and loads reproducible initial weights through its public API.</summary>
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int seed = 42)
    {
        if (inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputSize), "Layer sizes must be positive.");
        _inputSize = inputSize;
        _outputSize = outputSize;
        _engine = new NeuralNet(inputSize, hiddenSize, outputSize);
        var random = new Random(seed);
        // loadWeights expects input weights, hidden weights, hidden biases, output biases.
        var values = new List<double>();
        for (int i = 0; i < inputSize * hiddenSize + hiddenSize * outputSize; i++)
            values.Add(random.Next(-1000, 1000) / 2000.0);
        values.AddRange(Enumerable.Repeat(0.01, hiddenSize + outputSize));
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllLines(path, values.Select(value => value.ToString("R", CultureInfo.CurrentCulture)));
            _engine.loadWeights(path);
        }
        finally { File.Delete(path); }
    }

    /// <summary>Returns outputs computed by the supplied coordinator.</summary>
    public double[] Predict(double[] x)
    {
        SetInputs(x);
        _engine.run();
        return ReadOutputs();
    }

    /// <summary>Calls the supplied learn method with its fixed rates and returns pre-update MSE.</summary>
    public double TrainStep(double[] x, double[] target)
    {
        if (target.Length != _outputSize)
            throw new ArgumentException("Target size must match the network.", nameof(target));
        SetInputs(x);
        for (int k = 0; k < _outputSize; k++) _engine.setDesiredOutput(k, target[k]);
        _engine.learn();
        // learn retains the activations computed before updating weights.
        return ReadOutputs().Select((value, k) => (value - target[k]) * (value - target[k])).Average();
    }

    /// <summary>Returns the class index with the highest activation.</summary>
    public int PredictClass(double[] x)
    {
        var outputs = Predict(x);
        int best = 0;
        for (int k = 1; k < outputs.Length; k++) if (outputs[k] > outputs[best]) best = k;
        return best;
    }

    private void SetInputs(double[] x)
    {
        if (x.Length != _inputSize) throw new ArgumentException("Input size must match the network.", nameof(x));
        for (int i = 0; i < _inputSize; i++) _engine.setInputs(i, x[i]);
    }

    private double[] ReadOutputs() => Enumerable.Range(0, _outputSize).Select(_engine.getOuputData).ToArray();
}
