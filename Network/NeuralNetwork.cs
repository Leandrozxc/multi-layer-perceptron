using Backprop;

namespace Multi_layer_perceptron.Network;

/// <summary>Adapts the Backprop engine to the training and evaluation pipeline.</summary>
public class NeuralNetwork
{
    private readonly NeuralNet _engine;
    private readonly int _inputSize;
    private readonly int _outputSize;

    /// <summary>Creates a reproducible three-layer network using Backprop initialization.</summary>
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int seed = 42)
    {
        _inputSize = inputSize;
        _outputSize = outputSize;
        _engine = new NeuralNet(inputSize, hiddenSize, outputSize, seed);
    }

    /// <summary>Runs the engine and returns a snapshot of its output activations.</summary>
    public double[] Predict(double[] x)
    {
        SetInputs(x);
        _engine.run();
        return ReadOutputs();
    }

    /// <summary>Trains on one sample and returns its pre-update mean squared error.</summary>
    public double TrainStep(double[] x, double[] target, double lr)
    {
        if (target.Length != _outputSize)
            throw new ArgumentException("Target size must match the network.", nameof(target));
        _engine.SetLearningRates(lr, lr);
        SetInputs(x);
        for (int k = 0; k < _outputSize; k++)
            _engine.setDesiredOutput(k, target[k]);
        _engine.learn();
        // learn() retains the activations from its forward pass, before updating weights.
        var outputs = ReadOutputs();
        double sum = 0;
        for (int k = 0; k < _outputSize; k++)
        {
            double error = outputs[k] - target[k];
            sum += error * error;
        }
        return sum / _outputSize;
    }

    /// <summary>Returns the class index with the highest activation.</summary>
    public int PredictClass(double[] x)
    {
        var outputs = Predict(x);
        int best = 0;
        for (int k = 1; k < outputs.Length; k++)
            if (outputs[k] > outputs[best]) best = k;
        return best;
    }

    private void SetInputs(double[] x)
    {
        if (x.Length != _inputSize)
            throw new ArgumentException("Input size must match the network.", nameof(x));
        for (int i = 0; i < _inputSize; i++)
            _engine.setInputs(i, x[i]);
    }

    private double[] ReadOutputs()
    {
        var outputs = new double[_outputSize];
        for (int k = 0; k < outputs.Length; k++)
            outputs[k] = _engine.getOuputData(k);
        return outputs;
    }
}
