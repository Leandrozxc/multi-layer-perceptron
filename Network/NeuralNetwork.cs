using Backprop;

namespace Multi_layer_perceptron.Network;

/// <summary>Coordinates the supplied Backprop neuron classes without modifying their source.</summary>
public class NeuralNetwork
{
    private readonly INeuron[] _inputs;
    private readonly HNeuron[] _hidden;
    private readonly ONeuron[] _outputs;

    /// <summary>Creates a seeded network using the supplied neurons and their weight setters.</summary>
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int seed = 42)
    {
        if (inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputSize), "Layer sizes must be positive.");
        _inputs = Enumerable.Range(0, inputSize).Select(i => new INeuron(i, hiddenSize)).ToArray();
        _hidden = Enumerable.Range(0, hiddenSize).Select(j => new HNeuron(j, outputSize)).ToArray();
        _outputs = Enumerable.Range(0, outputSize).Select(k => new ONeuron(k)).ToArray();
        var random = new Random(seed);
        foreach (var input in _inputs)
            for (int j = 0; j < hiddenSize; j++)
                input.setWeight(j, random.Next(-1000, 1000) / 2000.0);
        foreach (var hidden in _hidden)
            for (int k = 0; k < outputSize; k++)
                hidden.setWeight(k, random.Next(-1000, 1000) / 2000.0);
    }

    /// <summary>Runs forward propagation through the supplied neurons and returns output values.</summary>
    public double[] Predict(double[] x)
    {
        if (x.Length != _inputs.Length)
            throw new ArgumentException("Input size must match the network.", nameof(x));
        for (int i = 0; i < _inputs.Length; i++)
            _inputs[i].setInput(x[i]);
        for (int j = 0; j < _hidden.Length; j++)
        {
            double sum = 0;
            foreach (var input in _inputs)
                sum += input.getInput() * input.getWeight(j);
            _hidden[j].setHactivation(Sigmoid(sum + _hidden[j].getBias()));
        }
        for (int k = 0; k < _outputs.Length; k++)
        {
            double sum = 0;
            foreach (var hidden in _hidden)
                sum += hidden.getHactivation() * hidden.getWeight(k);
            _outputs[k].setOActivation(Sigmoid(sum + _outputs[k].getBias()));
        }
        return _outputs.Select(output => output.getOActivation()).ToArray();
    }

    /// <summary>Applies one MSE gradient update through the supplied neuron methods.</summary>
    public double TrainStep(double[] x, double[] target, double lr)
    {
        if (target.Length != _outputs.Length)
            throw new ArgumentException("Target size must match the network.", nameof(target));
        if (!double.IsFinite(lr) || lr <= 0)
            throw new ArgumentOutOfRangeException(nameof(lr));
        var outputs = Predict(x);
        var derivatives = new double[outputs.Length];
        double sum = 0;
        for (int k = 0; k < outputs.Length; k++)
        {
            double error = target[k] - outputs[k];
            derivatives[k] = (2.0 / outputs.Length) * outputs[k] * (1 - outputs[k]) * error;
            sum += error * error;
        }
        // Compute hidden errors before any output connection weights change.
        foreach (var hidden in _hidden) hidden.calculateErr(derivatives);
        foreach (var hidden in _hidden) hidden.setWeight(lr, derivatives);
        foreach (var input in _inputs)
            for (int j = 0; j < _hidden.Length; j++)
                input.setWeight(j, _hidden[j].getErr(), lr);
        foreach (var output in _outputs) output.changeBias(lr, derivatives);
        foreach (var hidden in _hidden) hidden.changeBias(lr);
        return sum / outputs.Length;
    }

    /// <summary>Returns the index of the highest output activation.</summary>
    public int PredictClass(double[] x)
    {
        var outputs = Predict(x);
        int best = 0;
        for (int k = 1; k < outputs.Length; k++)
            if (outputs[k] > outputs[best]) best = k;
        return best;
    }

    // Match the supplied NeuralNet activation rule, including positive saturation.
    private static double Sigmoid(double value)
    {
        if (value >= 20.0) value = 32;
        return 1.0 / (1.0 + Math.Exp(-value));
    }
}
