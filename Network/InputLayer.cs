namespace Multi_layer_perceptron.Network;

/// <summary>
/// Represents the input layer of the neural network holding input features.
/// </summary>
public class InputLayer
{
    /// <summary>
    /// Gets or sets the input feature values.
    /// </summary>
    public double[] Values { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="InputLayer"/> class.
    /// </summary>
    /// <param name="size">The number of input features.</param>
    public InputLayer(int size)
    {
        Values = new double[size];
    }
}

