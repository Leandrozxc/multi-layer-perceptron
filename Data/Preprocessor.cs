namespace Multi_layer_perceptron.Data;

/// <summary>
/// Handles data preprocessing including shuffling, train/validation/test splitting, min-max normalization, and one-hot encoding.
/// </summary>
public static class Preprocessor
{
    /// <summary>
    /// Creates disjoint seeded training, validation, and test partitions.
    /// Normalization bounds are fitted exclusively on training features.
    /// </summary>
    /// <param name="X">Raw feature vectors.</param>
    /// <param name="labels">Class labels corresponding to feature vectors.</param>
    /// <param name="classes">Class names used for one-hot encoding.</param>
    /// <param name="trainRatio">Training fraction.</param>
    /// <param name="validationRatio">Validation fraction.</param>
    /// <param name="seed">Shuffle seed.</param>
    /// <returns>Normalized features and one-hot targets for all three partitions.</returns>
    public static (double[][] trainX, double[][] trainY,
        double[][] validationX, double[][] validationY, double[][] testX, double[][] testY) Split(
        double[][] X, string[] labels, string[] classes,
        double trainRatio = 0.6, double validationRatio = 0.2, int seed = 42)
    {
        if (X.Length != labels.Length || X.Length == 0)
            throw new ArgumentException("Features and labels must have matching nonzero lengths.");
        if (!double.IsFinite(trainRatio) || !double.IsFinite(validationRatio) ||
            trainRatio <= 0 || validationRatio <= 0 || trainRatio + validationRatio >= 1)
            throw new ArgumentException("Split ratios must leave nonempty training, validation, and test fractions.");
        int featureCount = X[0].Length;
        if (featureCount == 0 || X.Any(row => row.Length != featureCount))
            throw new ArgumentException("Feature vectors must have a consistent nonzero width.");

        int trainCount = (int)Math.Round(X.Length * trainRatio);
        int validationCount = (int)Math.Round(X.Length * validationRatio);
        if (trainCount == 0 || validationCount == 0 || trainCount + validationCount >= X.Length)
            throw new ArgumentException("Each partition must contain at least one sample.");

        int[] indices = Enumerable.Range(0, X.Length).ToArray();
        var rng = new Random(seed);
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var trainingIndices = indices.Take(trainCount).ToArray();
        var min = Enumerable.Range(0, featureCount)
            .Select(f => trainingIndices.Min(i => X[i][f])).ToArray();
        var max = Enumerable.Range(0, featureCount)
            .Select(f => trainingIndices.Max(i => X[i][f])).ToArray();

        (double[][] features, double[][] targets) Prepare(IEnumerable<int> partition)
        {
            var rows = partition.ToArray();
            var features = rows.Select(i => Enumerable.Range(0, featureCount).Select(f =>
                max[f] - min[f] <= 1e-12 ? 0.0 : (X[i][f] - min[f]) / (max[f] - min[f])).ToArray()).ToArray();
            var targets = rows.Select(i => OneHotEncode(labels[i], classes)).ToArray();
            return (features, targets);
        }

        var train = Prepare(trainingIndices);
        var validation = Prepare(indices.Skip(trainCount).Take(validationCount));
        var test = Prepare(indices.Skip(trainCount + validationCount));
        return (train.features, train.targets, validation.features, validation.targets, test.features, test.targets);
    }

    /// <summary>
    /// Converts a class label into a one-hot encoded double array.
    /// </summary>
    /// <param name="label">The string class label.</param>
    /// <param name="classes">The sorted array of all unique class labels.</param>
    /// <returns>A one-hot encoded vector representing the class.</returns>
    public static double[] OneHotEncode(string label, string[] classes)
    {
        double[] encoded = new double[classes.Length];
        int index = Array.IndexOf(classes, label);
        if (index >= 0)
        {
            encoded[index] = 1.0;
        }
        else
        {
            throw new ArgumentException($"Label '{label}' not recognized among classes.");
        }
        return encoded;
    }
}

