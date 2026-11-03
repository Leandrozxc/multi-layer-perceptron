namespace Multi_layer_perceptron.Data;

/// <summary>
/// Handles data preprocessing including shuffling, train/test splitting, min-max normalization, and one-hot encoding.
/// </summary>
public static class Preprocessor
{
    /// <summary>
    /// Shuffles and splits features and labels into training and test sets, applying min-max normalization
    /// fit solely on the training set to prevent data leakage, and one-hot encoding class labels.
    /// </summary>
    /// <param name="X">The raw feature matrix.</param>
    /// <param name="labels">The class label strings for each sample.</param>
    /// <param name="classes">The distinct sorted class labels.</param>
    /// <param name="trainRatio">The fraction of samples to allocate to the training set (default: 0.8).</param>
    /// <param name="seed">Random seed for reproducible shuffling.</param>
    /// <returns>Normalized train features, one-hot train targets, normalized test features, and one-hot test targets.</returns>
    public static (double[][] trainX, double[][] trainY, double[][] testX, double[][] testY) Split(
        double[][] X,
        string[] labels,
        string[] classes,
        double trainRatio = 0.8,
        int seed = 42)
    {
        if (X.Length != labels.Length)
        {
            throw new ArgumentException("Feature count and label count must match.");
        }

        int totalSamples = X.Length;
        int featureCount = X[0].Length;

        // 1. Deterministic Fisher-Yates shuffle
        int[] indices = Enumerable.Range(0, totalSamples).ToArray();
        var rng = new Random(seed);
        for (int i = totalSamples - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // 2. Train / test index partition
        int trainCount = (int)Math.Round(totalSamples * trainRatio);
        int testCount = totalSamples - trainCount;

        var rawTrainX = new double[trainCount][];
        var trainY = new double[trainCount][];
        for (int i = 0; i < trainCount; i++)
        {
            int idx = indices[i];
            rawTrainX[i] = (double[])X[idx].Clone();
            trainY[i] = OneHotEncode(labels[idx], classes);
        }

        var rawTestX = new double[testCount][];
        var testY = new double[testCount][];
        for (int i = 0; i < testCount; i++)
        {
            int idx = indices[trainCount + i];
            rawTestX[i] = (double[])X[idx].Clone();
            testY[i] = OneHotEncode(labels[idx], classes);
        }

        // 3. Compute Min and Max on training set ONLY (preventing data leakage)
        double[] minVals = new double[featureCount];
        double[] maxVals = new double[featureCount];
        for (int f = 0; f < featureCount; f++)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            for (int i = 0; i < trainCount; i++)
            {
                double val = rawTrainX[i][f];
                if (val < min) min = val;
                if (val > max) max = val;
            }
            minVals[f] = min;
            maxVals[f] = max;
        }

        // 4. Normalize both sets with train min/max
        var trainX = new double[trainCount][];
        for (int i = 0; i < trainCount; i++)
        {
            trainX[i] = new double[featureCount];
            for (int f = 0; f < featureCount; f++)
            {
                double range = maxVals[f] - minVals[f];
                trainX[i][f] = range > 1e-12 ? (rawTrainX[i][f] - minVals[f]) / range : 0.0;
            }
        }

        var testX = new double[testCount][];
        for (int i = 0; i < testCount; i++)
        {
            testX[i] = new double[featureCount];
            for (int f = 0; f < featureCount; f++)
            {
                double range = maxVals[f] - minVals[f];
                testX[i][f] = range > 1e-12 ? (rawTestX[i][f] - minVals[f]) / range : 0.0;
            }
        }

        return (trainX, trainY, testX, testY);
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
