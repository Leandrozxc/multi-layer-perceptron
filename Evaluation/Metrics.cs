namespace Multi_layer_perceptron.Evaluation;

using Multi_layer_perceptron.Network;

/// <summary>
/// Holds metrics for an individual class.
/// </summary>
public record ClassMetric(string ClassName, double Precision, double Recall, double F1, int Support);

/// <summary>
/// Encapsulates comprehensive multi-class evaluation metrics.
/// </summary>
public class ClassificationReport
{
    /// <summary>
    /// Gets the class names.
    /// </summary>
    public string[] Classes { get; }

    /// <summary>
    /// Gets the confusion matrix [actual, predicted].
    /// </summary>
    public int[,] ConfusionMatrix { get; }

    /// <summary>
    /// Gets overall classification accuracy.
    /// </summary>
    public double Accuracy { get; }

    /// <summary>
    /// Gets the list of per-class metrics.
    /// </summary>
    public IReadOnlyList<ClassMetric> ClassMetrics { get; }

    /// <summary>
    /// Gets the unweighted macro-average precision.
    /// </summary>
    public double MacroPrecision => ClassMetrics.Count == 0 ? 0.0 : ClassMetrics.Average(c => c.Precision);

    /// <summary>
    /// Gets the unweighted macro-average recall.
    /// </summary>
    public double MacroRecall => ClassMetrics.Count == 0 ? 0.0 : ClassMetrics.Average(c => c.Recall);

    /// <summary>
    /// Gets the unweighted macro-average F1 score.
    /// </summary>
    public double MacroF1 => ClassMetrics.Count == 0 ? 0.0 : ClassMetrics.Average(c => c.F1);

    /// <summary>
    /// Initializes a new instance of the <see cref="ClassificationReport"/> class.
    /// </summary>
    public ClassificationReport(string[] classes, int[,] confusionMatrix, double accuracy, IReadOnlyList<ClassMetric> classMetrics)
    {
        Classes = classes;
        ConfusionMatrix = confusionMatrix;
        Accuracy = accuracy;
        ClassMetrics = classMetrics;
    }

    /// <summary>
    /// Prints the confusion matrix and evaluation report to the console in a structured table.
    /// </summary>
    public void Print()
    {
        Console.WriteLine("\n======================= CONFUSION MATRIX =======================");
        Console.Write($"{"Actual \\ Predicted",-20}");
        foreach (var cls in Classes)
        {
            Console.Write($"{cls,16}");
        }
        Console.WriteLine();
        Console.WriteLine(new string('-', 20 + Classes.Length * 16));

        for (int r = 0; r < Classes.Length; r++)
        {
            Console.Write($"{Classes[r],-20}");
            for (int c = 0; c < Classes.Length; c++)
            {
                Console.Write($"{ConfusionMatrix[r, c],16}");
            }
            Console.WriteLine();
        }

        Console.WriteLine("\n==================== CLASSIFICATION METRICS ====================");
        Console.WriteLine($"{"Class",-20}{"Precision",12}{"Recall",12}{"F1-Score",12}{"Support",10}");
        Console.WriteLine(new string('-', 66));
        foreach (var cm in ClassMetrics)
        {
            Console.WriteLine($"{cm.ClassName,-20}{cm.Precision,12:F4}{cm.Recall,12:F4}{cm.F1,12:F4}{cm.Support,10}");
        }
        Console.WriteLine(new string('-', 66));
        Console.WriteLine($"{"Accuracy",-20}{Accuracy,36:F4}{ClassMetrics.Sum(c => c.Support),10}");
        Console.WriteLine($"{"Macro Avg",-20}{MacroPrecision,12:F4}{MacroRecall,12:F4}{MacroF1,12:F4}{ClassMetrics.Sum(c => c.Support),10}");
        Console.WriteLine("================================================================\n");
    }
}

/// <summary>
/// Provides evaluation utilities for calculating classification metrics.
/// </summary>
public static class Metrics
{
    /// <summary>
    /// Evaluates the neural network predictions on a dataset against one-hot encoded ground truth.
    /// </summary>
    /// <param name="net">The trained neural network model.</param>
    /// <param name="X">The input feature matrix.</param>
    /// <param name="Y">The one-hot encoded ground truth matrix.</param>
    /// <param name="classNames">The array of class labels.</param>
    /// <returns>A structured <see cref="ClassificationReport"/>.</returns>
    public static ClassificationReport Evaluate(NeuralNetwork net, double[][] X, double[][] Y, string[] classNames)
    {
        int numClasses = classNames.Length;
        int[,] matrix = new int[numClasses, numClasses];
        int correctCount = 0;

        for (int i = 0; i < X.Length; i++)
        {
            int predicted = net.PredictClass(X[i]);
            int actual = ArgMax(Y[i]);

            matrix[actual, predicted]++;
            if (predicted == actual)
            {
                correctCount++;
            }
        }

        double accuracy = X.Length == 0 ? 0.0 : (double)correctCount / X.Length;
        var classMetrics = new List<ClassMetric>();

        for (int c = 0; c < numClasses; c++)
        {
            int tp = matrix[c, c];
            int fp = 0;
            int fn = 0;
            int support = 0;

            for (int r = 0; r < numClasses; r++)
            {
                if (r != c) fp += matrix[r, c];
                support += matrix[c, r];
                if (r != c) fn += matrix[c, r];
            }

            double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
            double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
            double f1 = (precision + recall) > 0 ? (2.0 * precision * recall) / (precision + recall) : 0.0;

            classMetrics.Add(new ClassMetric(classNames[c], precision, recall, f1, support));
        }

        return new ClassificationReport(classNames, matrix, accuracy, classMetrics);
    }

    /// <summary>
    /// Calculates accuracy for a set of samples.
    /// </summary>
    /// <param name="net">The neural network model.</param>
    /// <param name="X">Input features.</param>
    /// <param name="Y">One-hot targets.</param>
    /// <returns>The proportion of correctly classified instances.</returns>
    public static double Accuracy(NeuralNetwork net, double[][] X, double[][] Y)
    {
        if (X.Length == 0) return 0.0;
        int correct = 0;
        for (int i = 0; i < X.Length; i++)
        {
            if (net.PredictClass(X[i]) == ArgMax(Y[i]))
            {
                correct++;
            }
        }
        return (double)correct / X.Length;
    }

    /// <summary>
    /// Calculates mean squared error loss over a dataset.
    /// </summary>
    /// <param name="net">The neural network model.</param>
    /// <param name="X">Input features.</param>
    /// <param name="Y">Target vectors.</param>
    /// <returns>Average MSE over all samples.</returns>
    public static double MeanSquaredError(NeuralNetwork net, double[][] X, double[][] Y)
    {
        if (X.Length == 0) return 0.0;
        double totalLoss = 0.0;
        for (int i = 0; i < X.Length; i++)
        {
            var pred = net.Predict(X[i]);
            double sampleLoss = 0.0;
            for (int k = 0; k < pred.Length; k++)
            {
                double diff = pred[k] - Y[i][k];
                sampleLoss += diff * diff;
            }
            totalLoss += sampleLoss / pred.Length;
        }
        return totalLoss / X.Length;
    }

    private static int ArgMax(double[] values)
    {
        int bestIdx = 0;
        double maxVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > maxVal)
            {
                maxVal = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}
