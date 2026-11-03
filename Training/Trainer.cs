using System.Globalization;
using System.Text;
using Multi_layer_perceptron.Evaluation;
using Multi_layer_perceptron.Network;

namespace Multi_layer_perceptron.Training;

/// <summary>
/// Represents performance metrics recorded for a single training epoch.
/// </summary>
public record EpochLog(int Epoch, double TrainLoss, double ValidationLoss, double TrainAcc, double ValidationAcc);

/// <summary>
/// Encapsulates the summary of a full training session.
/// </summary>
public record TrainingResult(
    double FinalTrainLoss,
    double FinalValidationLoss,
    double FinalTrainAcc,
    double FinalValidationAcc,
    List<EpochLog> History);

/// <summary>
/// Coordinates the training loop, loss logging, and visualization export.
/// </summary>
public static class Trainer
{
    /// <summary>
    /// Executes the epoch training loop for the neural network using stochastic gradient descent.
    /// </summary>
    /// <param name="net">The neural network to train.</param>
    /// <param name="trainX">Training feature vectors.</param>
    /// <param name="trainY">Training one-hot target vectors.</param>
    /// <param name="validationX">Validation feature vectors.</param>
    /// <param name="validationY">Validation one-hot target vectors.</param>
    /// <param name="lr">Learning rate for weight updates.</param>
    /// <param name="epochs">Total number of epochs to train.</param>
    /// <param name="logEvery">Interval in epochs at which progress is printed to console.</param>
    /// <param name="verbose">Whether to print epoch progress to the console.</param>
    /// <param name="exportCsvPath">Optional file path to export epoch-by-epoch training logs.</param>
    /// <returns>A <see cref="TrainingResult"/> containing final metrics and full epoch history.</returns>
    public static TrainingResult Train(
        NeuralNetwork net,
        double[][] trainX,
        double[][] trainY,
        double[][] validationX,
        double[][] validationY,
        double lr,
        int epochs,
        int logEvery = 100,
        bool verbose = true,
        string? exportCsvPath = null)
    {
        var history = new List<EpochLog>(epochs);
        int[] indices = Enumerable.Range(0, trainX.Length).ToArray();
        var rng = new Random(42);

        for (int ep = 1; ep <= epochs; ep++)
        {
            // Per-epoch Fisher-Yates shuffle of training data order (SGD requirement)
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            // Train over each sample
            for (int i = 0; i < indices.Length; i++)
            {
                int idx = indices[i];
                net.TrainStep(trainX[idx], trainY[idx], lr);
            }
            // Evaluate both partitions using the same fixed end-of-epoch weights.
            double meanTrainLoss = Metrics.MeanSquaredError(net, trainX, trainY);

            // Evaluate validation loss and accuracies
            double meanValidationLoss = Metrics.MeanSquaredError(net, validationX, validationY);
            double trainAcc = Metrics.Accuracy(net, trainX, trainY);
            double validationAcc = Metrics.Accuracy(net, validationX, validationY);

            var logEntry = new EpochLog(ep, meanTrainLoss, meanValidationLoss, trainAcc, validationAcc);
            history.Add(logEntry);

            if (verbose && logEvery > 0 && (ep % logEvery == 0 || ep == epochs))
            {
                Console.WriteLine(string.Format(
                    CultureInfo.InvariantCulture,
                    "Epoch {0,4}/{1} | Train Loss: {2:F5} | Validation Loss: {3:F5} | Train Acc: {4,5:P1} | Validation Acc: {5,5:P1}",
                    ep, epochs, meanTrainLoss, meanValidationLoss, trainAcc, validationAcc));
            }
        }

        if (!string.IsNullOrEmpty(exportCsvPath))
        {
            ExportLog(exportCsvPath, history);
        }

        var last = history[^1];
        return new TrainingResult(last.TrainLoss, last.ValidationLoss, last.TrainAcc, last.ValidationAcc, history);
    }

    /// <summary>
    /// Exports epoch training metrics to a CSV file.
    /// </summary>
    /// <param name="filePath">Target destination path.</param>
    /// <param name="history">Collection of epoch log records.</param>
    public static void ExportLog(string filePath, IEnumerable<EpochLog> history)
    {
        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        using var writer = new StreamWriter(filePath, false, Encoding.UTF8);
        writer.WriteLine("epoch,train_loss,validation_loss,train_acc,validation_acc");
        foreach (var entry in history)
        {
            writer.WriteLine(string.Format(
                CultureInfo.InvariantCulture,
                "{0},{1:F6},{2:F6},{3:F6},{4:F6}",
                entry.Epoch, entry.TrainLoss, entry.ValidationLoss, entry.TrainAcc, entry.ValidationAcc));
        }
    }

    /// <summary>
    /// Generates a clean SVG plot of the train and validation loss curves without any external libraries.
    /// </summary>
    /// <param name="filePath">Output SVG file path.</param>
    /// <param name="history">List of epoch logs containing loss values.</param>
    public static void ExportLossSvg(string filePath, IReadOnlyList<EpochLog> history)
    {
        if (history.Count == 0) return;

        int width = 800;
        int height = 460;
        int padLeft = 70;
        int padRight = 30;
        int padTop = 50;
        int padBottom = 50;

        int plotWidth = width - padLeft - padRight;
        int plotHeight = height - padTop - padBottom;

        double maxLoss = history.Max(h => Math.Max(h.TrainLoss, h.ValidationLoss));
        double minLoss = 0.0;
        if (maxLoss <= minLoss) maxLoss = 1.0;

        int totalEpochs = history.Count;

        string FormatPoint(int epoch, double loss)
        {
            double x = padLeft + ((double)(epoch - 1) / Math.Max(1, totalEpochs - 1)) * plotWidth;
            double normY = (loss - minLoss) / (maxLoss - minLoss);
            double y = padTop + (1.0 - normY) * plotHeight;
            return string.Format(CultureInfo.InvariantCulture, "{0:F1},{1:F1}", x, y);
        }

        var trainPoints = string.Join(" ", history.Select(h => FormatPoint(h.Epoch, h.TrainLoss)));
        var validationPoints = string.Join(" ", history.Select(h => FormatPoint(h.Epoch, h.ValidationLoss)));

        var sb = new StringBuilder();
        sb.AppendLine("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
        sb.AppendLine($"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" width=\"100%\" height=\"100%\" style=\"background:#ffffff; font-family:sans-serif;\">");

        // Title
        sb.AppendLine($"  <text x=\"{width / 2}\" y=\"30\" text-anchor=\"middle\" font-size=\"16\" font-weight=\"bold\" fill=\"#1e293b\">Iris MLP Training &amp; Validation Loss Curve</text>");

        // Grid lines & Y-axis ticks
        int gridSteps = 5;
        for (int i = 0; i <= gridSteps; i++)
        {
            double val = minLoss + (maxLoss - minLoss) * i / gridSteps;
            double y = padTop + (1.0 - (double)i / gridSteps) * plotHeight;
            sb.AppendLine($"  <line x1=\"{padLeft}\" y1=\"{y:F1}\" x2=\"{padLeft + plotWidth}\" y2=\"{y:F1}\" stroke=\"#e2e8f0\" stroke-width=\"1\" />");
            sb.AppendLine($"  <text x=\"{padLeft - 10}\" y=\"{y + 4:F1}\" text-anchor=\"end\" font-size=\"11\" fill=\"#64748b\">{val:F3}</text>");
        }

        // X-axis ticks
        int xTicks = 5;
        for (int i = 0; i <= xTicks; i++)
        {
            int epochVal = 1 + (int)Math.Round((double)(totalEpochs - 1) * i / xTicks);
            double x = padLeft + ((double)i / xTicks) * plotWidth;
            sb.AppendLine($"  <line x1=\"{x:F1}\" y1=\"{padTop}\" x2=\"{x:F1}\" y2=\"{padTop + plotHeight}\" stroke=\"#e2e8f0\" stroke-width=\"1\" stroke-dasharray=\"3,3\" />");
            sb.AppendLine($"  <text x=\"{x:F1}\" y=\"{padTop + plotHeight + 20}\" text-anchor=\"middle\" font-size=\"11\" fill=\"#64748b\">{epochVal}</text>");
        }

        // Axes
        sb.AppendLine($"  <line x1=\"{padLeft}\" y1=\"{padTop}\" x2=\"{padLeft}\" y2=\"{padTop + plotHeight}\" stroke=\"#475569\" stroke-width=\"2\" />");
        sb.AppendLine($"  <line x1=\"{padLeft}\" y1=\"{padTop + plotHeight}\" x2=\"{padLeft + plotWidth}\" y2=\"{padTop + plotHeight}\" stroke=\"#475569\" stroke-width=\"2\" />");

        // Axis labels
        sb.AppendLine($"  <text x=\"{padLeft + plotWidth / 2}\" y=\"{height - 12}\" text-anchor=\"middle\" font-size=\"12\" fill=\"#334155\">Epoch</text>");
        sb.AppendLine($"  <text x=\"18\" y=\"{padTop + plotHeight / 2}\" text-anchor=\"middle\" transform=\"rotate(-90 18 {padTop + plotHeight / 2})\" font-size=\"12\" fill=\"#334155\">Loss (MSE)</text>");

        // Polylines
        sb.AppendLine($"  <polyline fill=\"none\" stroke=\"#2563eb\" stroke-width=\"2.5\" points=\"{trainPoints}\" />");
        sb.AppendLine($"  <polyline fill=\"none\" stroke=\"#dc2626\" stroke-width=\"2.5\" stroke-dasharray=\"5,3\" points=\"{validationPoints}\" />");

        // Legend
        int legendX = width - padRight - 220;
        int legendY = padTop + 15;
        sb.AppendLine($"  <rect x=\"{legendX - 10}\" y=\"{legendY - 12}\" width=\"220\" height=\"52\" fill=\"#f8fafc\" stroke=\"#cbd5e1\" rx=\"4\" />");
        sb.AppendLine($"  <line x1=\"{legendX}\" y1=\"{legendY}\" x2=\"{legendX + 30}\" y2=\"{legendY}\" stroke=\"#2563eb\" stroke-width=\"2.5\" />");
        sb.AppendLine($"  <text x=\"{legendX + 38}\" y=\"{legendY + 4}\" font-size=\"12\" fill=\"#1e293b\">Train Loss</text>");
        sb.AppendLine($"  <line x1=\"{legendX}\" y1=\"{legendY + 22}\" x2=\"{legendX + 30}\" y2=\"{legendY + 22}\" stroke=\"#dc2626\" stroke-width=\"2.5\" stroke-dasharray=\"5,3\" />");
        sb.AppendLine($"  <text x=\"{legendX + 38}\" y=\"{legendY + 26}\" font-size=\"12\" fill=\"#1e293b\">Validation Loss</text>");

        sb.AppendLine("</svg>");

        string? dir = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        File.WriteAllText(filePath, sb.ToString(), Encoding.UTF8);
    }
}

