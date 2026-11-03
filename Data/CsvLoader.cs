using System.Globalization;

namespace Multi_layer_perceptron.Data;

/// <summary>
/// Provides functionality to load and parse dataset CSV files.
/// </summary>
public static class CsvLoader
{
    /// <summary>
    /// Loads a CSV dataset containing numeric features and a trailing class label column.
    /// </summary>
    /// <param name="filePath">The file path of the CSV file.</param>
    /// <returns>A tuple containing feature vectors and corresponding string class labels.</returns>
    /// <exception cref="FileNotFoundException">Thrown when the CSV file does not exist.</exception>
    /// <exception cref="InvalidOperationException">Thrown when no valid data rows are found.</exception>
    public static (double[][] features, string[] labels) Load(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"CSV dataset file not found at: {filePath}");
        }

        var lines = File.ReadAllLines(filePath)
            .Where(line => !string.IsNullOrWhiteSpace(line))
            .Select(line => line.Trim())
            .ToList();

        if (lines.Count == 0)
        {
            throw new InvalidOperationException("The dataset file is empty.");
        }

        int startIndex = 0;
        // Check if the first line is a header by trying to parse the first token
        string[] firstTokens = lines[0].Split(',');
        if (!double.TryParse(firstTokens[0], NumberStyles.Any, CultureInfo.InvariantCulture, out _))
        {
            startIndex = 1;
        }

        var featureList = new List<double[]>();
        var labelList = new List<string>();

        for (int i = startIndex; i < lines.Count; i++)
        {
            string[] parts = lines[i].Split(',');
            if (parts.Length < 2)
            {
                continue;
            }

            int featureCount = parts.Length - 1;
            double[] rowFeatures = new double[featureCount];
            for (int f = 0; f < featureCount; f++)
            {
                rowFeatures[f] = double.Parse(parts[f], CultureInfo.InvariantCulture);
            }

            string label = parts[^1].Trim();
            featureList.Add(rowFeatures);
            labelList.Add(label);
        }

        return (featureList.ToArray(), labelList.ToArray());
    }
}
