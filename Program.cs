using System.Globalization;
using Multi_layer_perceptron.Data;
using Multi_layer_perceptron.Evaluation;
using Multi_layer_perceptron.Network;
using Multi_layer_perceptron.Training;

if (args.Contains("--xor"))
{
    RunXorSanityTest();
    return;
}

Console.WriteLine("================================================================");
Console.WriteLine("        C# MLP FROM SCRATCH - IRIS CLASSIFICATION               ");
Console.WriteLine("================================================================\n");

string dataPath = Path.Combine(AppContext.BaseDirectory, "Data", "iris.csv");
if (!File.Exists(dataPath))
{
    dataPath = Path.Combine("Data", "iris.csv");
}

if (!File.Exists(dataPath))
{
    Console.WriteLine($"Dataset not found at {dataPath}. Downloading from UCI repository...");
    Directory.CreateDirectory("Data");
    using var http = new HttpClient();
    string rawData = http.GetStringAsync("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data").GetAwaiter().GetResult();
    var rawLines = rawData.Split('\n')
        .Select(l => l.Trim())
        .Where(l => !string.IsNullOrEmpty(l));
    File.WriteAllLines("Data/iris.csv", new[] { "sepal_length,sepal_width,petal_length,petal_width,species" }.Concat(rawLines));
    dataPath = "Data/iris.csv";
    Console.WriteLine("Dataset successfully downloaded and saved to Data/iris.csv\n");
}

// 1. Load and prepare dataset
Console.WriteLine($"Loading dataset from: {dataPath}");
var (X, labels) = CsvLoader.Load(dataPath);
var classes = labels.Distinct().OrderBy(l => l).ToArray();
Console.WriteLine($"Loaded {X.Length} samples with {X[0].Length} features across {classes.Length} classes: {string.Join(", ", classes)}");

var (trainX, trainY, validationX, validationY, testX, testY) = Preprocessor.Split(X, labels, classes, seed: 42);
Console.WriteLine($"Data split: {trainX.Length} training (60%), {validationX.Length} validation (20%), {testX.Length} testing (20%)\n");

// 2. Hyperparameter tuning sweep
Console.WriteLine("Starting hyperparameter grid search...");
string tuningResultsPath = "tuning_results.csv";
using (var sw = new StreamWriter(tuningResultsPath, false))
{
    sw.WriteLine("learning_rate,hidden_size,epochs,train_loss,validation_loss,train_acc,validation_acc");
}

var best = (acc: 0.0, loss: double.MaxValue, lr: 0.0, hidden: 0, epochs: 0);
var lrs = new[] { 0.01, 0.05, 0.1, 0.5 };
var hiddens = new[] { 4, 8, 16 };
var epochsArr = new[] { 500, 1000, 2000 };
int totalConfigs = lrs.Length * hiddens.Length * epochsArr.Length;
int configIdx = 0;

foreach (var lr in lrs)
foreach (var h in hiddens)
foreach (var ep in epochsArr)
{
    configIdx++;
    var net = new NeuralNetwork(4, h, 3, seed: 42);
    var result = Trainer.Train(net, trainX, trainY, validationX, validationY, lr, ep, logEvery: 0, verbose: false);

    // Append to tuning_results.csv
    using (var sw = new StreamWriter(tuningResultsPath, true))
    {
        sw.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "{0},{1},{2},{3:F6},{4:F6},{5:F6},{6:F6}",
            lr, h, ep, result.FinalTrainLoss, result.FinalValidationLoss, result.FinalTrainAcc, result.FinalValidationAcc));
    }

    bool isBetter = (result.FinalValidationAcc > best.acc) ||
                    (Math.Abs(result.FinalValidationAcc - best.acc) < 1e-9 && result.FinalValidationLoss < best.loss);

    if (isBetter)
    {
        best = (result.FinalValidationAcc, result.FinalValidationLoss, lr, h, ep);
    }

    Console.WriteLine(string.Format(
        CultureInfo.InvariantCulture,
        "[{0,2}/{1}] lr={2,4:F2}, hidden={3,2}, epochs={4,4} -> TrainAcc: {5,5:P1}, ValidationAcc: {6,5:P1}, ValidationLoss: {7:F5}{8}",
        configIdx, totalConfigs, lr, h, ep, result.FinalTrainAcc, result.FinalValidationAcc, result.FinalValidationLoss,
        isBetter ? " *" : ""));
}

Console.WriteLine($"\nGrid search complete. Results saved to: {tuningResultsPath}");
Console.WriteLine(string.Format(
    CultureInfo.InvariantCulture,
    "Best configuration: lr={0}, hidden={1}, epochs={2} (Validation Acc: {3:P1}, Validation Loss: {4:F5})\n",
    best.lr, best.hidden, best.epochs, best.acc, best.loss));

// 3. Retrain best model with full epoch logging and SVG visualization
Console.WriteLine("================================================================");
Console.WriteLine($" Retraining Best Model (lr={best.lr}, hidden={best.hidden}, epochs={best.epochs}) ");
Console.WriteLine("================================================================");

var bestNet = new NeuralNetwork(4, best.hidden, 3, seed: 42);
string logPath = "training_log.csv";
var finalResult = Trainer.Train(bestNet, trainX, trainY, validationX, validationY, best.lr, best.epochs, logEvery: 100, verbose: true, exportCsvPath: logPath);
Console.WriteLine($"\nEpoch log saved to: {logPath}");

string svgPath = "loss_curve.svg";
Trainer.ExportLossSvg(svgPath, finalResult.History);
Console.WriteLine($"Loss curve graphic exported to: {svgPath}");

// 4. Evaluate the untouched test partition only after model selection and training.
Console.WriteLine($"Final test MSE: {Metrics.MeanSquaredError(bestNet, testX, testY):F6}");
var report = Metrics.Evaluate(bestNet, testX, testY, classes);
report.Print();

static void RunXorSanityTest()
{
    Console.WriteLine("Running XOR sanity test (2-4-1, lr 0.5, 5000 epochs)...");
    var net = new NeuralNetwork(2, 4, 1, seed: 42);
    double[][] xorX = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    double[][] xorY = [[0.0], [1.0], [1.0], [0.0]];
    var result = Trainer.Train(net, xorX, xorY, xorX, xorY, lr: 0.5, epochs: 5000, logEvery: 1000, verbose: true);

    Console.WriteLine($"\nFinal XOR Loss: {result.FinalTrainLoss:F6} (Threshold: < 0.01)");
    for (int i = 0; i < 4; i++)
    {
        double pred = net.Predict(xorX[i])[0];
        Console.WriteLine($"Input: [{xorX[i][0]}, {xorX[i][1]}] -> Target: {xorY[i][0]}, Prediction: {pred:F4}");
    }

    double accuracy = Metrics.Accuracy(net, xorX, xorY);
    Console.WriteLine($"Final XOR Accuracy: {accuracy:P1} (Threshold: 0.5)");
    if (result.FinalTrainLoss < 0.01 && accuracy == 1.0)
    {
        Console.WriteLine("\n[PASS] XOR sanity test passed successfully!\n");
    }
    else
    {
        Console.WriteLine("\n[FAIL] XOR sanity test requires loss < 0.01 and all four predictions correct.\n");
        Environment.ExitCode = 1;
    }
}
