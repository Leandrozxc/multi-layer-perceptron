# C# Multilayer Perceptron (MLP) From Scratch

A C# application using the Backprop neural network library for a 3-layer Multilayer Perceptron (Input, Hidden, Output) with backpropagation trained using Stochastic Gradient Descent (SGD) on the Fisher's Iris dataset.

> **Zero external ML libraries:** Built strictly with standard .NET 8 libraries (`System.*`). No ML.NET, TorchSharp, Math.NET, or external dependencies.

---

## Table of Contents

- [Dataset Description](#dataset-description)
- [Architecture & Mathematical Engine](#architecture--mathematical-engine)
  - [Network Layers](#network-layers)
  - [Forward Pass & Activation](#forward-pass--activation)
  - [Backpropagation & Weight Updates](#backpropagation--weight-updates)
  - [Preprocessing & Leak-Free Normalization](#preprocessing--leak-free-normalization)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
  - [Prerequisites](#prerequisites)
  - [Running the Sanity Test (XOR)](#running-the-sanity-test-xor)
  - [Running the End-to-End Iris Pipeline](#running-the-end-to-end-iris-pipeline)
- [Hyperparameter Sweep Results](#hyperparameter-sweep-results)
- [Final Model Evaluation](#final-model-evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [Classification Metrics](#classification-metrics)
  - [Training & Validation Loss Curve](#training--validation-loss-curve)

---

## Dataset Description

- **Source:** [UCI Machine Learning Repository - Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)
- **File location:** `Data/iris.csv`
- **Total instances:** 150 samples
- **Features (4 continuous numerical measurements in cm):**
  1. `sepal_length`
  2. `sepal_width`
  3. `petal_length`
  4. `petal_width`
- **Classes (3 classes, 50 samples each):**
  1. `Iris-setosa` (linearly separable from the other two)
  2. `Iris-versicolor`
  3. `Iris-virginica`
- **Split Ratio:** 60% training (90 samples) / 20% validation (30 samples) / 20% testing (30 samples) with fixed seed `42`.

---

## Architecture & Mathematical Engine

### Network Layers

```
[Input Layer: 4 Features]
           │
           ▼
[Hidden Layer: 4 Neurons (Sigmoid)]
           │
           ▼
[Output Layer: 3 Neurons (Sigmoid + One-Hot Classes)]
```

### Forward Pass & Activation

The supplied coordinator uses sigmoid activation with a positive-input saturation rule: preactivation values at least 20 are replaced with 32 before evaluating the formula below. Otherwise, ordinary sigmoid is used:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

For hidden neuron $j$:
$$\text{out}_j = \sigma\left(\sum_i \text{in}_i \cdot w_{ij} + b_j\right)$$

For output neuron $k$:
$$\text{out}_k = \sigma\left(\sum_j \text{out}_j \cdot w_{jk} + b_k\right)$$

### Backpropagation & Weight Updates

Training calls the supplied `Backprop.NeuralNet.learn()` unchanged. Its optimization objective is half the summed squared output error:

$$L = \frac{1}{2}\sum_{k=1}^K (\text{out}_k - \text{target}_k)^2$$

It stores negative preactivation gradients:

$$d_k = (\text{target}_k - \text{out}_k)\text{out}_k(1-\text{out}_k)$$
$$d_j = \text{out}_j(1-\text{out}_j)\sum_k d_k w_{jk}$$

Hidden errors are calculated before output weights change. The supplied neuron methods add the corresponding weight and bias updates. Learning rates are fixed by the supplied coordinator: **0.15** for input-to-hidden weights and hidden biases, **0.2** for hidden-to-output weights and output biases.

The application reports **MSE** for all partitions:

$$\mathrm{MSE} = \frac{1}{K}\sum_{k=1}^K (\text{out}_k - \text{target}_k)^2 = \frac{2}{K}L$$

Thus reporting and optimization objectives differ by a constant scale. The supplied derivative is correct for its half-sum objective; the application does not add MSE gradient scaling or alter the library's learning rule.

All four supplied files under `Libraries/` remain unchanged. `Network/NeuralNetwork.cs` delegates forward propagation to `run()` and training to `learn()`. For reproducibility, it writes seeded initial weights in the supplied persistence format and loads them once through `loadWeights()`, deleting the temporary file afterward. Initial weights are integer draws in `[-1000, 1000)` divided by 2000; hidden and output biases are `0.01`. This uses the coordinator's public API without accessing private fields.

### Preprocessing & Leak-Free Normalization

To prevent data leakage:
- Min and max bounds for each feature are calculated **strictly from the training partition** ($N = 90$).
- The validation and test sets are normalized using these exact training parameters:
  $$x_{\text{norm}} = \frac{x - \min_{\text{train}}}{\max_{\text{train}} - \min_{\text{train}}}$$
- Targets are one-hot encoded into 3-element vectors:
  - `Iris-setosa` $\rightarrow [1, 0, 0]$
  - `Iris-versicolor` $\rightarrow [0, 1, 0]$
  - `Iris-virginica` $\rightarrow [0, 0, 1]$

---

## Project Structure

```
.
├── Data/
│   ├── iris.csv            # 150 samples with CSV header
│   ├── CsvLoader.cs        # CSV parser into numeric features & labels
│   └── Preprocessor.cs     # Min-max normalization (train-only fit), one-hot, 60/20/20 split
├── Libraries/
│   ├── BackNeural 1.cs     # Supplied coordinator, preserved as provided
│   ├── InputN 1.cs         # Input neurons and input-to-hidden weights
│   ├── HiddenN 1.cs        # Hidden neurons, error propagation, output weights
│   └── OuputN 1.cs         # Output activations and biases
├── Network/
│   ├── InputLayer.cs       # Legacy layer source (excluded from compilation)
│   ├── HiddenLayer.cs      # Legacy layer source (excluded from compilation)
│   ├── OutputLayer.cs      # Legacy layer source (excluded from compilation)
│   └── NeuralNetwork.cs    # Adapter calling supplied NeuralNet.run() and learn()
├── Training/
│   └── Trainer.cs          # SGD epoch loop, training logger, SVG generator
├── Evaluation/
│   └── Metrics.cs          # Accuracy, confusion matrix, precision, recall, F1-score
├── Program.cs              # CLI runner, XOR sanity test, grid search, evaluation
├── Multi-layer perceptron.csproj
├── training_log.csv        # Epoch-by-epoch loss & accuracy history
├── tuning_results.csv      # Hyperparameter sweep logs (9 configurations)
├── loss_curve.svg          # Standalone vector graphic of the training loss curve
├── .gitignore              # Ignores bin/, obj/, user files
└── README.md
```

---

The supplied library sources are `Libraries/BackNeural 1.cs`, `Libraries/InputN 1.cs`, `Libraries/HiddenN 1.cs`, and `Libraries/OuputN 1.cs`. Application adaptation lives in `Network/NeuralNetwork.cs`; the supplied coordinator performs forward propagation and learning. The three legacy layer files under `Network/` are retained as source references and excluded from the build. Compiler warnings in the supplied library are retained to preserve its source.

## How to Run

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or higher

### Running the Sanity Test (XOR)

The XOR sanity check uses a 2-4-1 network, the supplied fixed rates, seed `42`, and 30,000 epochs. It requires MSE below `0.01` and all four predictions correct at a threshold of `0.5`. Failure returns exit code 1. The same four samples are used for training and validation in this sanity check.

```bash
dotnet run -- --xor
```

Recorded result: MSE `0.000295`, accuracy `100%`. Predictions for `[0,0]`, `[0,1]`, `[1,0]`, and `[1,1]` are `0.0156`, `0.9835`, `0.9842`, and `0.0203` respectively. The former 5,000-epoch limit did not converge with these fixed rates; 30,000 epochs allows this deterministic check to pass without changing the library.

### Running the End-to-End Iris Pipeline

Parses the bundled CSV, prepares the partitions, runs a 9-configuration hidden-size/epoch search, retrains the selected model, exports `training_log.csv` and `loss_curve.svg`, and prints final test metrics. CSV parsing is not a comprehensive dataset validation step.

```bash
dotnet run
```

---

## Hyperparameter Sweep Results

The search tests hidden sizes `{4, 8, 16}` and epoch counts `{500, 1000, 2000}`: nine configurations, all recorded in [tuning_results.csv](tuning_results.csv). Configurations use the same seed and split, and are ranked by validation accuracy, then lower validation MSE on ties. Test data is not evaluated during selection or epoch logging. Retraining uses only the same 90 training samples.

**Learning rates are not tuned:** the unchanged supplied `learn()` method fixes them at input rate `0.15` and output rate `0.2`. This preserves the supplied implementation but does not fulfill the rubric's Excellent-level expectation of experimenting with different learning rates. Hidden size and epoch count are systematically tested.

| Hidden neurons | Epochs | Train MSE | Validation MSE | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 500 | 0.009050 | 0.023030 | 96.7% |
| 4 | 1000 | 0.005510 | 0.028860 | 93.3% |
| 4 | 2000 | 0.004001 | 0.043982 | 90.0% |
| 8 | 500 | 0.009340 | 0.023121 | 96.7% |
| 8 | 1000 | 0.005550 | 0.028631 | 93.3% |
| 8 | 2000 | 0.004060 | 0.043972 | 90.0% |
| 16 | 500 | 0.009548 | 0.023366 | 96.7% |
| 16 | 1000 | 0.005659 | 0.028255 | 93.3% |
| 16 | 2000 | 0.004147 | 0.043021 | 90.0% |

The selected configuration has **4 hidden neurons and 500 epochs**, training accuracy **98.9%**, validation accuracy **96.7%**, training MSE **0.009050**, and validation MSE **0.023030**. All widths reach the same validation accuracy at 500 epochs. Four neurons has the lowest validation MSE, although the differences between widths are small and do not establish consistent superiority across seeds.

At four neurons, extending training from 500 to 2,000 epochs reduces training MSE from `0.009050` to `0.004001`, but increases validation MSE from `0.023030` to `0.043982` and reduces validation accuracy from 96.7% to 90.0%. This divergence is consistent with overfitting. The earlier tested endpoint is preferable under the implemented selection rule; this does not establish a globally optimal epoch count.

## Final Model Evaluation

After model selection, the 30 test samples yield **96.7% accuracy (29 correct)** and **MSE 0.008002**.

### Confusion Matrix

Rows are actual classes; columns are predicted classes.

| Actual / Predicted | Setosa | Versicolor | Virginica |
| --- | ---: | ---: | ---: |
| Setosa | 11 | 0 | 0 |
| Versicolor | 0 | 8 | 0 |
| Virginica | 0 | 1 | 10 |

### Classification Metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Setosa | 1.0000 | 1.0000 | 1.0000 | 11 |
| Versicolor | 0.8889 | 1.0000 | 0.9412 | 8 |
| Virginica | 1.0000 | 0.9091 | 0.9524 | 11 |
| Macro average | 0.9630 | 0.9697 | 0.9645 | 30 |

One virginica sample is predicted as versicolor, lowering virginica recall and versicolor precision. Setosa has no test errors. These counts do not establish a general pattern of errors across other samples or splits.

Accuracy uses the largest output, whereas MSE measures distance from one-hot targets. The independent sigmoid outputs do not necessarily sum to one and are not calibrated class probabilities.

### Training & Validation Loss Curve

[training_log.csv](training_log.csv) and [loss_curve.svg](loss_curve.svg) report training and validation metrics after each epoch, using the same fixed model weights and MSE function. Test MSE uses that function only after training.

Training MSE decreases overall from `0.223984` to `0.009050`; validation MSE goes from `0.224461` to `0.023030`. SGD causes fluctuations, so loss need not decrease every epoch.

### Learning behavior and limitations

- These are single runs on one seeded, shuffled 60/20/20 split, not averages across seeds. The split is not stratified.
- Each validation or test sample changes accuracy by about 3.3 percentage points. Test class supports are only 8 to 11 samples each.
- Comparing nine configurations on the same validation partition can make the winning validation score optimistic. Future work could use stratified cross-validation within development data.
- Earlier experiments used these test samples for tuning. The current pipeline excludes them from selection, but a stronger future assessment would use genuinely unseen data after fixing the procedure.
- The supplied fixed learning rates constrain experimentation. Any learning-rate study would require an approved alternative integration or permission to extend the supplied coordinator.
- Results concern Iris classification with four features and do not demonstrate performance on other datasets or readiness for real decision-support deployment.
