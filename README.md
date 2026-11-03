# C# Multilayer Perceptron (MLP) From Scratch

A pure C# implementation of a 3-layer Multilayer Perceptron (Input, Hidden, Output) with backpropagation trained using Stochastic Gradient Descent (SGD) on the Fisher's Iris dataset.

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
  - [Training & Test Loss Curve](#training--test-loss-curve)

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
- **Split Ratio:** 80% train (120 samples) / 20% test (30 samples) with fixed seed `42`.

---

## Architecture & Mathematical Engine

### Network Layers

```
[Input Layer: 4 Features]
           │
           ▼
[Hidden Layer: 8 Neurons (Sigmoid)]
           │
           ▼
[Output Layer: 3 Neurons (Sigmoid + One-Hot Classes)]
```

### Forward Pass & Activation

Standard Sigmoid activation function is used:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

For hidden neuron $j$:
$$\text{out}_j = \sigma\left(\sum_i \text{in}_i \cdot w_{ij} + b_j\right)$$

For output neuron $k$:
$$\text{out}_k = \sigma\left(\sum_j \text{out}_j \cdot w_{jk} + b_k\right)$$

### Backpropagation & Weight Updates

Using Mean Squared Error (MSE) loss:
$$E = \frac{1}{K}\sum_{k=1}^K (\text{out}_k - \text{target}_k)^2$$

1. **Output Layer Deltas:**
   $$\delta_k = (\text{out}_k - \text{target}_k) \cdot \text{out}_k \cdot (1 - \text{out}_k)$$

2. **Hidden Layer Deltas:**
   Computed using the **unmodified (old)** weights of the output layer:
   $$\delta_j = \text{out}_j \cdot (1 - \text{out}_j) \sum_k \delta_k \cdot w_{jk}$$

3. **Weight & Bias Updates (Stochastic Gradient Descent):**
   - Output layer:
     $$w_{jk} \leftarrow w_{jk} - \eta \cdot \delta_k \cdot \text{out}_j$$
     $$b_k \leftarrow b_k - \eta \cdot \delta_k$$
   - Hidden layer:
     $$w_{ij} \leftarrow w_{ij} - \eta \cdot \delta_j \cdot \text{in}_i$$
     $$b_j \leftarrow b_j - \eta \cdot \delta_j$$

Weights are initialized using **Xavier uniform initialization**:
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}, +\sqrt{\frac{6}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}}\right)$$

### Preprocessing & Leak-Free Normalization

To prevent data leakage:
- Min and max bounds for each feature are calculated **strictly from the training partition** ($N = 120$).
- The test set is normalized using these exact training parameters:
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
│   └── Preprocessor.cs     # Min-max normalization (train-only fit), one-hot, 80/20 split
├── Network/
│   ├── InputLayer.cs       # Stores input features
│   ├── HiddenLayer.cs      # Weights, biases, sigmoid activation & backward pass
│   ├── OutputLayer.cs      # Weights, biases, sigmoid activation, deltas & updates
│   └── NeuralNetwork.cs    # Forward, backward, train step, Xavier init, argmax predict
├── Training/
│   └── Trainer.cs          # SGD epoch loop, training logger, SVG generator
├── Evaluation/
│   └── Metrics.cs          # Accuracy, confusion matrix, precision, recall, F1-score
├── Program.cs              # CLI runner, XOR sanity test, grid search, evaluation
├── Multi-layer perceptron.csproj
├── training_log.csv        # Epoch-by-epoch loss & accuracy history
├── tuning_results.csv      # Hyperparameter sweep logs (36 configurations)
├── loss_curve.svg          # Standalone vector graphic of the training loss curve
├── .gitignore              # Ignores bin/, obj/, user files
└── README.md
```

---

## How to Run

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or higher

### Running the Sanity Test (XOR)

To verify the backpropagation algorithm on the non-linearly separable XOR problem (2-4-1, $\eta=0.5$, 5000 epochs, target loss $< 0.01$):

```bash
dotnet run -- --xor
```

**Output:**
```
Running XOR sanity test (2-4-1, lr 0.5, 5000 epochs)...
Epoch 5000/5000 | Train Loss: 0.00080 | Test Loss: 0.00080 | Train Acc: 100.0 % | Test Acc: 100.0 %

Final XOR Loss: 0.000801 (Threshold: < 0.01)
Input: [0, 0] -> Target: 0, Prediction: 0.0250
Input: [0, 1] -> Target: 1, Prediction: 0.9730
Input: [1, 0] -> Target: 1, Prediction: 0.9737
Input: [1, 1] -> Target: 0, Prediction: 0.0339

[PASS] XOR sanity test passed successfully!
```

### Running the End-to-End Iris Pipeline

Executes the full pipeline: automated data verification, 36-configuration hyperparameter grid search, retrains the optimal model, exports `training_log.csv` and `loss_curve.svg`, and prints the evaluation report.

```bash
dotnet run
```

---

## Hyperparameter Sweep Results

A grid search was executed over 36 configurations:
- **Learning rate ($\eta$):** `{0.01, 0.05, 0.1, 0.5}`
- **Hidden layer units ($h$):** `{4, 8, 16}`
- **Epochs:** `{500, 1000, 2000}`

All results are logged to `tuning_results.csv`.

**Best Configuration Selected:**
- **Learning Rate:** `0.5`
- **Hidden Size:** `8`
- **Epochs:** `500`
- **Train Loss:** `0.01252`
- **Test Loss:** `0.00119`
- **Train Accuracy:** `97.5%`
- **Test Accuracy:** `100.0%`

---

## Final Model Evaluation

Evaluated on the held-out test set ($N = 30$ samples, 20% of dataset).

### Confusion Matrix

```
Actual \ Predicted       Iris-setosa Iris-versicolor  Iris-virginica
--------------------------------------------------------------------
Iris-setosa                       11               0               0
Iris-versicolor                    0               8               0
Iris-virginica                     0               0              11
```

### Classification Metrics

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Iris-setosa** | 1.0000 | 1.0000 | 1.0000 | 11 |
| **Iris-versicolor** | 1.0000 | 1.0000 | 1.0000 | 8 |
| **Iris-virginica** | 1.0000 | 1.0000 | 1.0000 | 11 |
| **Accuracy** | | | **1.0000 (100%)** | 30 |
| **Macro Average** | 1.0000 | 1.0000 | 1.0000 | 30 |

### Training & Test Loss Curve

The training history was logged to `training_log.csv` and plotted directly into `loss_curve.svg`:

- **Train Loss:** Decreased smoothly from `0.225` down to `0.0125`.
- **Test Loss:** Decreased smoothly from `0.200` down to `0.00119`.
- **Generalization:** No overfitting observed; test accuracy remained at 100%.

