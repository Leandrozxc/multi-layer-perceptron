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
   $$\delta_k = \frac{2}{K}(\text{out}_k - \text{target}_k) \cdot \text{out}_k \cdot (1 - \text{out}_k)$$

The formulas here use positive loss gradients and subtractive updates. The Backprop engine stores their negatives using `(target - output)` and adds the updates, which is equivalent. The `2 / K` factor makes the gradient match the reported mean squared error.

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

The engine uses `Backprop.NeuralNet`, `INeuron`, `HNeuron`, and `ONeuron` from `Libraries/`. `Network/NeuralNetwork.cs` adapts prediction and per-sample training to this API. Weights are initialized from integer draws in `[-1000, 1000)` divided by 2000, with hidden/output biases of `0.01`. A seeded constructor makes experiments reproducible. The pipeline's learning rate is applied to both input-to-hidden and hidden-to-output updates; the library's original constructors retain their defaults of `0.15` and `0.2` respectively.

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
├── Network/
│   ├── InputLayer.cs       # Legacy layer source (excluded from compilation)
│   ├── HiddenLayer.cs      # Legacy layer source (excluded from compilation)
│   ├── OutputLayer.cs      # Legacy layer source (excluded from compilation)
│   └── NeuralNetwork.cs    # Adapter to Backprop engine, sample MSE, argmax predict
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

The active neural network engine is in `Libraries/BackNeural 1.cs`, `Libraries/InputN 1.cs`, `Libraries/HiddenN 1.cs`, and `Libraries/OuputN 1.cs`. The three legacy layer files under `Network/` are retained as source references and excluded from the build.

## How to Run

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or higher

### Running the Sanity Test (XOR)

To verify the backpropagation algorithm on the non-linearly separable XOR problem (2-4-1, $\eta=0.5$, 5000 epochs, target loss $< 0.01$ and all four binary predictions correct):

```bash
dotnet run -- --xor
```

**Output:**
```
Running XOR sanity test (2-4-1, lr 0.5, 5000 epochs)...
Epoch 5000/5000 | Train Loss: 0.00033 | Validation Loss: 0.00033 | Train Acc: 100.0 % | Validation Acc: 100.0 %

Final XOR Loss: 0.000331 (Threshold: < 0.01)
Input: [0, 0] -> Target: 0, Prediction: 0.0161
Input: [0, 1] -> Target: 1, Prediction: 0.9827
Input: [1, 0] -> Target: 1, Prediction: 0.9827
Input: [1, 1] -> Target: 0, Prediction: 0.0216
Final XOR Accuracy: 100.0% (Threshold: 0.5)

[PASS] XOR sanity test passed successfully!
```

Single-output accuracy classifies outputs `>= 0.5` as 1 and lower outputs as 0. XOR succeeds only when MSE is below `0.01` and all four predictions are correct; failure returns exit code 1. Iris continues to use argmax for its three outputs. XOR uses the same four samples for training and validation as a sanity check.

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

Configurations are ranked by validation accuracy, then by lower validation MSE on ties. Test metrics are not calculated during the sweep or epoch logging. The selected configuration is retrained on the same 90 training samples with the same seed; validation samples are not added to training.

**Best Configuration Selected:**
- **Learning Rate:** `0.1`
- **Hidden Size:** `4`
- **Epochs:** `1000`
- **Train Loss:** `0.010104`
- **Validation Loss:** `0.020241`
- **Train Accuracy:** `98.9%`
- **Validation Accuracy:** `96.7%`

---

### Why this configuration was selected

The table below compares selected rows from [the complete 36-configuration sweep](tuning_results.csv). Every configuration uses the same split, normalization procedure, initialization seed, and training-order seed. These are single runs rather than averages across repeated experiments.

| Learning rate | Hidden neurons | Epochs | Train MSE | Validation MSE | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 4 | 500 | 0.124245 | 0.127303 | 76.7% |
| 0.05 | 4 | 1000 | 0.015970 | 0.023129 | 96.7% |
| 0.1 | 4 | 500 | 0.016066 | 0.024151 | 96.7% |
| 0.1 | 4 | 1000 | 0.010104 | 0.020241 | 96.7% |
| 0.1 | 8 | 1000 | 0.010188 | 0.020247 | 96.7% |
| 0.1 | 16 | 1000 | 0.010346 | 0.020668 | 96.7% |
| 0.1 | 4 | 2000 | 0.006507 | 0.026270 | 96.7% |
| 0.5 | 4 | 2000 | 0.002479 | 0.052614 | 86.7% |

The highest recorded validation accuracy is 96.7% (29 of 30 samples). Among configurations reaching that accuracy, learning rate `0.1`, 4 hidden neurons, and 1,000 epochs has the lowest final validation MSE, `0.020241`. This is why the implemented selection rule chooses it; test accuracy does not participate in the decision.

- **Learning rate:** With 4 hidden neurons and 500 epochs, `0.01` reaches only 76.7% validation accuracy, whereas `0.1` reaches 96.7%. The lower rate appears to need more training in this comparison. At the other extreme, `0.5` reaches 100% training accuracy after 2,000 epochs but only 86.7% validation accuracy. Lower training error alone is therefore a poor selection rule.
- **Hidden size:** At learning rate `0.1` and 1,000 epochs, all three widths reach 96.7% validation accuracy. Four neurons gives validation MSE `0.020241`, versus `0.020247` for eight and `0.020668` for sixteen. The measured advantage over eight is very small; this single run does not establish that four neurons is consistently superior. It is the winner under the stated rule and also uses the smallest tested hidden layer.
- **Epoch count:** At learning rate `0.1` with 4 hidden neurons, increasing training from 1,000 to 2,000 epochs reduces training MSE from `0.010104` to `0.006507` but increases validation MSE from `0.020241` to `0.026270`, with unchanged validation accuracy. This divergence is consistent with overfitting and supports selecting the earlier tested endpoint for this run.

The configuration is best among the tested grid endpoints under the chosen ranking rule. The experiment does not establish a globally optimal learning rate, width, or epoch count.

---

## Final Model Evaluation

Evaluated after model selection on the held-out test set ($N = 30$ samples, 20% of dataset). Final test MSE: `0.006164`.

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

### What the metrics mean

The selected model correctly classifies 89 of 90 training samples, 29 of 30 validation samples, and all 30 test samples. The confusion matrix contains no test errors: 11 setosa, 8 versicolor, and 11 virginica samples are classified correctly. Consequently, each class has precision, recall, and F1 of 1.0 on this test partition; the macro averages are also 1.0. There are no test mistakes here to characterize by species.

Test MSE is `0.006164` despite perfect accuracy because sigmoid outputs are not exactly equal to their one-hot targets. Accuracy checks only which output is largest; MSE also measures the distance between output values and targets. These metrics therefore describe different aspects of the predictions. The three sigmoid outputs are not constrained to sum to one and should not be treated as calibrated class probabilities.

### Training & Validation Loss Curve

The training history was logged to `training_log.csv` and plotted directly into `loss_curve.svg`. Training and validation MSE are both evaluated after all updates in each epoch, using the same fixed model weights and the same MSE function. Final test MSE uses that function after training:

- **Train Loss:** Decreased overall from `0.230856` to `0.010104`.
- **Validation Loss:** Decreased overall from `0.232095` to `0.020241`.
- **Final accuracy:** 98.9% training, 96.7% validation, and 100% test. Per-epoch results fluctuate; consult the CSV for the full history.

The CSV and curve show training and validation metrics only. Test metrics are calculated after the configuration is fixed. The test set contains only 30 samples, so perfect accuracy on this split does not establish perfect performance on new data. This partition was also used in earlier experiments; separating selection now does not make those previously seen samples a completely fresh benchmark.

### Learning behavior and limitations

Both loss curves decrease substantially overall, but neither is guaranteed to decrease at every epoch under SGD. Validation MSE reaches its lowest logged value, `0.018753`, at epoch 860, then ends at `0.020241` at epoch 1,000. This shows fluctuations and some late deterioration in validation loss; it does not justify a claim of steadily improving generalization. The sweep compares only the configured endpoints of 500, 1,000, and 2,000 epochs. It does not implement early stopping or select a checkpoint at epoch 860.

Final validation MSE (`0.020241`) is higher than training MSE (`0.010104`), while test MSE (`0.006164`) is lower than both. Different sample difficulty can produce this ordering. It does not prove that the model performs better on unseen populations than on training data.

The conclusions have several limits:

- **Small partitions:** Each validation or test sample changes accuracy by about 3.3 percentage points. The test supports are only 8 to 11 samples per species, so perfect metrics provide limited evidence about future errors.
- **One split and seed:** Seed `42` makes this experiment reproducible, but it does not measure sensitivity to initialization or partition choice. The shuffled split is not stratified, so class proportions can differ between partitions.
- **Repeated selection on validation data:** Thirty-six configurations are compared against the same 30 validation samples. The winning validation score can be optimistic. A future study could use stratified cross-validation within development data and report variation across seeds.
- **Previously observed test samples:** Earlier experiments used this same test partition for tuning. The current pipeline keeps it out of selection, but a stronger future assessment would use genuinely unseen data after fixing the modeling procedure.
- **Dataset scope:** These results concern Iris flower classification with four measured features. They do not demonstrate performance on other datasets or establish readiness for a real decision-support deployment.
