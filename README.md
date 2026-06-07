# Income Bracket Classification with PyTorch

## Overview
This project implements a feedforward neural network using PyTorch to classify income levels into two brackets: >50K and <=50K. The model is trained, tuned, and evaluated on the UCL Adult Income dataset, and it includes an explainability component using SHAP to understand feature importance.

## Dataset
The project utilizes the UCL Adult Income dataset, which contains 14 input features including age, work class, education, marital status, and occupation. 
* **Preprocessing:** Missing values were handled via imputation, using the median for numerical columns and the mode for categorical columns.
* **Splitting:** The dataset was divided into training, validation, and test sets using a 70/15/15 split.
* **Encoding & Scaling:** Based on an ablation study, numerical values were standard-scaled and categorical values were one-hot encoded.

## Project Structure & Methodology

### 1. Model Architecture & Ablation Studies
A flexible feedforward neural network was built to test 36 combinations of network parameters. 
* **Layers tested:** 1, 2, and 3 hidden layers.
* **Activations tested:** ReLU, Tanh, LeakyReLU.
* **Regularization tested:** Dropout, Batch Normalization.
* **Baseline Best:** A 3-layer architecture with ReLU, Dropout, and Batch Normalization yielded the highest validation F1 score.

### 2. Custom Training Loop
A custom training loop was implemented to identify the optimal loss function, optimizer, and learning rate.
* **Loss Functions:** BCELoss and BCEWithLogitsLoss.
* **Optimizers:** Adam, SGD, and RMSprop.
* **Result:** The best configuration utilized BCELoss, the RMSprop optimizer, and a learning rate of 0.01.

### 3. Regularization & Hyperparameter Tuning
Extensive hyperparameter tuning was conducted to control overfitting and maximize performance.
* **Variables tuned:** Dropout rates (0.1 - 0.5), weight decay (0 - 0.01), batch sizes (16 - 128), and network architectures (small, medium, large).
* **Optimal Hyperparameters:** Medium architecture (2 hidden layers), batch size of 128, dropout rate of 0.1, weight decay of 0.001, and a learning rate of 0.001 without early stopping.

## Final Model Evaluation
The final model featured 4 layers (256 -> 128 -> 64 -> 1) with ReLU activation, utilizing Dropout (0.2) and Batch Normalization after each hidden layer.

**Test Set Metrics:**
* **Accuracy:** 85.83%
* **F1 Score:** 0.6812
* **Precision:** 0.7379
* **Recall:** 0.6326

**Bias Analysis:** The evaluation identified distinct demographic biases, noting significantly higher error rates for males (18.5% vs 7% for females), married individuals, and foreign-born individuals.

## Explainability (SHAP)
SHAP (SHapley Additive exPlanations) was utilized to interpret the model's predictions.
* **Key Positive Drivers:** High capital gains and longer working hours per week were the strongest indicators for a >50K prediction.
* **Demographic Impact:** Marital status (married), higher education levels, and older age showed positive correlations with higher predicted incomes. 

## Author
* **Name:** Akshay Deepak
* **Roll No:** BE24B002
