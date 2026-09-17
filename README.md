# Insurance Model Performance Metrics

## Project Goal

This project investigates how different machine learning evaluation metrics tell fundamentally different stories about model behavior on the exact same dataset. Using the Medical Insurance Charges dataset, we evaluate both:
- **Regression**: Predicting continuous medical charges (`charges`) in dollars.
- **Classification**: Predicting whether an individual's medical cost belongs to a `cheap` or `expensive` risk tier.

## Dataset

- **Source**: Canonical Medical Insurance Charges dataset (*Machine Learning with R* by Brett Lantz).
- **Dimensions**: 1,338 raw records, 7 columns.
- **Features**: `age`, `sex`, `bmi`, `children`, `smoker`, `region`.
- **Target**: `charges` (continuous annual medical costs billed to insurance).

## Project Workflow

```
Raw Data
 → Data Audit
 → Encoding + Scaling
 → Train/Test Split
 → Regression Baseline
 → KNN Regression
 → Regression Metrics
 → KNN Hyperparameters
 → Cross Validation
 → Classification Target
 → DummyClassifier
 → KNN Classification
 → Confusion Matrix
 → Precision / Recall / F1
 → Probability Thresholds
 → Precision–Recall Curve
 → ROC-AUC
 → Error Analysis
```

## Regression Results

Evaluated on the 30% holdout test set (402 samples):

| Model | $R^2$ | MAE ($) | RMSE ($) | Max Error ($) |
|---|---|---|---|---|
| **DummyRegressor (mean)** | `-0.0000` | `8,426.33` | `11,390.13` | `41,866.37` |
| **KNN Regressor ($K=5$)** | `0.7298` | `3,588.39` | `5,921.13` | `28,240.34` |

5-Fold Cross-Validation on the training partition yielded a Mean CV $R^2$ of `0.7406` (fold range `0.6538` to `0.7869`), Mean CV MAE of `$3,910.84`, and Mean CV RMSE of `$6,300.38`, confirming that test performance is broadly consistent with cross-validation estimates.

## Classification Results

Target defined using the training median threshold ($8,968.33):

| Model | Accuracy | Precision | Recall | F1 Score | False Positives | False Negatives |
|---|---|---|---|---|---|---|
| **DummyClassifier (most_frequent)** | `0.4403` | `0.0000` | `0.0000` | `0.0000` | `0` | `225` |
| **KNN Default (Threshold 0.5)** | `0.8930` | `0.9505` | `0.8533` | `0.8993` | `10` | `33` |
| **KNN Custom (Threshold $\ge 0.4$)** | `0.8706` | `0.8650` | `0.9111` | `0.8874` | `32` | `20` |

Lowering the decision threshold from $0.5$ to $\ge 0.4$ achieved the hypothetical business target ($\text{Recall} \ge 90\%$), reducing missed expensive patients by ~39% (FN dropped from 33 to 20) at the expense of an increase in False Positives (from 10 to 32).

## Key Learnings

- **Baselines provide essential context**: A model with 44% accuracy or $R^2 \approx 0.0$ establishes the performance floor of naive majority/mean guessing.
- **Different regression metrics capture different error behaviors**: MAE measures typical dollar error, RMSE penalizes large errors quadratically, and Max Error bounds the worst single failure.
- **Distance-based algorithms require strict scaling**: KNN relies directly on distance geometry; unscaled features or arbitrary distance metrics ($p$) fundamentally change neighborhood formation.
- **Accuracy alone is insufficient**: Accuracy conceals error asymmetry. In insurance risk management, failing to detect an expensive patient (False Negative) is often far more costly than an unnecessary review (False Positive).
- **Thresholds shift trade-offs without retraining**: Precision and Recall trade off directly; operational requirements determine the optimal decision threshold.
- **ROC-AUC evaluates ranking power**: Test ROC-AUC of `0.9416` demonstrates strong class-separation ability across all possible operating thresholds.
- **Aggregate metrics must be complemented by error & cohort analysis**: Aggregate numbers masked that KNN achieves 100% recall on smokers, while almost all severe prediction errors occur on non-smokers who experienced unexpected medical complications.

## Methodology Safeguards

- Exact duplicates were identified and removed before train/test splitting to prevent identical observation leakage.
- Preprocessing transformers (OneHotEncoder and MinMaxScaler) were fitted strictly on training data (`X_train`) to prevent test snooping.
- `charges` was strictly excluded from classification feature inputs to eliminate direct target leakage.
- The binary classification threshold was derived exclusively from `y_train.median()`.
- The custom decision threshold was selected solely from training out-of-fold cross-validation probabilities without looking at test labels.

## Limitations

- **Educational target definition**: The classification boundary is based on a simple median split rather than commercial underwriting guidelines or actuarial risk brackets.
- **Discrete KNN probability granularity**: With $K=5$ and uniform voting, predicted probabilities exist in discrete steps of $0.2$ ($0.0, 0.2, 0.4, 0.6, 0.8, 1.0$), limiting fine-grained threshold calibration compared to parametric models.
- **Lesson-style CV preprocessing**: Preprocessing before CV was fitted on the full training partition per lesson curriculum, rather than inside each fold via a Pipeline.
- **Educational scope**: This project is designed to master evaluation metrics and diagnostics, not to provide an actuarial production pricing model.

## How to Run

```bash
# Clone the repository
git clone https://github.com/abed-dvp/insurance-model-performance-metrics.git
cd insurance-model-performance-metrics

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook notebook.ipynb
```
