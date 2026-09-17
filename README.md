# CSCI316 Big Data Mining — Credit Risk Prediction

Final project for CSCI316 (Big Data Mining). We predict whether a Lending Club loan will **default** (`default_ind`) using a large, heavily imbalanced loan-origination dataset, comparing a PySpark MLlib pipeline against a TensorFlow/Keras pipeline.

## Project structure

| Notebook | Description |
|---|---|
| [stage1_preprocessing.ipynb](stage1_preprocessing.ipynb) | EDA, data cleaning, leakage/column removal, outlier capping, imputation. Produces the cleaned dataset used by every downstream notebook, and ranks the 7 most/least relevant features. |
| [stage2_PySpark.ipynb](stage2_PySpark.ipynb) | PySpark MLlib pipeline (Logistic Regression, Decision Tree, Random Forest) trained on the **full feature set**. |
| [stage2_PySpark_7_features.ipynb](stage2_PySpark_7_features.ipynb) | Same PySpark MLlib pipeline restricted to the **7 most relevant features** identified in stage 1. |
| [stage2_tensorflow.ipynb](stage2_tensorflow.ipynb) | TensorFlow/Keras pipeline (MLP, TabNet, ResMLP) trained on the **full feature set**, preprocessed via Spark ML. |
| [stage2_tensorflow_7_features.ipynb](stage2_tensorflow_7_features.ipynb) | Same TensorFlow/Keras pipeline restricted to the **7 most relevant features**. |
| [visualization.ipynb](visualization.ipynb) | Cross-comparison of all four Stage 2 runs (PySpark vs TensorFlow, 7 features vs all features). |
| [Slide Deck.pdf](Slide%20Deck.pdf) | Presentation slides summarizing the project. |

## Data

The raw and intermediate data files (`data.csv`, `cleaned_loan_data.csv`, `annual_inc_csv`) are **not committed** to this repo (see `.gitignore`) — place the source Lending Club CSV as `data.csv` in the project root before running `stage1_preprocessing.ipynb`. That notebook writes `cleaned_loan_data.csv`, which every Stage 2 notebook reads as its input.

Target variable: `default_ind` (1 = loan defaulted, 0 = did not). The dataset is highly imbalanced — only ~5.4% of loans default.

## Pipeline

1. **Stage 1 — Preprocessing (`stage1_preprocessing.ipynb`)**
   - Drop leakage columns (anything only known after a loan defaults, e.g. `recoveries`, `out_prncp`, `total_pymnt`), near-useless columns (IDs, free text, near-constant fields), and rows with a null target.
   - Parse string fields to numeric (`int_rate`, `revol_util`, `term`, `emp_length`, dates).
   - Cap outliers (`revol_util`, `dti`, `annual_inc`) and log-transform `annual_inc`.
   - Impute remaining nulls (median for numeric, `'unknown'` for categorical).
   - Rank features by relevance; the **7 most relevant** are `sub_grade`/`grade`, `int_rate`, `purpose`, `term`, `inq_last_6mths`, `revol_util`, `verification_status`.

2. **Stage 2 — Modeling**, run twice per framework (full feature set and 7-feature subset):
   - **PySpark MLlib**: Logistic Regression, Decision Tree, Random Forest, via a `Pipeline` of `Imputer` → `StringIndexer` → `OneHotEncoder` → `VectorAssembler`.
   - **TensorFlow/Keras**: MLP (Dense/BatchNorm/Dropout), TabNet (attention-based feature selection), ResMLP (pre-norm residual FFN), using Spark ML for preprocessing and feeding NumPy arrays into Keras.
   - All models evaluated on Accuracy, Precision, Recall, F1, and ROC-AUC — ROC-AUC is treated as the primary metric because of the class imbalance.

3. **Visualization (`visualization.ipynb`)** aggregates results across all four runs for the final comparison.

## Results summary

Because of the ~5.4% default rate, accuracy alone is misleading (a model predicting "no default" for everything scores ~94.6%). ROC-AUC is the key comparison metric.

**PySpark MLlib (7 features):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6592 | 0.1004 | 0.6577 | 0.1742 | 0.7182 |
| Decision Tree | 0.6571 | 0.1057 | 0.7070 | 0.1840 | 0.5790 |
| Random Forest | 0.6174 | 0.0969 | 0.7206 | 0.1708 | **0.7267** |

**TensorFlow/Keras (7 features):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| MLP | 0.6602 | 0.1069 | 0.7109 | 0.1859 | 0.7479 |
| TabNet | 0.6865 | 0.1115 | 0.6805 | 0.1915 | 0.7506 |
| ResMLP | 0.6745 | 0.1105 | 0.7042 | 0.1910 | **0.7535** |

Across all four runs, `int_rate` and `sub_grade`/`grade` are consistently the strongest predictors, and the neural network models (ResMLP, TabNet) slightly outperform PySpark's Random Forest on ROC-AUC.

## Requirements

- Apache Spark / PySpark (with `pyspark.ml`)
- TensorFlow / Keras
- pandas, numpy, seaborn, matplotlib
- Jupyter (to run the `.ipynb` notebooks)

## Running

Run the notebooks in this order:

1. `stage1_preprocessing.ipynb` — requires `data.csv` in the project root, produces `cleaned_loan_data.csv`.
2. `stage2_PySpark.ipynb` and/or `stage2_PySpark_7_features.ipynb`
3. `stage2_tensorflow.ipynb` and/or `stage2_tensorflow_7_features.ipynb`
4. `visualization.ipynb` — compares results from all Stage 2 notebooks.
