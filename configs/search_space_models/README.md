# Hyperparameter Search Space & Model Registry (`configs/search_space_models/`)

This directory contains the declarative hyperparameter search space specifications and master registry for all candidate machine learning algorithms evaluated in the **Astro Object Classification** pipeline.

Each configuration file defines the mathematical prior distributions (uniform, log-uniform, categorical) explored by **Optuna** during Bayesian Hyperparameter Optimization (HPO), as well as fixed operational constants required for training execution, parallelization, and reproducibility.

---

## Table of Contents
1. [Search Space Domain-Specific Language (DSL)](#search-space-domain-specific-language-dsl)
2. [Namespace Isolation & Naming Convention](#namespace-isolation--naming-convention)
3. [Master Architecture Matrix (9 Models)](#master-architecture-matrix-9-models)
4. [Master Registry (`defined_models.yaml`)](#master-registry-defined_modelsyaml)
5. [Model Family Architectural Breakdown](#model-family-architectural-breakdown)
6. [HPO Calibration & Best Practices](#hpo-calibration--best-practices)
7. [Checklist: Adding a New Search Space](#checklist-adding-a-new-search-space)

---

## Search Space Domain-Specific Language (DSL)

All model configurations are parsed by `OptunaConfigLoader` (`configs/optuna_config.py`) and translated into native Optuna `Trial` suggestions in `src/optimization/core.py`. The schema supports four fundamental parameter types:

### 1. Integer Parameters (`type: int`)
Defines bounded discrete integer search intervals.
```yaml
catboost_iterations:
  type: int
  low: 80
  high: 1000
```
* **Translation:** `trial.suggest_int(name, low, high, log=False)`
* **Optional Field:** `log: true` activates logarithmic discretization (useful when exploring across multiple orders of magnitude).

### 2. Float Parameters (`type: float`)
Defines continuous bounded real-valued search intervals.
```yaml
lightgbm_learning_rate:
  type: float
  low: 0.001
  high: 0.3
  log: true
```
* **Translation:** `trial.suggest_float(name, low, high, log=True)`
* **Usage Rule:** Use `log: true` for multiplicative scales (learning rates, regularization parameters $\alpha$, $\lambda$, $C$, weight decay). Use `log: false` for linear additive scales (dropout rate, subsample ratios).

### 3. Categorical Parameters (`type: categorical`)
Defines discrete sets of unordered or nominal choices.
```yaml
extra_trees_max_features:
  type: categorical
  choices: ["sqrt", "log2", null]
```
* **Translation:** `trial.suggest_categorical(name, choices)`
* **Note:** `null` in YAML maps directly to Python `None` (e.g., using all available features).

### 4. Fixed Parameters (`type: fixed`)
Defines invariant operational constraints and objective functions passed directly to the model constructor without Optuna optimization.
```yaml
extra_trees_class_weight:
  type: fixed
  value: "balanced"
```
* **Translation:** Injected directly into the model keyword arguments dictionary: `kwargs[name] = value`.
* **Standard Fixed Fields:** Parallelism (`n_jobs: -1`), verbosity (`verbose: 0`), loss objectives (`multiclass`, `multi:softprob`), and class weighting (`balanced`).

---

## Namespace Isolation & Naming Convention

All optimizable and fixed parameters **must strictly follow the prefixed naming convention**:

All optimizable and fixed parameters **must strictly follow the prefixed naming convention**:

$$
\text{Parameter Key} = \texttt{model\_name\_param\_name}
$$

### Why this is mandatory:
1. **Multi-Model Studies:** During a single global Optuna study (e.g. 500 trials), Optuna samples `model_name` from `candidate_models` and persists all sampled parameters into a single global study database (`sqlite:///` or memory).
2. **Namespace Collisions:** Prefixing prevents collision between parameters shared across different algorithms (e.g., `random_forest_n_estimators` vs `lightgbm_n_estimators` vs `xgboost_n_estimators`).
3. **Automated Stripping:** During model instantiation, `apply_optuna_suggestions` in `src/optimization/core.py` programmatically strips the `{model_name}_` prefix, passing clean kwargs (`n_estimators`, `max_depth`, etc.) to the underlying estimator.

---

## Master Architecture Matrix (9 Models)

The repository provides hyperparameter spaces across 5 distinct algorithmic families:

| Model ID | Family | Underlying Implementation | Primary Tuned Hyperparameters | Critical Fixed Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **`random_forest`** | Bagging Trees | `sklearn.ensemble.RandomForestClassifier` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`, `bootstrap` | `class_weight: balanced`, `n_jobs: -1`, `verbose: 0` |
| **`extra_trees`** | Randomized Bagging | `sklearn.ensemble.ExtraTreesClassifier` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`, `bootstrap` | `class_weight: balanced`, `n_jobs: -1`, `verbose: 0` |
| **`lightgbm`** | Leaf-wise GBDT | `lightgbm.LGBMClassifier` | `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` | `objective: multiclass`, `class_weight: balanced`, `n_jobs: -1`, `verbose: -1` |
| **`xgboost`** | Depth-wise GBDT | `xgboost.XGBClassifier` | `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` | `objective: multi:softprob`, `n_jobs: -1`, `verbosity: 0` |
| **`catboost`** | Oblivious GBDT | `catboost.CatBoostClassifier` | `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `random_strength`, `border_count` | `loss_function: MultiClass`, `auto_class_weights: Balanced`, `thread_count: -1`, `verbose: 0` |
| **`dense_nn`** | Deep MLP | `tensorflow.keras.Sequential` | `num_layer` (2-5), `num_perceptron` (32-256), `drop_out`, `batch_norm`, `learning_rate`, `batch_size`, `optimizer`, `weight_decay`, betas/momenta | `metric_monitor: val_loss`, `early_stopping_patience: 3`, `act_func: softmax`, `class_weights: true` |
| **`svc`** | Kernel SVM | `sklearn.svm.SVC` | `C`, `decision_function_shape` (`ovr`/`ovo`), `kernel` (`linear`/`rbf`), `gamma`, `degree`, `coef0`, `shrinking`, `tol` | `probability: true`, `class_weight: balanced`, `max_iter: 2500` *(prevents $O(N^3)$ hang)* |
| **`logreg`** | Linear Classifier | `sklearn.linear_model.LogisticRegression` | `C`, `l1_ratio`, `max_iter` | `solver: saga`, `penalty: elasticnet`, `class_weight: balanced`, `n_jobs: -1`, `verbose: 0` |
| **`sgd`** | Linear SGD | `sklearn.linear_model.SGDClassifier` | `loss` (`log_loss`/`modified_huber`), `penalty` (`l1`/`l2`/`elasticnet`), `alpha`, `l1_ratio`, `learning_rate` (`optimal`/`adaptive`/`invscaling`), `eta0` | `early_stopping: true`, `max_iter: 5000`, `class_weight: balanced`, `n_jobs: -1`, `verbose: 0` |

---

## Master Registry (`defined_models.yaml`)

[defined_models.yaml](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/search_space_models/defined_models.yaml) acts as the single source of truth for recognized model keys across the codebase:

```yaml
names_defined_models:
  # Ensemble & Tree-based Classifiers
  - "random_forest"
  - "extra_trees"
  - "xgboost"
  - "lightgbm"
  - "catboost"

  # Deep Neural Network Architecture
  - "dense_nn"

  # Linear & Kernel Classifiers
  - "svc"
  - "sgd"
  - "logreg"
```

Any model listed here must have:
1. A corresponding YAML file: `configs/search_space_models/{model_name}.yaml`.
2. A path reference in `configs/paths.py`: `{model_name}_search_space`.
3. An implementation registered in `ModelFactory` (`src/models/model_factory.py`).

---

## Model Family Architectural Breakdown

### 1. Tree Ensembles & Gradient Boosters
* **Random Forest vs Extra Trees:** Random Forest computes exact discriminative thresholds per feature split, while Extra Trees draws random thresholds and selects the best, providing stronger variance reduction and faster training.
* **LightGBM:** Builds leaf-wise (asymmetric) trees, constrained by `num_leaves` and `min_child_samples` to prevent extreme overfitting on minority astronomical targets.
* **CatBoost:** Leverages symmetric (oblivious) decision trees, making it naturally resilient to overfitting and robust across categorical/continuous tabular data.
* **XGBoost:** Utilizes exact second-order Taylor expansion gradients with $L_1$ (`reg_alpha`) and $L_2$ (`reg_lambda`) leaf regularization.

### 2. Deep Feedforward Neural Network (`dense_nn`)
* **Funnel Architecture:** The first hidden layer instantiates `num_perceptron` units ($32$ to $256$), halving capacity across successive layers ($N \to N/2 \to N/4$).
* **Dynamic Optimizer Exploration:** Evaluates `adam`, `adamw`, `rmsprop`, and `sgd`, sampling their respective momentum, decay, and beta hyperparameter spaces.
* **Built-in Callbacks:** Automated `ReduceLROnPlateau` (factor: 0.75, patience: 2) and `EarlyStopping` (patience: 3, restoring best weights).

### 3. Support Vector Classifier (`svc`)
* **Complexity Guardrail:** On datasets exceeding 50,000 samples, standard LIBSVM solvers scale with $\mathcal{O}(N^2)$ to $\mathcal{O}(N^3)$ computational complexity. To prevent trial timeouts or multi-hour solver hangs, `svc_max_iter: 2500` is enforced as a hard fixed limit.
* **Platt Scaling:** `svc_probability: true` is fixed to generate calibrated probability estimates required for multi-class log-loss and ROC-AUC evaluation.

### 4. Linear & Online Classifiers (`logreg`, `sgd`)
* **ElasticNet Regularization:** Both algorithms explore the full continuum between pure $L_2$ weight shrinkage (`l1_ratio: 0.0`) and pure $L_1$ feature sparsity (`l1_ratio: 1.0`).
* **SAGA Solver:** In `logreg`, `solver: saga` is fixed because it is the only scikit-learn solver capable of handling non-smooth ElasticNet penalties across multi-class problems.

---

## HPO Calibration & Best Practices

1. **Logarithmic vs Linear Scales:**
   * **Use `log: true`** when the order of magnitude matters more than the absolute difference (e.g., $10^{-4}$ to $10^{-1}$ for learning rates or regularization coefficients).
   * **Use `log: false`** for bounded proportions or percentages (e.g., subsampling fractions between $0.5$ and $1.0$, dropout rates between $0.0$ and $0.5$).
2. **Coarse-to-Fine Tuning:**
   * Run an initial exploratory sweep (e.g. 100-300 trials) with broad bounds.
   * Inspect Optuna parameter importances and slice plots (`optuna.visualization.plot_param_importances`, `plot_slice`).
   * Tighten the `low` and `high` bounds around the high-performing basins for production fine-tuning.
3. **Class Imbalance Strategy:**
   * Tree models rely on `class_weight: balanced` to penalize errors on minority classes (QSO: ~19%, Star: ~21%, Galaxy: ~59%).
   * This is complemented by Optuna exploring outer data resampling strategies (`smote`, `undersampling`, `class_weight`) defined in `configs/optuna.yaml`.

---

## Checklist: Adding a New Search Space

To integrate a new algorithm (e.g., `adaboost.yaml`):

- [ ] Create `configs/search_space_models/adaboost.yaml` with the standard 77-char banner.
- [ ] Ensure all search space parameters have the `adaboost_` prefix.
- [ ] Add exactly two comment lines per hyperparameter (Functionality + Trade-off).
- [ ] Add the `# Fixed Execution & Objective Parameters` block with `verbose: 0`, parallelism, and objective settings.
- [ ] Register `"adaboost"` in [defined_models.yaml](file:///Users/antoniomartella/Progetti%20GitHub/astro_object_classif/configs/search_space_models/defined_models.yaml).
- [ ] Add `adaboost_search_space: Path = search_space_models_config_dir / "adaboost.yaml"` to `configs/paths.py`.
- [ ] Register `AdaBoostModel` in `src/models/model_factory.py`.
- [ ] Add `"adaboost"` to `search_space.candidate_models` in `configs/optuna.yaml`.
