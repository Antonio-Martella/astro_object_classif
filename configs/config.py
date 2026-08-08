from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class KaggleConfig:
    dataset_path: str = "fedesoriano/stellar-classification-dataset-sdss17"
    file_name: str = "star_classification.csv"


@dataclass
class SplitDatasetConfig:
    # percentuale di record da togliere dal dataset principale
    holdout_split: float = 0.2
    # colonna temporale per fare lo splitting per il dataset di produzione
    ref_col: str = "MJD"
    # imposto un random seed per la riproducibilità
    random_seed = 42
    # colonna da gruppare con GroupShuffleSplit
    col_group = "field_ID"
    # percentuali di records per i set di training
    train_split: float = 0.85


@dataclass
class PreprocessingConfig:
    # features da droppare
    columns_to_drop = [
        "obj_ID",
        "alpha",
        "delta",
        "u",
        "g",
        "i",
        "z",
        "run_ID",
        "rerun_ID",
        "cam_col",
        "spec_obj_ID",
        "plate",
        "MJD",
        "fiber_ID",
    ]

    # featurs valide per lo scaling
    columns_to_scale = ["r", "redshift", "u-g", "g-r", "r-i", "i-z"]

    # condizione per applicare lo scaling
    use_scaler: bool = True

    # strategia del resempling ['none', 'smote', 'undersampling']
    resempling = "none"


@dataclass
class ModelsConfig:
    # modello da usare
    models_name = ["xgboost"]

    # modelli supoortati e loro nomi
    supported_models = [
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "logreg",
        "sgd",
        "svc",
        # "linear_svc",
        "dense_nn",
    ]

    # -------------------------------------------------------
    # -------------- SETTING ENSAMBLE MODELS ----------------
    # -------------------------------------------------------
    random_forest_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced",
            "verbose": 0,
            "n_jobs": -1,
            # "random_state": RandomSeedConfig.random_seed_training
        }
    )

    xgboost_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "multi:softprob",
            "class_weight": "balanced",
            "verbosity": 0,
            "n_jobs": -1,
            # "random_state": RandomSeedConfig.random_seed_training
        }
    )

    lightgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": -1,
            "learning_rate": 0.1,
            "objective": "multiclass",
            "class_weight": "balanced",
            "verbose": -1,
            "n_jobs": -1,
            # "random_state": RandomSeedConfig.random_seed_training
        }
    )

    catboost_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "loss_function": "MultiClass",
            "auto_class_weights": "Balanced",
            "verbose": 0,
            # "random_state": RandomSeedConfig.random_seed_training
        }
    )

    # -------------------------------------------------------
    # -------------- SETTING KERNEL MODELS -----------------
    # -------------------------------------------------------
    svc_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "decision_function_shape": "ovr",
            "class_weight": "balanced",
            "verbose": 0,
            # "random_state": RandomSeedConfig.random_seed_training
        }
    )

    # -------------------------------------------------------
    # -------------- SETTING LINEAR MODELS ------------------
    # -------------------------------------------------------
    logreg_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "C": 1.0,
            "solver": "saga",
            "penalty": "l2",
            "max_iter": 1000,
            "class_weight": "balanced",
            "verbose": 0,
            "n_jobs": -1,
            # "random_state": RandomSeedConfig.random_seed_training
        }
    )

    sgd_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 0.0001,
            "max_iter": 2000,
            "tol": 1e-3,
            "class_weight": "balanced",
            "verbose": 0,
            "n_jobs": -1,
            # "random_state": RandomSeedConfig.random_seed_training
        }
    )

    # -------------------------------------------------------
    # ----------------- SETTING DL MODELS -------------------
    # -------------------------------------------------------
    dense_nn_params: Dict[str, Any] = field(
        default_factory=lambda: {
            # --- Parametri di Inizializzazione Architettura ---
            "hidden_units": [64, 16],  # Neuroni per ogni livello nascosto
            "dropout_rate": 0.3,  # dropout per tutti i layer densi
            "hidden_activation_func": "relu",  # funzione di attivazione degli hidden layers
            # --- Parametri per l'ottimizzatore ---
            "learning_rate": 0.001,  # Passo di apprendimento iniziale dell'ottimizzatore
            "optimizer": "adam",  # ottimizzatore supportati: adam, adamw, rmsprop, sgd
            "beta_1": 0.9,
            "beta_2": 0.999,
            # "optimizer": "adamw"
            # "weight_decay": 0.004
            # "beta_1": 0.9
            # "optimizer": "rmsprop"
            # "rho": 0.9
            # "momentum": 0.0
            # "optimizer": "sgd"
            # "momentum": 0.0
            # "nesterov": False
            # --- Callback ----
            "monitor": "val_loss",
            "lr_factor_reducer": 0.8,
            "patience_lrreducer": 2,
            "patience_earlystop": 5,
            "verbose_cb": 0,
            # --- Parametri del Metodo .fit() ---
            "epochs": 100,  # Numero massimo di epoche
            "batch_size": 256,  # Quanti campioni analizzare prima di aggiornare i pesi
            "shuffle": True,  # Mescola i dati a ogni epoca (ottimo per la convergenza)
            "verbose": 0,
        }
    )

    def get_active_params(self, model_name: str) -> Dict[str, Any]:
        if model_name == "random_forest":
            return self.random_forest_params
        elif model_name == "xgboost":
            return self.xgboost_params
        elif model_name == "lightgbm":
            return self.lightgbm_params
        elif model_name == "catboost":
            return self.catboost_params
        elif model_name == "svc":
            return self.svc_params
        elif model_name == "logreg":
            return self.logreg_params
        elif model_name == "sgd":
            return self.sgd_params
        elif model_name == "dense_nn":
            return self.dense_nn_params
        else:
            return {}
