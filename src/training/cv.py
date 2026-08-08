import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedGroupKFold

from configs.random_seed_loader import load_random_seed_config
from configs.schemas_loader import load_preprocessing_config
from src.data.preprocessing import build_stateful_ml_pipeline
from src.data.resampling import ResamplerFactory
from src.models.model_factory import ModelFactory
from src.utils.metrics import evaluate_classification_metrics


def run_cross_validation(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    resampling_strategy: str = "class_weight",
    scaler_strategy: str = "standard",
    custom_params: dict | None = None,
):
    """
    Esegue una Cross-Validation robusta (Stratified Group K-Fold) per valutare le performance di un modello.

    Per ogni fold, la funzione costruisce dinamicamente un'intera pipeline MLOps (ImbPipeline)
    che include:
    * Il preprocessing dei dati.
    * Un eventuale ribilanciamento delle classi ('class_weight', 'smote' e 'undersampling')
      configurabile nel file configs/... .
    * L'addestramento del modello.
    La validazione è strutturata in modo da prevenire il data leakage, assicurando che i record
    appartenenti allo stesso gruppo (es. stesso 'field_ID') non vengano divisi tra set di
    addestramento e validazione.

    **Args**:
        * *model_name* (str): L'identificativo testuale del modello da istanziare tramite ModelFactory
                          (es. 'xgboost', 'random_forest', 'dense_nn', etc.).
        * *X* (pd.DataFrame): Il dataset contenente le feature di addestramento.
        * *y* (pd.Series): La variabile target (etichette).
        * *groups* (pd.Series): La feature utilizzata per raggruppare i dati ed evitare leakage spaziale/temporale.
        * *n_splits* (int, opzionale): Il numero di fold in cui dividere il dataset. Default a 5.
        * *resampling_strategy* (str): Indica la strategia di resempling da applicare al dataset di training.
                             Di default è 'None' che impica 'classe_weight' = 'balanced' per tutti i modelli.
        * *custom_params* (dict, opzionale): Dizionario di iperparametri personalizzati (es. iniettati da Optuna).
                                         Se None, il modello utilizzerà i parametri di default dal file config.

    **Returns**:
        * *dict*: Un dizionario contenente la media aritmetica delle metriche di valutazione
              (es. log_loss, accuracy, f1, precision, recall, roc-auc) calcolate su tutti i fold.
    """

    if X.empty or y.empty:
        raise ValueError("Attenzione i dataset passati alla cross validation sono vuoti!")

    if len(X) != len(y):
        raise ValueError("Attenzione le lunghezze dei dataset (X e y) nella cross validation non coincidono!")

    if groups.empty:
        raise ValueError("Attenzione gruppo passato alla cross validation vuoto!")

    if isinstance(n_splits, int) is not True:
        raise ValueError("Attenzione, lo split per la cross validation deve essere un intero!")

    if isinstance(custom_params, dict) is not True and custom_params is not None:
        raise TypeError(
            "Attenzione lo spazio degli iperparametri fornita per l'ottimizzazione deve essere un dizionario!"
        )

    preprocessing_config = load_preprocessing_config()
    random_seed_config = load_random_seed_config()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed_config.random_seed_sgkf)

    all_folds_metrics = []

    for train_idx, val_idx in sgkf.split(X, y, groups=groups):
        # print(f"Valutazione fold {fold + 1}/{n_splits}...")

        # ##########################################################
        # ### 1. CREAZIONE DEI DATASET DI TRAINING E DI VALIDAZIONE ###
        # ##########################################################
        # Definisco i dataset di training e di validazione del fold
        X_fold_train, X_fold_val = pd.DataFrame(X).iloc[train_idx], pd.DataFrame(X).iloc[val_idx]
        y_fold_train, y_fold_val = pd.Series(y).iloc[train_idx], pd.Series(y).iloc[val_idx]

        # ###############################################################
        # ### 2. DEFINIZIONE DELLA PIPELINE DI PREPROCESSING PER IL FOLD ###
        # ###############################################################
        # Definisco i vari step della pipeline
        # 1. Preprocessing
        preprocessor = build_stateful_ml_pipeline(preprocessing_config, scaler_strategy=scaler_strategy)
        # 2. strategia di resempling scelto nel file di configurazione (configs/config.py)
        resempler = ResamplerFactory.get_resempler(strategy_name=resampling_strategy)
        # 3. Modello di trainingin cui verifico se vengono passati parametri al modello
        # (per esempio dal file di optimization.py) oppure se pescarli dal file di configurazione
        if custom_params is not None:
            model = ModelFactory.get_model(model_name, **custom_params)
        else:
            model = ModelFactory.get_model(model_name)

        # Aggrego tutti gli steps
        steps = preprocessor.steps.copy()
        if resempler is not None:
            steps.append(("resempler", resempler))
        steps.append(("model", model))

        # Definisco la pipeline con gli steps scritti sopra
        pipeline = ImbPipeline(steps=steps)

        # ############################
        # ### 3. TRAINING DEL MODELLO ###
        # ############################
        pipeline.fit(X_fold_train, y_fold_train)

        # Valuatazuone degli score sul dataset di validazioe del fold (loss, acc, f1, precision, recall, roc-auc)
        fold_scores = evaluate_classification_metrics(pipeline, X_fold_val, y_fold_val)
        all_folds_metrics.append(fold_scores)

    # Trasformo la lista degli score in un DataFrame pandas per facilità di integrazione
    df_metrics = pd.DataFrame(all_folds_metrics)

    # Viene restituita un dizionario di metriche medie
    return df_metrics.mean().to_dict()
