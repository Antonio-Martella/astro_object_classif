import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle

from src.models import BaseModel


def compute_class_weights(y):
    y = pd.Series(y)

    counts = y.value_counts().sort_index()
    total = len(y)
    n_classes = len(counts)

    weights = {cls: total / (n_classes * count) for cls, count in counts.items()}

    return weights


class OptimizerFactory:
    """
    Factory per la creazione di ottimizzatori Keras in base al nome e ai parametri specificati.
    Supporta diversi tipi di ottimizzatori come:
    - Adam
    - AdamW
    - RMSprop
    - SGD
    Altri ottimizzatori possono essere aggiunti facilmente estendendo questa classe.
    Per ognuno di essi, è possibile specificare parametri come: learning rate, momentum, weight decay, tramite kwargs.
    """

    # La funzione prende in input il nome dell'ottimizzatore, il learning rate e altri parametri specifici
    # per ogni ottimizzatore tramite kwargs.
    @staticmethod
    def get_optimizer(optimizer_name, learning_rate, **kwargs):
        # Adam
        if optimizer_name == "adam":
            return tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=kwargs.get("beta_1", 0.9),
                beta_2=kwargs.get("beta_2", 0.999),
            )
        # AdamW
        if optimizer_name == "adamw":
            return tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=kwargs.get("weight_decay", 0.004),
                beta_1=kwargs.get("beta_1", 0.9),
            )
        # RMSprop
        elif optimizer_name == "rmsprop":
            return tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                rho=kwargs.get("rho", 0.9),
                momentum=kwargs.get("momentum_rsprop", 0.0),
            )
        # SGD
        elif optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=kwargs.get("momentum_sgd", 0.9),
                nesterov=kwargs.get("nesterov", True),
            )
        else:
            raise ValueError("Optimizer non supportato, scegli tra: 'adam', 'adamw', 'rmsprop', 'sgd'")


def callback(
    monitor: str = "val_loss",
    factor_reducer: float = 0.5,
    patience_lrreducer: int = 3,
    patience_earlystop: int = 7,
    verbose: int = 0,
):
    """
    Crea una lista di callback per il training del modello Keras, includendo:
    - ReduceLROnPlateau: Riduce il learning rate se la metrica monitorata non migliora
      per un certo numero di epoche consecutive.
    - EarlyStopping: Interrompe il training se la metrica monitorata non migliora
    per un certo numero di epoche consecutive, ripristinando i pesi del modello alla migliore epoca.
    """

    # Riduce il learning rate se la metrica monitorata non migliora per 'patience_lrreducer' epoche consecutive
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor_reducer,
        patience=patience_lrreducer,
        min_lr=1e-6,
        verbose=verbose,
    )
    # Interrompe il training se la metrica monitorata non migliora per 'patience_earlystop' epoche consecutive,
    # ripristinando i pesi del modello alla migliore epoca
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience_earlystop, restore_best_weights=True, verbose=verbose
    )

    return lr_reducer, early_stopper


class LossFunctionFactory:
    """
    Factory per la creazione di funzioni di perdita Keras in base al nome specificato. Supporta una loss function
    molto comune per problemi di classificazione multi-classe (Sparse Categorical Crossentropy),
    ma altre possono essere aggiunte facilmente estendendo questa classe.
    """

    @staticmethod
    def get_lossfunction(lossfunc_name: str = "sparse_categorical_crossentropy"):
        if lossfunc_name == "sparse_categorical_crossentropy":
            return tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            raise ValueError(f"Unknown loss function: {lossfunc_name}")


class DenseNNModel(BaseModel, BaseEstimator, ClassifierMixin):
    """
    Modello di rete neurale densa (fully connected) per la classificazione multi-classe.
    L'architettura e i parametri di training sono configurabili tramite il costruttore, con valori
    di default che possono essere sovrascritti. Supporta callback per il callback di Keras come ReduceLROnPlateau
    e EarlyStopping, e permette di gestire il bilanciamento delle classi tramite class weights. La classe
    implementa i metodi fit, predict e predict_proba per integrarsi con l'ecosistema di scikit-learn,
    e include anche metodi per salvare e caricare il modello. L'architettura è costruita dinamicamente
    in base ai parametri specificati, e il modello è compilato con un ottimizzatore configurabile
    tramite la factory degli ottimizzatori.
    """

    def __init__(self, **kwargs):
        optimizer = kwargs.pop("optimizer", "adam")
        self.optimizer = optimizer

        num_layer = kwargs.pop("num_layer", 2)
        self.num_layer = num_layer

        num_perceptron = kwargs.pop("num_perceptron", 128)
        self.num_perceptron = num_perceptron

        drop_out = kwargs.pop("drop_out", 0.3)
        self.drop_out = drop_out

        batch_norm = kwargs.pop("batch_norm", True)
        self.batch_norm = batch_norm

        learning_rate = kwargs.pop("learning_rate", 1e-2)
        self.learning_rate = learning_rate

        hidden_act_func = kwargs.pop("hidden_act_func", "relu")
        self.hidden_act_func = hidden_act_func

        act_func = kwargs.pop("act_func", "softmax")
        self.act_func = act_func

        class_weights = kwargs.pop("class_weights", True)
        self.class_weights = class_weights

        verbose = kwargs.pop("verbose", 0)
        self.verbose = verbose

        random_seed = kwargs.pop("random_seed", 42)
        self.random_seed = random_seed

        batch_size = kwargs.pop("batch_size", 256)
        self.batch_size = batch_size

        # --- Callback ---
        metric_monitor_callback = kwargs.pop("metric_monitor_callback", "val_loss")
        self.metric_monitor_callback = metric_monitor_callback

        factor_reducer_callback = kwargs.pop("factor_reducer_callback", 0.75)
        self.factor_reducer_callback = factor_reducer_callback

        patience_lrreducer = kwargs.pop("patience_lrreducer", 2)
        self.patience_lrreducer = patience_lrreducer

        patience_earlystop = kwargs.pop("patience_earlystop", 3)
        self.patience_earlystop = patience_earlystop

        verbose_cb = kwargs.pop("verbose_cb", 0)
        self.verbose_cb = verbose_cb

        # Salviamo eventuali altri parametri specifici per l'ottimizzatore (es. beta_1, weight_decay, ecc.)
        # in un dizionario da passare alla factory degli ottimizzatori
        self.kwargs = kwargs

        # Inizializziamo il modello a None, sarà costruito dinamicamente al momento del fit in base alla forma dei dati
        # e al numero di classi del target
        self.model = None

    # Costruiamo dinamicamente l'architettura del modello in base ai parametri specificati e alla
    # forma dei dati di input e al numero di classi del target.
    # Compiliamo il modello con la loss function e l'ottimizzatore configurati tramite le factory.
    def _build_model(self, input_dim, num_classes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        # Aggiungiamo i layer densi con il numero di neuroni decrescente, la funzione di attivazione specificata
        # opzionalmente la batch normalization e il dropout
        # Il numero di neuroni in ogni layer è calcolato come num_perceptron diviso per 2^units
        # per creare un'architettura a imbuto
        for units in range(self.num_layer):
            model.add(tf.keras.layers.Dense(int(self.num_perceptron / 2**units), activation=self.hidden_act_func))
            if self.batch_norm:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.drop_out))

        model.add(tf.keras.layers.Dense(num_classes, activation=self.act_func))

        model.compile(
            optimizer=OptimizerFactory.get_optimizer(
                optimizer_name=self.optimizer, learning_rate=self.learning_rate, **self.kwargs
            ),
            loss=LossFunctionFactory.get_lossfunction(),
            metrics=["accuracy"],
        )

        return model

    def fit(self, X, y, **kwargs):
        # Imposto il seed per la riproducibilità
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.random_seed)

        # Mescoliamo X e y questo affinche il validation split sia rappresentativo e non ci siano pattern
        # spaziali/temporali nei dati o di soli gruppi di dati simili, che potrebbero portare
        # a un overfitting del modello
        X, y = shuffle(X, y, random_state=self.random_seed)

        # deduciamo dinamicamente la forma dei dati!
        input_dim = X.shape[1]

        # identifichiamo le classi univoche per il target
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)

        # setup del modello, per numero di datati in input e numero di classi nel target
        self.model = self._build_model(input_dim, num_classes)

        # calcolo i pesi delle classi
        if self.class_weights:
            weights = compute_class_weights(y)
        else:
            weights = None

        # definisco il modello
        self.history = self.model.fit(
            X,
            y,
            epochs=20,
            batch_size=self.batch_size,
            class_weight=weights,
            shuffle=True,
            validation_split=0.15,
            callbacks=callback(
                monitor=self.metric_monitor_callback,
                factor_reducer=self.factor_reducer_callback,
                patience_lrreducer=self.patience_lrreducer,
                patience_earlystop=self.patience_earlystop,
                verbose=self.verbose_cb,
            ),
            verbose=self.verbose,
        )

        self.is_fitted_ = True

        return self

    def predict(self, X):
        proba = self.model.predict(X, verbose=0)

        return proba.argmax(axis=1)

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)

    def save(self, path):
        self.model.save(path)

    @classmethod
    def load(cls, path):
        # Creiamo un'istanza
        instance = cls()

        # Carichiamo il modello Keras dal disco e sovrascriviamo quello vuoto
        instance.model = tf.keras.models.load_model(path)

        return instance
