import argparse
import logging
import os
import yaml
from pathlib import Path
from datetime import datetime
import uvicorn

from configs.paths import DataPathConfig
from src.data.make_datasets import make_datasets
from src.data.retraining_data_collector import consolidate_batches_for_retraining
from src.optimization.search import run_optimization_pipeline
from src.monitoring.monitoring_pipeline import run_monitoring
from src.api.main import app
from src.predict.batch_predict import run_prediction
from src.predict.streaming_predict import run_single_prediction
from src.training.train import run_training
from src.utils.logger import setup_logger
from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    # Parser initialization
    parser = argparse.ArgumentParser(
        description="MLOps orchestrator for the classification of astronomical objects."
    )

    # Addition of the --mode argument
    parser.add_argument(
        "--mode",
        type = str,
        required = True,
        choices = [
            "make-datasets", 
            "monitoring", 
            "optimization", 
            "api", 
            "prediction", 
            "single-prediction", 
            "training"
        ],
        help = "Select the pipeline stage to execute.",
    )

    # # Addition of the --batch argument for monitoring
    parser.add_argument(
        "--batch",
        type=str,
        nargs="+",
        default=None,
        help="One or more batch files to monitor (e.g., --batch day_6.csv day_7.csv). If omitted, " \
            "all of them are monitored.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        nargs="+",
        default=None,
        help="Path to the dataset to be predicted.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Classification_Astro_Model",
        help="Name of the model registered on MLflow to be used for inference (default: Classification_Astro_Model)",
    )

    # Return the arguments read from the terminal.
    return parser.parse_args()

def optimization_setup(path_config: DataPathConfig | None = None) -> None:
    """
    """
    validate_type(path_config=(path_config, (DataPathConfig, type(None))))

    path_config = DataPathConfig() if path_config is None else path_config

    # crero il path se non presente nella directory
    Path(path_config.optimization_dir_log).parent.mkdir(parents=True, exist_ok=True)

    # Chiamo la run e la directory secondo l'ora in cui l'ottimizzazione è stata fatta partire
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_timestamp = path_config.optimization_dir_log / f"run_{run_timestamp}"

    # Creo i path dei log 
    run_log_file = dir_timestamp / f"run_{run_timestamp}.log"
    trials_log_file = dir_timestamp / f"trials_{run_timestamp}.log"

    # Definisco il logger
    setup_logger(
        run_log_file_path=run_log_file, 
        handler_mode="a",
        main_log_file_path=path_config.main_log
    )

    # Avvio la pipeline di ottimizzazione
    logger.info("Starting the search with the optimization of the best model...")
    run_optimization_pipeline(run_log_file, trials_log_file)

def ask_confirmation(question: str) -> bool:
    """
    """
    validate_type(question=(question, str))
    while True:
        answer = input(f"\n{question} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            logger.info("User confirmed with 'yes'. Proceeding with the operation.")
            return True
        if answer in ("n", "no"):
            logger.info("User rejected the confirmation. Aborting the operation.")
            return False
        logger.info("Invalid input. Enter 'y' (yes) or 'n' (no).")


def main(path_config: DataPathConfig | None = None) -> None:
    validate_type(path_config=(path_config, (DataPathConfig, type(None))))
    args = parse_arguments()

    Path(path_config.make_dataset_log).parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "make-datasets":
        setup_logger(
            run_log_file_path=path_config.make_dataset_log, 
            handler_mode="a",
            main_log_file_path=path_config.main_log
        )
        try:
            logger.info("Starting dataset creation...")
            make_datasets()
        except Exception as e:
            logger.error("It was not possible to complete the dataset construction process.")
            raise RuntimeError(f"Dataset creation process failed:: {e}") from e
        logger.info("The datasets have been successfully downloaded, split, and cleaned.")
        logger.info("-------------------------------------------------------------------")

    # Avvia la fase di monitoring sui batch file presenti in 'data/production/daily_batch/'
    elif args.mode == "monitoring":
        Path(path_config.monitoring_log).parent.mkdir(parents=True, exist_ok=True)
        setup_logger(
            run_log_file_path=path_config.monitoring_log, 
            handler_mode="a",
            main_log_file_path=path_config.main_log
        )
        try:
            logger.info("Starting monitoring process...")
            retraining, batch_name = run_monitoring(batch_file=args.batch)
            if retraining:
                logger.info(f"Retraining required (triggered at '{batch_name}').")
                if ask_confirmation(
                    f"Retraining required (triggered at '{batch_name}'). "
                    "Do you want to update the training dataset by consolidating the monitored batches?"
                ):
                    try: 
                        logger.info("Starting training dataset consolidation...")
                        consolidate_batches_for_retraining(batch_name)
                        if ask_confirmation(
                            "Training dataset updated. "
                            "Do you want to start the model retraining and hyperparameter optimization now?"
                        ):
                            logger.info("Retraining has been requested.")
                            try:
                                logger.info("Starting retraining...")
                                optimization_setup()
                            except Exception as e:
                                logger.error("Error during the retraining phase")
                                raise RuntimeError(f"Failed retraining: {e}") from e 
                    except Exception as e:
                        logger.error("Error while consolidating dataset following retraining trigger")
                        raise RuntimeError(f"Dataset consolidation failed:: {e}") from e
        except Exception as e:
            logger.error("Error during the monitoring process.")
            raise RuntimeError(f"Monitoring process failed: {e}") from e
        logger.info("Monitoring process completed.")
        logger.info("-------------------------------------------------------------------")

    elif args.mode == "optimization":
        try:
            optimization_setup()
        except Exception as e:
            logger.error("Optimization did not take place.")
            raise RuntimeError(f"Error during the optimization phase: {e}") from e
        logger.info("Hyperparameter optimization process completed.")
        logger.info("-------------------------------------------------------------------")
        
    elif args.mode == "api":
        Path(path_config.api_log).parent.mkdir(parents=True, exist_ok=True)
        setup_logger(
            run_log_file_path=path_config.api_log, 
            handler_mode="a",
            main_log_file_path=path_config.main_log
        )
        try:
            logger.info("Starting API interface..")
            os.environ["MODEL_NAME"] = args.model_name
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8000,
            )
        except Exception as e:
            logger.error("Error opening the API interface")
            raise RuntimeError(f"Error opening the API interface: {e}") from e
        logger.info("API session ended.")
        logger.info("-------------------------------------------------------------------")

    elif args.mode == "prediction":
        Path(path_config.inference_log).parent.mkdir(parents=True, exist_ok=True)
        setup_logger(
            run_log_file_path=path_config.inference_log, 
            handler_mode="a",
            main_log_file_path=path_config.main_log
        )
        try:
            logger.info("Start inference session...")
            for path in args.dataset_path:
                run_prediction(input_path=path, model_name=args.model_name)
        except Exception as e:
            logger.error("Error during the inference phase")
            raise RuntimeError(f"Unsuccessful inference process: {e}") from e
        logger.info("Inference session ended.")
        logger.info("-------------------------------------------------------------------")

    elif args.mode == "single-prediction":
        Path(path_config.inference_log).parent.mkdir(parents=True, exist_ok=True)
        setup_logger(
            run_log_file_path=path_config.inference_log, 
            handler_mode="a",
            main_log_file_path=path_config.main_log
        )
        try:
            run_single_prediction(
                model_name=args.model_name
            )
        except Exception as e:
            logger.error(f"Error during evaluation of the single record.")
            raise ValueError(f"Prediction single record failed: {e}") from e

        logger.info("Inference session ended.")
        logger.info("-------------------------------------------------------------------")
    elif args.mode == "training":
        Path(path_config.training_log).parent.mkdir(parents=True, exist_ok=True)
        setup_logger(
            run_log_file_path=path_config.training_log, 
            handler_mode="a",
            main_log_file_path=path_config.main_log
        )
        logger.info("Start training...")
        # copio la lista di tutti i modelli supportati dalla repository
        with open(path_config.names_defined_models, "r", encoding="utf-8") as file:
            models = yaml.safe_load(file)["names_defined_models"]

        # Ci facciamo dire quale sia il modello scelto per il training
        model_name = input(f"Choose a model from {models}: ").strip()

        # Verifica che effettivamente il nome corrisponda ad uno dei supportati
        while True:
            if model_name not in models:
                model_name = input(f"Error, choose one of the following models {models}: ").strip()
                continue
            else:
                break

        logger.info(f"The model selected to be train is: {model_name}")
        params_prompt = (
            "Enter hyperparameters separated by comma\n"
            "(e.g. n_estimators: 50, criterion: gini, learning_rate: 0.05, bootstrap: true)\n"
            "or press ENTER to use default parameters: "
        )
        custom_params_str = input(params_prompt).strip()
        if not custom_params_str:
            custom_params_dict = None
            logger.info("Using default hyperparameters.")
        else:
            try:
                custom_params_dict = yaml.safe_load(f"{{{custom_params_str}}}")
                logger.info(f"Chosen hyperparameters: {custom_params_dict}")
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid hyperparameters format: {e}") from e
        try:
            run_training(
                model_name, 
                custom_params_dict,
            )
        except Exception as e:
            logger.error("Error during training of the selected model.")
            raise ValueError(f"Training failed: {e}") from e

if __name__ == "__main__":
    path_config=DataPathConfig()
    main(path_config)