from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from configs.schemas import PreprocessingConfig
from src.training.train import fit_and_evaluate_model


@pytest.fixture
def dummy_config():
    model_name = "random_forest"

    X_train = pd.DataFrame({"feature_1": [1, 2, 3, 4], "feature_2": [1, 2, 3, 4]})
    X_test = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [1, 2, 3]})
    y_train_encoded = pd.Series([0, 2, 1, 0], name="class")
    y_test_encoded = pd.Series([1, 1, 2], name="class")

    preprocess_config = MagicMock(spec=PreprocessingConfig)

    custom_params = {"n_estimators": 50}
    resampling_strategy = "smote"
    scaler_strategy = "standard"

    return (
        model_name,
        X_train,
        X_test,
        y_train_encoded,
        y_test_encoded,
        preprocess_config,
        custom_params,
        resampling_strategy,
        scaler_strategy,
    )


class TestFitAndEvaluateModel:
    @pytest.mark.parametrize(
        "field_to_break, bad_value",
        [
            ("model_name", 123),
            ("X_train", "not_a_dataframe"),
            ("X_test", 456),
            ("y_train_encoded", "not_a_series"),
            ("y_test_encoded", [1, 2, 3]),
            ("preprocess_config", "not_a_config"),
            ("custom_params", "not_a_dict"),
            ("resampling_strategy", 42),
            ("scaler_strategy", 3.14),
        ],
    )
    def test_fit_and_evaluate_raises_typeerror_for_invalid_field_type(self, dummy_config, field_to_break, bad_value):
        (
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            preprocess_config,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        ) = dummy_config

        kwargs = {
            "model_name": model_name,
            "X_train": X_train,
            "X_test": X_test,
            "y_train_encoded": y_train_encoded,
            "y_test_encoded": y_test_encoded,
            "preprocess_config": preprocess_config,
            "custom_params": custom_params,
            "resampling_strategy": resampling_strategy,
            "scaler_strategy": scaler_strategy,
        }
        kwargs[field_to_break] = bad_value

        with pytest.raises(TypeError):
            fit_and_evaluate_model(**kwargs)

    def test_fit_and_evaluate_train_dataset_length_error(self, dummy_config):
        (
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            preprocess_config,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        ) = dummy_config

        # train
        y_train_err = y_train_encoded.iloc[:-1]

        with pytest.raises(
            ValueError, match="The number of training samples in X_train and y_train_encoded must be the same"
        ):
            fit_and_evaluate_model(
                model_name,
                X_train,
                X_test,
                y_train_err,
                y_test_encoded,
                preprocess_config,
                custom_params,
                resampling_strategy,
                scaler_strategy,
            )

    def test_fit_and_evaluate_test_dataset_length_error(self, dummy_config):
        (
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            preprocess_config,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        ) = dummy_config

        # test
        y_test_err = y_test_encoded.iloc[:-1]

        with pytest.raises(
            ValueError, match="The number of training samples in X_test and y_test_encoded must be the same"
        ):
            fit_and_evaluate_model(
                model_name,
                X_train,
                X_test,
                y_train_encoded,
                y_test_err,
                preprocess_config,
                custom_params,
                resampling_strategy,
                scaler_strategy,
            )

    def test_fit_and_evaluate_empty_dataset_error(self, dummy_config):
        (
            model_name,
            _,
            X_test,
            _,
            y_test_encoded,
            preprocess_config,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        ) = dummy_config

        X_train_empty = pd.DataFrame(columns=["feature_1", "feature_2"])
        y_train_encoded_empty = pd.Series(name="class")

        with pytest.raises(ValueError, match="Training and test datasets cannot be empty"):
            fit_and_evaluate_model(
                model_name,
                X_train_empty,
                X_test,
                y_train_encoded_empty,
                y_test_encoded,
                preprocess_config,
                custom_params,
                resampling_strategy,
                scaler_strategy,
            )

    @patch("src.training.train.load_preprocessing_config")
    @patch("src.training.train.build_training_pipeline")
    @patch("src.training.train.evaluate_classification_metrics")
    def test_fit_and_evaluate_load_preprocess_config_when_none_provided(
        self,
        mock_evaluate_classification_metrics,
        mock_build_training_pipeline,
        mock_load_preprocessing_config,
        dummy_config,
    ):
        (
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            _,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        ) = dummy_config

        preprocess_config_none = None

        mock_load_preprocessing_config.return_value = MagicMock()
        mock_build_training_pipeline.return_value = MagicMock()
        mock_evaluate_classification_metrics.return_value = MagicMock()

        fit_and_evaluate_model(
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            preprocess_config_none,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        )

        mock_load_preprocessing_config.assert_called_once_with()
        mock_build_training_pipeline.assert_called_once_with(
            mock_load_preprocessing_config.return_value,
            model_name,
            scaler_strategy,
            resampling_strategy,
            custom_params,
        )

    @patch("src.training.train.load_preprocessing_config")
    @patch("src.training.train.build_training_pipeline")
    @patch("src.training.train.evaluate_classification_metrics")
    def test_fit_and_evaluate_load_preprocces_config_when_is_provided(
        self,
        mock_evaluate_classification_metrics,
        mock_build_training_pipeline,
        mock_load_preprocessing_config,
        dummy_config,
    ):
        (
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            preprocess_config,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        ) = dummy_config

        mock_load_preprocessing_config.return_value = MagicMock()
        mock_build_training_pipeline.return_value = MagicMock()
        mock_evaluate_classification_metrics.return_value = MagicMock()

        fit_and_evaluate_model(
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            preprocess_config,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        )

        mock_load_preprocessing_config.assert_not_called()

    @patch("src.training.train.build_training_pipeline")
    def test_fit_and_evaluate_pipeline_definition_error(self, mock_build_training_pipeline, dummy_config):
        (
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            preprocess_config,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        ) = dummy_config

        mock_build_training_pipeline.side_effect = RuntimeError()

        with pytest.raises(ValueError, match="Attention: Problem during PIPELINE definition in training pipeline"):
            fit_and_evaluate_model(
                model_name,
                X_train,
                X_test,
                y_train_encoded,
                y_test_encoded,
                preprocess_config,
                custom_params,
                resampling_strategy,
                scaler_strategy,
            )

    @patch("src.training.train.build_training_pipeline")
    @patch("src.training.train.evaluate_classification_metrics")
    def test_fit_and_evaluate_metric_definition_error(
        self, mock_evaluate_classification_metrics, mock_build_training_pipeline, dummy_config
    ):
        (
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            preprocess_config,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        ) = dummy_config

        mock_build_training_pipeline.return_value = MagicMock()
        mock_evaluate_classification_metrics.side_effect = RuntimeError()

        with pytest.raises(ValueError, match="Attention: Problem during EVALUATION of the trained model"):
            fit_and_evaluate_model(
                model_name,
                X_train,
                X_test,
                y_train_encoded,
                y_test_encoded,
                preprocess_config,
                custom_params,
                resampling_strategy,
                scaler_strategy,
            )

    @patch("src.training.train.build_training_pipeline")
    @patch("src.training.train.evaluate_classification_metrics")
    def test_orchestrates_full_pipeline_creation(
        self, mock_evaluate_classification_metrics, mock_build_training_pipeline, dummy_config
    ):
        (
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            preprocess_config,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        ) = dummy_config

        pipeline_mock = MagicMock(spec=ImbPipeline)
        mock_build_training_pipeline.return_value = pipeline_mock
        mock_evaluate_classification_metrics.return_value = MagicMock(spec=dict)

        pipeline, metrics = fit_and_evaluate_model(
            model_name,
            X_train,
            X_test,
            y_train_encoded,
            y_test_encoded,
            preprocess_config,
            custom_params,
            resampling_strategy,
            scaler_strategy,
        )

        assert isinstance(pipeline, ImbPipeline)
        assert isinstance(metrics, dict)

        mock_build_training_pipeline.assert_called_once_with(
            preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params
        )
        pipeline_mock.fit.assert_called_once_with(X_train, y_train_encoded)
        mock_evaluate_classification_metrics.assert_called_once_with(pipeline_mock, X_test, y_test_encoded)
