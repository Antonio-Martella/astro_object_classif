from unittest.mock import MagicMock, patch

import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from configs.schemas import PreprocessingConfig
from src.training.pipeline import build_training_pipeline


@pytest.fixture
def dummy_configuration():
    preprocess_config = MagicMock(spec=PreprocessingConfig)
    model_name = "random_forest"
    scaler_strategy = "standard"
    resampling_strategy = "smote"
    custom_params = {"n_estimators": 50}

    return preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params


class TestBuildTrainingPipeline:
    def test_build_training_pipeline_typeerror(self, dummy_configuration):
        _, model_name, scaler_strategy, resampling_strategy, custom_params = dummy_configuration
        preprocess_config_err = "unknown_type"

        with pytest.raises(TypeError):
            build_training_pipeline(
                preprocess_config_err, model_name, scaler_strategy, resampling_strategy, custom_params
            )

    @patch("src.training.pipeline.load_preprocessing_config")
    @patch("src.training.pipeline.build_stateful_ml_pipeline")
    @patch("src.training.pipeline.ResamplerFactory.get_resampler")
    @patch("src.training.pipeline.ModelFactory.get_model")
    def test_uses_default_configs_when_none_provided(
        self,
        mock_get_model,
        mock_get_resampler,
        mock_build_stateful_ml_pipeline,
        mock_load_preprocessing_config,
        dummy_configuration,
    ):
        _, model_name, scaler_strategy, resampling_strategy, custom_params = dummy_configuration
        preprocess_config = None

        mock_load_preprocessing_config.return_value = MagicMock(spec=PreprocessingConfig)
        mock_build_stateful_ml_pipeline.return_value = MagicMock()
        mock_get_resampler.return_value = MagicMock()
        mock_get_model.return_value = MagicMock()

        build_training_pipeline(preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params)

        mock_load_preprocessing_config.assert_called_once_with()
        mock_build_stateful_ml_pipeline.assert_called_once_with(
            mock_load_preprocessing_config.return_value, scaler_strategy
        )

    @patch("src.training.pipeline.build_stateful_ml_pipeline")
    def test_preprocessor_pipeline_try_except_error(self, mock_build_stateful_ml_pipeline, dummy_configuration):
        preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params = dummy_configuration

        mock_build_stateful_ml_pipeline.side_effect = RuntimeError()

        with pytest.raises(ValueError, match="PREPROCESSOR"):
            build_training_pipeline(preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params)

    @patch("src.training.pipeline.build_stateful_ml_pipeline")
    @patch("src.training.pipeline.ResamplerFactory.get_resampler")
    def test_resampler_try_except_error(self, mock_get_resampler, mock_build_stateful_ml_pipeline, dummy_configuration):
        preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params = dummy_configuration

        mock_build_stateful_ml_pipeline.return_value = MagicMock()
        mock_get_resampler.side_effect = RuntimeError()

        with pytest.raises(ValueError, match="RESAMPLER"):
            build_training_pipeline(preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params)

    @patch("src.training.pipeline.build_stateful_ml_pipeline")
    @patch("src.training.pipeline.ResamplerFactory.get_resampler")
    @patch("src.training.pipeline.ModelFactory.get_model")
    def test_model_try_except_error(
        self, mock_get_model, mock_get_resampler, mock_build_stateful_ml_pipeline, dummy_configuration
    ):
        preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params = dummy_configuration

        mock_build_stateful_ml_pipeline.return_value = MagicMock()
        mock_get_resampler.return_value = MagicMock()
        mock_get_model.side_effect = RuntimeError()

        with pytest.raises(ValueError, match="MODEL"):
            build_training_pipeline(preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params)

    @patch("src.training.pipeline.build_stateful_ml_pipeline")
    @patch("src.training.pipeline.ResamplerFactory.get_resampler")
    @patch("src.training.pipeline.ModelFactory.get_model")
    def test_resampler_when_none_provided(
        self, mock_get_model, mock_get_resampler, mock_build_stateful_ml_pipeline, dummy_configuration
    ):
        preprocess_config, model_name, scaler_strategy, _, custom_params = dummy_configuration
        resampling_strategy = None

        mock_build_stateful_ml_pipeline.return_value = MagicMock()
        mock_get_resampler.return_value = MagicMock()
        mock_get_model.return_value = MagicMock()

        build_training_pipeline(preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params)

        mock_get_resampler.assert_not_called()

    @patch("src.training.pipeline.build_stateful_ml_pipeline")
    @patch("src.training.pipeline.ResamplerFactory.get_resampler")
    @patch("src.training.pipeline.ModelFactory.get_model")
    def test_resampler_weigthed_not_append_step_pipeline(
        self, mock_get_model, mock_get_resampler, mock_build_stateful_ml_pipeline, dummy_configuration
    ):
        preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params = dummy_configuration

        mock_preprocessor = MagicMock()
        mock_preprocessor.steps = [("scaler", MagicMock())]
        mock_build_stateful_ml_pipeline.return_value = mock_preprocessor

        mock_get_resampler.return_value = None
        mock_get_model.return_value = MagicMock()

        pipeline = build_training_pipeline(
            preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params
        )

        assert "scaler" in pipeline.named_steps
        assert "resampler" not in pipeline.named_steps
        assert "model" in pipeline.named_steps

    @patch("src.training.pipeline.build_stateful_ml_pipeline")
    @patch("src.training.pipeline.ResamplerFactory.get_resampler")
    @patch("src.training.pipeline.ModelFactory.get_model")
    def test_instantiates_model_with_default_params_when_custom_params_is_none(
        self, mock_get_model, mock_get_resampler, mock_build_stateful_ml_pipeline, dummy_configuration
    ):
        preprocess_config, model_name, scaler_strategy, resampling_strategy, _ = dummy_configuration
        custom_params_none = None

        mock_build_stateful_ml_pipeline.return_value = MagicMock()
        mock_get_resampler.return_value = MagicMock()
        mock_get_model.return_value = MagicMock()

        build_training_pipeline(preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params_none)

        mock_get_model.assert_called_once_with(model_name)

    @patch("src.training.pipeline.build_stateful_ml_pipeline")
    @patch("src.training.pipeline.ResamplerFactory.get_resampler")
    @patch("src.training.pipeline.ModelFactory.get_model")
    def test_instantiates_model_with_custom_params_when_provided(
        self, mock_get_model, mock_get_resampler, mock_build_stateful_ml_pipeline, dummy_configuration
    ):
        preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params = dummy_configuration

        mock_build_stateful_ml_pipeline.return_value = MagicMock()
        mock_get_resampler.return_value = MagicMock()
        mock_get_model.return_value = MagicMock()

        build_training_pipeline(preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params)

        mock_get_model.assert_called_once_with(model_name, n_estimators=50)

    @patch("src.training.pipeline.load_preprocessing_config")
    @patch("src.training.pipeline.build_stateful_ml_pipeline")
    @patch("src.training.pipeline.ResamplerFactory.get_resampler")
    @patch("src.training.pipeline.ModelFactory.get_model")
    def test_orchestrates_full_pipeline_creation(
        self,
        mock_get_model,
        mock_get_resampler,
        mock_build_stateful_ml_pipeline,
        mock_load_preprocessing_config,
        dummy_configuration,
    ):
        preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params = dummy_configuration

        mock_load_preprocessing_config.return_value = MagicMock()

        mock_preprocessor = MagicMock()
        mock_preprocessor.steps = [("imputer", MagicMock()), (scaler_strategy, MagicMock())]
        mock_build_stateful_ml_pipeline.return_value = mock_preprocessor
        mock_get_resampler.return_value = MagicMock()
        mock_get_model.return_value = MagicMock()

        pipeline = build_training_pipeline(
            preprocess_config, model_name, scaler_strategy, resampling_strategy, custom_params
        )

        mock_build_stateful_ml_pipeline.assert_called_once_with(preprocess_config, scaler_strategy)
        mock_get_resampler.assert_called_once_with(strategy_name=resampling_strategy)
        mock_get_model.assert_called_once_with(model_name, n_estimators=50)

        assert "imputer" in pipeline.named_steps
        assert scaler_strategy in pipeline.named_steps
        assert "resampler" in pipeline.named_steps
        assert "model" in pipeline.named_steps

        assert isinstance(pipeline, ImbPipeline)
