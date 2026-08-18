from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from configs.schemas import PreprocessingConfig
from src.training.cross_validation import run_cross_validation


@pytest.fixture
def dummy_cross_validation_config():
    model_name = "random_forest"
    X = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5, 6], "feature_2": [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 2, 1, 1, 1, 0], name="class")
    groups = pd.Series([151, 152, 155, 156, 157, 158], name="field_ID")
    n_splits = 3
    resampling_strategy = "smote"
    scaler_strategy = "standard"
    custom_params = {"n_estimators": 50}
    preprocessing_config = MagicMock(spec=PreprocessingConfig)

    return model_name, X, y, groups, n_splits, resampling_strategy, scaler_strategy, custom_params, preprocessing_config


class TestRunCrossValidation:
    @pytest.mark.parametrize(
        "field_to_break, bad_value",
        [
            ("model_name", 1),
            ("X", pd.Series(1)),
            ("y", pd.DataFrame(columns=["feature_1"])),
            ("groups", [1, 2, 3]),
            ("n_splits", 1.1),
            ("resampling_strategy", [1, 2, 3]),
            ("scaler_strategy", 3.4),
            ("custom_params", int(1.1)),
            ("preprocessing_config", bool),
        ],
    )
    def test_fit_and_evaluate_raises_typeerror_for_invalid_field_type(
        self, dummy_cross_validation_config, field_to_break, bad_value
    ):
        (
            model_name,
            X,
            y,
            groups,
            n_splits,
            resampling_strategy,
            scaler_strategy,
            custom_params,
            preprocessing_config,
        ) = dummy_cross_validation_config
        preprocessing_config.columns_to_scale = ["feature_1", "feature_2"]

        kwargs = {
            "model_name": model_name,
            "X": X,
            "y": y,
            "groups": groups,
            "n_splits": n_splits,
            "resampling_strategy": resampling_strategy,
            "scaler_strategy": scaler_strategy,
            "custom_params": custom_params,
            "preprocessing_config": preprocessing_config,
        }

        kwargs[field_to_break] = bad_value

        with pytest.raises(TypeError):
            run_cross_validation(**kwargs)

    @pytest.mark.parametrize(
        "dataset_to_break, empty_dataset",
        [
            ("X", pd.DataFrame(columns=["feature_1", "feature_2"])),
            ("y", pd.Series(name="class")),
        ],
    )
    def test_run_cross_validation_empty_dataset_error(
        self, dataset_to_break, empty_dataset, dummy_cross_validation_config
    ):
        (
            model_name,
            X,
            y,
            groups,
            n_splits,
            resampling_strategy,
            scaler_strategy,
            custom_params,
            preprocessing_config,
        ) = dummy_cross_validation_config

        kwargs = {
            "model_name": model_name,
            "X": X,
            "y": y,
            "groups": groups,
            "n_splits": n_splits,
            "resampling_strategy": resampling_strategy,
            "scaler_strategy": scaler_strategy,
            "custom_params": custom_params,
            "preprocessing_config": preprocessing_config,
        }

        kwargs[dataset_to_break] = empty_dataset

        with pytest.raises(ValueError, match="Please note that the datasets passed to cross-validation are empty!"):
            run_cross_validation(**kwargs)

    def test_run_cross_validation_length_datasets_error(self, dummy_cross_validation_config):
        (
            model_name,
            X,
            y,
            groups,
            n_splits,
            resampling_strategy,
            scaler_strategy,
            custom_params,
            preprocessing_config,
        ) = dummy_cross_validation_config

        y_err = y.iloc[:-1]

        with pytest.raises(ValueError, match="Please note that the lengths of the datasets"):
            run_cross_validation(
                model_name,
                X,
                y_err,
                groups,
                n_splits,
                resampling_strategy,
                scaler_strategy,
                custom_params,
                preprocessing_config,
            )

    def test_run_cross_validation_target_class_error(self, dummy_cross_validation_config):
        (
            model_name,
            X,
            y,
            groups,
            n_splits,
            resampling_strategy,
            scaler_strategy,
            custom_params,
            preprocessing_config,
        ) = dummy_cross_validation_config

        y_unique = pd.Series([0 for _ in range(len(X))])

        with pytest.raises(ValueError, match="y must contain at least 2 unique classes"):
            run_cross_validation(
                model_name,
                X,
                y_unique,
                groups,
                n_splits,
                resampling_strategy,
                scaler_strategy,
                custom_params,
                preprocessing_config,
            )

    def test_run_cross_validation_empty_group_error(self, dummy_cross_validation_config):
        model_name, X, y, _, n_splits, resampling_strategy, scaler_strategy, custom_params, preprocessing_config = (
            dummy_cross_validation_config
        )

        groups_empty = pd.Series(name="field_ID")

        with pytest.raises(ValueError, match="Group passed to empty cross-validation!"):
            run_cross_validation(
                model_name,
                X,
                y,
                groups_empty,
                n_splits,
                resampling_strategy,
                scaler_strategy,
                custom_params,
                preprocessing_config,
            )

    def test_run_cross_validation_length_group_error(self, dummy_cross_validation_config):
        (
            model_name,
            X,
            y,
            groups,
            n_splits,
            resampling_strategy,
            scaler_strategy,
            custom_params,
            preprocessing_config,
        ) = dummy_cross_validation_config

        groups_err = groups.iloc[:-1]

        with pytest.raises(ValueError, match="The length of 'groups' series does not match the dataset X length!"):
            run_cross_validation(
                model_name,
                X,
                y,
                groups_err,
                n_splits,
                resampling_strategy,
                scaler_strategy,
                custom_params,
                preprocessing_config,
            )

    def test_run_cross_validation_n_split_less_than_two_error(self, dummy_cross_validation_config):
        model_name, X, y, groups, _, resampling_strategy, scaler_strategy, custom_params, preprocessing_config = (
            dummy_cross_validation_config
        )

        n_split_err = 1

        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            run_cross_validation(
                model_name,
                X,
                y,
                groups,
                n_split_err,
                resampling_strategy,
                scaler_strategy,
                custom_params,
                preprocessing_config,
            )

    def test_run_cross_validation_n_split_length_greater_than_group_error(self, dummy_cross_validation_config):
        model_name, X, y, groups, _, resampling_strategy, scaler_strategy, custom_params, preprocessing_config = (
            dummy_cross_validation_config
        )

        n_split_err = groups.nunique() + 1

        with pytest.raises(ValueError, match="Number of unique groups"):
            run_cross_validation(
                model_name,
                X,
                y,
                groups,
                n_split_err,
                resampling_strategy,
                scaler_strategy,
                custom_params,
                preprocessing_config,
            )

    @patch("src.training.cross_validation.load_preprocessing_config")
    @patch("src.training.cross_validation.load_random_seed_config")
    @patch("src.training.cross_validation.StratifiedGroupKFold")
    @patch("src.training.cross_validation.build_training_pipeline")
    @patch("src.training.cross_validation.evaluate_classification_metrics")
    def test_run_cross_validation_load_preprocess_config_when_none_provided(
        self,
        mock_evaluate_classification_metrics,
        mock_build_training_pipeline,
        mock_StratifiedGroupKFold,
        mock_load_random_seed_config,
        mock_load_preprocessing_config,
        dummy_cross_validation_config,
    ):
        model_name, X, y, groups, n_split, resampling_strategy, scaler_strategy, custom_params, _ = (
            dummy_cross_validation_config
        )
        preprocessing_config_none = None

        mock_load_random_seed_config.return_value = MagicMock()
        mock_StratifiedGroupKFold.return_value = MagicMock()
        mock_build_training_pipeline.return_value = MagicMock()
        mock_evaluate_classification_metrics.return_value = MagicMock()

        run_cross_validation(
            model_name,
            X,
            y,
            groups,
            n_split,
            resampling_strategy,
            scaler_strategy,
            custom_params,
            preprocessing_config_none,
        )

        mock_load_preprocessing_config.assert_called_once_with()

    @patch("src.training.cross_validation.load_preprocessing_config")
    @patch("src.training.cross_validation.load_random_seed_config")
    @patch("src.training.cross_validation.StratifiedGroupKFold")
    @patch("src.training.cross_validation.build_training_pipeline")
    @patch("src.training.cross_validation.evaluate_classification_metrics")
    def test_run_cross_validation_load_preprocess_config_when_is_provided(
        self,
        mock_evaluate_classification_metrics,
        mock_build_training_pipeline,
        mock_StratifiedGroupKFold,
        mock_load_random_seed_config,
        mock_load_preprocessing_config,
        dummy_cross_validation_config,
    ):
        model_name, X, y, groups, n_split, resampling_strategy, scaler_strategy, custom_params, preprocessing_config = (
            dummy_cross_validation_config
        )

        mock_load_random_seed_config.return_value = MagicMock()
        mock_StratifiedGroupKFold.return_value = MagicMock()
        mock_build_training_pipeline.return_value = MagicMock()
        mock_evaluate_classification_metrics.return_value = MagicMock()

        run_cross_validation(
            model_name,
            X,
            y,
            groups,
            n_split,
            resampling_strategy,
            scaler_strategy,
            custom_params,
            preprocessing_config,
        )

        mock_load_preprocessing_config.assert_not_called()

    @patch("src.training.cross_validation.load_random_seed_config")
    def test_run_cross_validation_load_randoseed_config_raise_error(
        self, mock_load_random_seed_config, dummy_cross_validation_config
    ):
        model_name, X, y, groups, n_split, resampling_strategy, scaler_strategy, custom_params, preprocessing_config = (
            dummy_cross_validation_config
        )

        mock_load_random_seed_config.side_effect = ValueError()

        with pytest.raises(ValueError, match="Could not load random_seeds to handle reproducibility"):
            run_cross_validation(
                model_name,
                X,
                y,
                groups,
                n_split,
                resampling_strategy,
                scaler_strategy,
                custom_params,
                preprocessing_config,
            )

    @patch("src.training.cross_validation.load_random_seed_config")
    @patch("src.training.cross_validation.StratifiedGroupKFold")
    def test_run_cross_validation_StratifiedGroupKFold_raise_error(
        self, mock_StratifiedGroupKFold, mock_load_random_seed_config, dummy_cross_validation_config
    ):
        model_name, X, y, groups, n_split, resampling_strategy, scaler_strategy, custom_params, preprocessing_config = (
            dummy_cross_validation_config
        )

        mock_load_random_seed_config.return_value = MagicMock()
        mock_StratifiedGroupKFold.side_effect = ValueError()

        with pytest.raises(RuntimeError, match="group k fold failed"):
            run_cross_validation(
                model_name,
                X,
                y,
                groups,
                n_split,
                resampling_strategy,
                scaler_strategy,
                custom_params,
                preprocessing_config,
            )

    @patch("src.training.cross_validation.load_random_seed_config")
    @patch("src.training.cross_validation.StratifiedGroupKFold")
    @patch("src.training.cross_validation.build_training_pipeline")
    def test_run_cross_validation_build_trainin_pipeline_raise_error(
        self,
        mock_build_training_pipeline,
        mock_StratifiedGroupKFold,
        mock_load_random_seed_config,
        dummy_cross_validation_config,
    ):
        model_name, X, y, groups, n_split, resampling_strategy, scaler_strategy, custom_params, preprocessing_config = (
            dummy_cross_validation_config
        )

        mock_load_random_seed_config.return_value = MagicMock()
        mock_sgkf_instance = MagicMock()
        mock_sgkf_instance.split.return_value = [([0, 1, 2, 3], [4, 5])]
        mock_StratifiedGroupKFold.return_value = mock_sgkf_instance
        mock_build_training_pipeline.side_effect = ValueError()

        with pytest.raises(RuntimeError, match="the cross-validation PIPELINE failed on fold"):
            run_cross_validation(
                model_name,
                X,
                y,
                groups,
                n_split,
                resampling_strategy,
                scaler_strategy,
                custom_params,
                preprocessing_config,
            )

    @patch("src.training.cross_validation.load_random_seed_config")
    @patch("src.training.cross_validation.StratifiedGroupKFold")
    @patch("src.training.cross_validation.build_training_pipeline")
    @patch("src.training.cross_validation.evaluate_classification_metrics")
    def test_run_cross_validation_evaluation_metric_raise_error(
        self,
        mock_evaluate_classification_metrics,
        mock_build_training_pipeline,
        mock_StratifiedGroupKFold,
        mock_load_random_seed_config,
        dummy_cross_validation_config,
    ):
        model_name, X, y, groups, n_split, resampling_strategy, scaler_strategy, custom_params, preprocessing_config = (
            dummy_cross_validation_config
        )

        mock_random_seed = MagicMock()
        mock_random_seed.random_seed_sgkf = MagicMock()
        mock_load_random_seed_config.return_value = mock_random_seed

        mock_sgkf_instance = MagicMock()
        mock_sgkf_instance.split.return_value = [([0, 1, 2, 3], [4, 5])]
        mock_StratifiedGroupKFold.return_value = mock_sgkf_instance

        mock_piepline = MagicMock()
        mock_piepline.fit.return_value = MagicMock()
        mock_build_training_pipeline.return_value = mock_piepline

        mock_evaluate_classification_metrics.side_effect = ValueError()

        with pytest.raises(RuntimeError, match="the cross-validation EVALUATION failed on fold"):
            run_cross_validation(
                model_name,
                X,
                y,
                groups,
                n_split,
                resampling_strategy,
                scaler_strategy,
                custom_params,
                preprocessing_config,
            )

    @patch("src.training.cross_validation.load_random_seed_config")
    @patch("src.training.cross_validation.StratifiedGroupKFold")
    @patch("src.training.cross_validation.build_training_pipeline")
    @patch("src.training.cross_validation.evaluate_classification_metrics")
    def test_run_cross_validation_orchestrates_full_pipeline_creation(
        self,
        mock_evaluate_classification_metrics,
        mock_build_training_pipeline,
        mock_StratifiedGroupKFold,
        mock_load_random_seed_config,
        dummy_cross_validation_config,
    ):
        (
            model_name,
            X,
            y,
            groups,
            n_splits,
            resampling_strategy,
            scaler_strategy,
            custom_params,
            preprocessing_config,
        ) = dummy_cross_validation_config

        mock_random_seed = MagicMock()
        mock_random_seed.random_seed_sgkf = MagicMock()
        mock_load_random_seed_config.return_value = mock_random_seed

        mock_sgkf_instance = MagicMock()
        mock_sgkf_instance.split.return_value = [([0, 1, 2, 3], [4, 5])]
        mock_StratifiedGroupKFold.return_value = mock_sgkf_instance

        mock_piepline = MagicMock()
        mock_piepline.fit.return_value = MagicMock()
        mock_build_training_pipeline.return_value = mock_piepline

        mock_evaluate_classification_metrics.return_value = MagicMock()

        run_cross_validation(
            model_name,
            X,
            y,
            groups,
            n_splits,
            resampling_strategy,
            scaler_strategy,
            custom_params,
            preprocessing_config,
        )

        mock_load_random_seed_config.assert_called_once_with()
        mock_StratifiedGroupKFold.assert_called_once_with(
            n_splits=n_splits, shuffle=True, random_state=mock_load_random_seed_config.return_value.random_seed_sgkf
        )
        mock_build_training_pipeline.assert_called_with(
            preprocessing_config, model_name, scaler_strategy, resampling_strategy, custom_params
        )

        args, _ = mock_evaluate_classification_metrics.call_args
        assert args[0] is mock_build_training_pipeline.return_value
        pd.testing.assert_frame_equal(args[1], X.iloc[[4, 5]])
        pd.testing.assert_series_equal(args[2], y.iloc[[4, 5]])
