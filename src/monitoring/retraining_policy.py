import logging
from typing import Any

from configs.pipeline_config import RetrainingPolicyConfig
from configs.pipeline_loader import load_retraining_policy_config

from src.utils.validate_type import validate_type

logger = logging.getLogger(__name__)


class AstroRetrainingPolicy:
    def __init__(
            self,
            retraining_policy_config: RetrainingPolicyConfig | None = None,
            verbose: int = 2,
        ) -> None:
        """
        Class to evaluate the retraining policy based on performance and drift metrics.

        Args:
            retraining_policy_config (RetrainingPolicyConfig | None): Configuration for the retraining policy. 
                                                                      If None, it will load the default configuration.
            verbose (int): Verbosity level for logging (Default is 2):
                - 0 for logging all messages
                - 1 for only warnings
                - 2 for only errors
        
        Raises:
            ValueError: If any of the configuration parameters are not structured correctly.
        """
        validate_type(
            retraining_policy_config=(retraining_policy_config, (RetrainingPolicyConfig,type(None))),
            verbose=(verbose, int),
        )

        retraining_policy_config = \
            load_retraining_policy_config() if retraining_policy_config is None else retraining_policy_config

        self.metric_to_monitor = retraining_policy_config.metric_to_monitor
        self.consecutive_critical_threshold = retraining_policy_config.consecutive_critical_threshold
        self.catastrophic_drop_threshold = retraining_policy_config.catastrophic_drop_threshold
        self.max_drift_ratio = retraining_policy_config.max_drift_ratio
        self.cooldown_batches = retraining_policy_config.cooldown_batches

        if not isinstance(self.metric_to_monitor, str):
            raise ValueError("Attention: 'metric_to_monitor' should be a string.")

        if not isinstance(self.consecutive_critical_threshold, int) or self.consecutive_critical_threshold <= 0:
            raise ValueError("Attention: 'consecutive_critical_threshold' should be a positive integer.")

        if (
            not isinstance(self.catastrophic_drop_threshold, float) or 
            self.catastrophic_drop_threshold <= 0 or 
            self.catastrophic_drop_threshold > 1
        ):
            raise ValueError("Attention: 'catastrophic_drop_threshold' should be a float between (0,1].")

        if (
            not isinstance(self.max_drift_ratio, float) or 
            self.max_drift_ratio <= 0 or 
            self.max_drift_ratio > 1
        ):
            raise ValueError("Attention: 'max_drift_ratio' should be a float between (0,1].")

        if (
            not isinstance(self.cooldown_batches, int) or 
            self.cooldown_batches <= 0
        ):
            raise ValueError("Attention: 'cooldown_batches' should be a positive integer.")

        self.verbose = verbose
        if self.verbose == 1:
            logger.setLevel(logging.WARNING)
        elif self.verbose == 2:
            logger.setLevel(logging.ERROR)

    def _evaluate_performance_triggers(
            self, 
            daily_performance: dict[str, dict[str, dict[str, Any]]],
            past_report: dict[str, dict[str, dict[str, Any]]],
        ) -> dict[str, Any]:
        """
        Evaluate whether the performance triggers retraining based on the daily performance and past report.

        Args:
            daily_performance (dict[str, dict[str, dict[str, Any]]]): The performance metrics for the current batch.
            past_report (dict[str, dict[str, dict[str, Any]]]): The historical performance report.

        Returns:
            dict[str, Any]: The evaluation result indicating whether retraining is triggered and the reason.
        
        Raises:
            ValueError: If the daily performance or past report are not structured correctly.
        """
        # Validate input types
        validate_type(
            daily_performance=(daily_performance, dict),
            past_report=(past_report, dict),
        )

        if not daily_performance:
            # Initialize the daily performance report dictionary
            performance_report: dict[str, Any] = {
                "triggered": None,
                "consecutive_critical_count": None,
                "is_catastrophic": None,
                "reason": "⚠️ Attention: No daily performance data available, maybe the batch file doesn't have y_true!"
            }
            return performance_report
        
        if len(daily_performance.keys()) != 1:
            raise ValueError("The 'daily_performance' should contain exactly one file's performance data.")

        # Extract the file name and corresponding daily performance data
        name_file = list(daily_performance.keys())[0]
        daily_performance = daily_performance[name_file]

        # Initialize the daily performance report dictionary
        performance_report: dict[str, Any] = {
            "triggered": False,
            "consecutive_critical_count": 0,
            "is_catastrophic": False,
            "reason": None
        }

        if ("alarm" not in daily_performance or 
            "severity" not in daily_performance["alarm"]
        ):
            raise ValueError("The 'daily_performance' doesn't contain the 'alarm' key or 'severity' information.")
        # Check if today's performance is critical and update the consecutive critical count accordingly
        if daily_performance["alarm"]["severity"] == "CRITICAL":
            performance_report["consecutive_critical_count"]  += 1
        else:
            performance_report["reason"] = "🟢 OK"
            logger.info("Performance is not critical. No retraining triggered.")
            return performance_report

        # Update the consecutive critical count based on past performance reports if available
        if past_report:
            days = list(past_report.keys())
            # Determine the last day for which we have a performance report
            if name_file in days:
                idx = days.index(name_file)
                last_day = days[idx - 1] if idx > 0 else None
            else:
                last_day = days[-1] if days else None

            if last_day is not None:
                if (
                    "performance" not in past_report[last_day] or 
                    "consecutive_critical_count" not in past_report[last_day]["performance"]
                ):
                    raise ValueError("The 'past_report' for the last day doesn't contain the 'consecutive_critical_count' information.")
                performance_report["consecutive_critical_count"] += \
                    past_report[last_day]["performance"]["consecutive_critical_count"] 

        if (
            "delta_vs_baseline" not in daily_performance or 
            self.metric_to_monitor not in daily_performance["delta_vs_baseline"] or 
            "relative_drop_pct" not in daily_performance["delta_vs_baseline"][self.metric_to_monitor]
        ):
            raise ValueError("The 'daily_performance' for the current batch doesn't contain the 'relative_drop_pct' information.")
        # Check for catastrophic drop before evaluating consecutive critical count
        if daily_performance["delta_vs_baseline"][self.metric_to_monitor]["relative_drop_pct"] >= self.catastrophic_drop_threshold:
            performance_report["triggered"] = True    # <------- Retraining Enablement
            performance_report["is_catastrophic"] = True
            performance_report["reason"] = (
                f"🔴🔴🔴 Catastrophic drop {self.metric_to_monitor} "
                f"({daily_performance['delta_vs_baseline'][self.metric_to_monitor]['relative_drop_pct'] * 100:.2f}%) "
                f"-> TRIGGERED"
            )
            logger.warning("Catastrophic drop detected for metric: %s", self.metric_to_monitor)
            return performance_report

        # Check if the consecutive critical count has reached the threshold
        if performance_report["consecutive_critical_count"] >= self.consecutive_critical_threshold:
            performance_report["triggered"] = True    # <------- Retraining Enablement
            performance_report["reason"]= (
                f"🔴 {self.metric_to_monitor} for {performance_report['consecutive_critical_count']} consecutive days"
                f" -> TRIGGERED"
            )
            logger.warning(
                "The metric %s has been critical for %d consecutive days. Triggering retraining.", 
                self.metric_to_monitor, performance_report["consecutive_critical_count"]
            )
            return performance_report

        # If neither catastrophic nor consecutive critical conditions are met, provide a warning reason
        performance_report["reason"] = (
            f"🟡 {self.metric_to_monitor} critical: "
            f"{performance_report['consecutive_critical_count']}/{self.consecutive_critical_threshold} "
            f"(drop of {daily_performance['delta_vs_baseline'][self.metric_to_monitor]['relative_drop_pct']*100:.2f}%)"
        )
        logger.warning(
            "The metric %s has been critical for %d with a drop of %.2f%%.", 
            self.metric_to_monitor, performance_report["consecutive_critical_count"], 
            daily_performance['delta_vs_baseline'][self.metric_to_monitor]['relative_drop_pct'] * 100
        )

        return performance_report


    def _evaluate_drift_triggers(
            self, 
            daily_drift: dict[str, dict[str, Any]],
        ) -> dict[str, Any]:
        """
        Evaluate the drift triggers based on the daily drift report.

        Args:
            daily_drift (dict[str, dict[str, Any]]): The daily drift report for a single file.

        Returns:
            dict[str, Any]: The drift report indicating whether retraining is triggered and the reasons.
        
        Raises:
            ValueError: If the daily drift report is not structured correctly.
        """

        validate_type(
            daily_drift=(daily_drift, dict),
        )

        if len(daily_drift.keys()) != 1:
            raise ValueError("The 'daily_drift' should contain exactly one file's drift data.")

        # Extract the file name and corresponding daily drift data
        name_file = list(daily_drift.keys())[0]
        daily_drift = daily_drift[name_file]

        # Initialize the drift report dictionary with default values
        drift_report: dict[str, Any] = {
            "triggered": False,
            "drift_ratio": 0.0,
            "drifted_features_count": 0,
            "total_features_count": 0,
            "drifted_features": [],
            "prediction_drift_detected": False,
            "target_drift_detected": None,
            "reason": "🟢 OK"
        }

        # Validate that the daily drift report contains the expected keys and structure
        keys_drift_report = list(daily_drift.keys())
        keys_drift = ["features_drift", "prediction_drift", "true_target_drift"]
        if not all(key in keys_drift_report for key in keys_drift):
            raise ValueError("Some expected drift features are missing in the daily drift report!")

        # Validate that the 'features_drift' section contains the expected keys
        features_drift_report = list(daily_drift["features_drift"].keys())
        features_drift = ["drift_ratio", "drifted_features_count", "total_features_count", "drifted_features"]
        if not all(feature in features_drift_report for feature in features_drift):
            raise ValueError("Some expected features are missing in the 'features_drift' section of the daily drift report!")

        # Populate the drift report with the relevant information from the daily drift report
        drift_report["drift_ratio"] = daily_drift["features_drift"]["drift_ratio"]
        drift_report["drifted_features_count"] = daily_drift["features_drift"]["drifted_features_count"]
        drift_report["total_features_count"] = daily_drift["features_drift"]["total_features_count"]
        drift_report["drifted_features"] = daily_drift["features_drift"]["drifted_features"]

        # Validate that the 'prediction_drift' section contains the expected key
        if "prediction_drift_detected" not in daily_drift["prediction_drift"]:
            raise ValueError("Some expected features are missing in the 'prediction_drift' section of the daily drift report!")
        drift_report["prediction_drift_detected"] = daily_drift["prediction_drift"]["prediction_drift_detected"]

        # Validate that the 'true_target_drift' section contains the expected key
        if daily_drift["true_target_drift"]:
            drift_report["target_drift_detected"] = daily_drift["true_target_drift"]["target_drift_detected"]

        # Check if the drift ratio exceeds the maximum allowed threshold and update the drift report accordingly
        if drift_report["drift_ratio"] > self.max_drift_ratio:
            drift_report["triggered"] = True
            drift_report["reason"] = (
                "🔴 Feature drift detected: "
                f"{drift_report['drifted_features_count']}/{drift_report['total_features_count']}"
                f" features drifted ({drift_report['drift_ratio']*100:.2f}% >= {self.max_drift_ratio*100:.2f}%)" 
                " -> TRIGGERED"
            )
        # If the drift ratio is within the acceptable range, provide a warning reason
        if drift_report["triggered"]:
            logger.warning(
                "The batch has triggered a feature drift with a drift ratio of %.2f%%.", 
                drift_report["drift_ratio"] * 100
            )

        return drift_report

    def _check_cooldown(
            self,
            daily_report: dict,
            past_report: dict[str, dict[str, Any]],
        ) -> dict[str, Any]:
        """
        Check if the retraining policy is currently in a cooldown period based on the daily report and past report.

        Args:
            daily_report (dict): The daily report for the current batch.
            past_report (dict[str, dict[str, Any]]): The historical report containing past retraining decisions.

        Returns:
            dict[str, Any]: The cooldown report indicating whether the policy is in cooldown and related information.

        Raises:
            ValueError: If the daily report or past report are not structured correctly.
        """

        validate_type(
            daily_report=(daily_report, dict),
            past_report=(past_report, dict)
        )

        # Check if the daily report contain only one key and extract the file name
        if len(daily_report.keys()) != 1:
            raise ValueError("Attention the daily report should contain exactly one key representing the current day.")
        name_file = str(list(daily_report.keys())[0])

        # initialize cooldown setup
        cooldown_report: dict[str, Any] = {
            "in_cooldown": False,
            "batches_since_last_retraining": None,  
            "remaining_cooldown_batches": 0,
            "last_retraining_batch": None,
            "reason": "🟢 Cooldown inactive"
        }

        if past_report:
            # Select yesterday file name
            days = list(past_report.keys())
            if name_file in days:
                idx = days.index(name_file)
                last_day = days[idx - 1] if idx > 0 else None
            else:
                last_day = days[-1] if days else None

            # If this file is the first (so last day is None)
            if last_day is None:
                return cooldown_report

            # If there was retraining trigger yesterday
            if past_report[last_day]["decision"]["retraining"]:
                cooldown_report["in_cooldown"] = True
                cooldown_report["batches_since_last_retraining"] = 1
                cooldown_report["remaining_cooldown_batches"] = int(self.cooldown_batches)
                cooldown_report["last_retraining_batch"] = last_day
                cooldown_report["reason"] = "🔴 Cooldown active"
                return cooldown_report

            # If there was no retraining trigger yesterday, but we are still in cooldown
            if past_report[last_day]["cooldown"]["remaining_cooldown_batches"] > 0:
                cooldown_report["batches_since_last_retraining"] = \
                    int(past_report[last_day]["cooldown"]["batches_since_last_retraining"] + 1)
                cooldown_report["remaining_cooldown_batches"] = \
                    int(past_report[last_day]["cooldown"]["remaining_cooldown_batches"] - 1)
                cooldown_report["last_retraining_batch"] = past_report[last_day]["cooldown"]["last_retraining_batch"]
                # If the remaining cooldown batches is zero, the cooldown has ended
                if cooldown_report["remaining_cooldown_batches"] > 0:
                    cooldown_report["in_cooldown"] = True
                    cooldown_report["reason"] = "🔴 Cooldown active"
                return cooldown_report
            else:
                if past_report[last_day]["cooldown"]["last_retraining_batch"] is not None:
                    cooldown_report["batches_since_last_retraining"] = \
                        int(past_report[last_day]["cooldown"]["batches_since_last_retraining"] + 1)
                    cooldown_report["last_retraining_batch"] = past_report[last_day]["cooldown"]["last_retraining_batch"]
                cooldown_report["remaining_cooldown_batches"] = \
                    past_report[last_day]["cooldown"]["remaining_cooldown_batches"]
                return cooldown_report
        else:
            return cooldown_report

    def _synthesize_decision(
            self,
            daily_performance_report: dict[str, Any],
            daily_drift_report: dict[str, Any],
            daily_cooldown_report: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Synthesize the retraining decision based on the daily performance, drift, and cooldown reports.

        Args:
            daily_performance_report (dict[str, Any]): The daily performance report.
            daily_drift_report (dict[str, Any]): The daily drift report.
            daily_cooldown_report (dict[str, Any]): The daily cooldown report.

        Returns:
            dict[str, Any]: The synthesized decision report indicating whether retraining is triggered and the reasons.
        
        Raises:
            ValueError: If any of the input reports are not structured correctly.
        """

        validate_type(
            daily_performance_report=(daily_performance_report, dict),
            daily_drift_report=(daily_drift_report, dict),
            daily_cooldown_report=(daily_cooldown_report, dict),
        )

        # Initialize the decision report with default values
        decision_report: dict[str, Any] = {
            "retraining": False,
            "action": "NO_ACTION",
            "primary_trigger": None,
            "reason": "🟢 All systems normal"
        }

        # Check if the retraining policy is currently in a cooldown period
        if daily_cooldown_report["in_cooldown"]:
            decision_report["action"] = "COOLDOWN_HOLD"
            decision_report["reason"] = (
                f"⚠️ Retraining is on hold: cooldown period is active "
                f"({daily_cooldown_report['remaining_cooldown_batches']} batches remaining)."
            )
            return decision_report
        else:
            # If not in cooldown, evaluate performance and drift triggers
            if daily_performance_report.get("triggered") is not None:
                # If the daily performance report indicates a trigger, update the decision report accordingly
                if daily_performance_report["triggered"]:
                    decision_report["retraining"] = True
                    decision_report["action"] = "RETRAINING"
                    # If the performance is catastrophic, set the primary trigger and reason accordingly
                    if daily_performance_report.get("is_catastrophic"):
                        decision_report["primary_trigger"] = "PERFORMANCE_CATASTROPHIC"
                        decision_report["reason"] = "🔴 Catastrophic drop in performance, retraining required."
                    # If the performance is critical but not catastrophic, set the trigger and reason accordingly
                    else:
                        decision_report["primary_trigger"] = "PERFORMANCE_DEGRADATION"
                        decision_report["reason"] = "🔴 Persistent performance degradation, retraining required."
                    return decision_report
                # If the daily performance report does not indicate a trigger, check for drift triggers
                else:
                    if daily_drift_report.get("triggered"):
                        decision_report["reason"] = "🟢 Performance healthy, but feature drift detected (monitoring)."
                    return decision_report
            # If the daily performance report does not indicate a trigger, check for drift triggers
            else:
                # If the daily drift report indicates a trigger, update the decision report accordingly
                if daily_drift_report.get("triggered"):
                    decision_report["retraining"] = True
                    decision_report["action"] = "RETRAINING"
                    decision_report["primary_trigger"] = "DRIFT_DETECTED"
                    decision_report["reason"] = "🔴 Data drift detected on unlabeled batch, retraining required."
                    return decision_report
                # If neither performance nor drift triggers are present, indicate that all systems are normal
                else:
                    return decision_report


    def evaluate(
            self,
            name_file: str,
            daily_drift: dict[str, dict[str, Any]],
            daily_performance: dict[str, dict[str, dict[str, Any]]],
            past_report: dict[str, dict[str, dict[str, Any]]],
        ) -> dict[str, dict[str, Any]]:
        """
        Evaluate the retraining policy based on the past report, current batch, and drift/performance detectors.

        Args:
            name_file (str): The name of the current file.
            daily_drift (dict[str, dict[str, Any]]): The drift report for the current batch.
            daily_performance (dict[str, dict[str, dict[str, Any]]]): The performance report for the current batch.
            past_report (dict[str, dict[str, dict[str, Any]]]): The past report containing previous evaluation results.

        Returns:
            dict[str, dict[str, Any]]: The daily report containing performance, drift, cooldown, 
                                       and decision information for the current batch.
        
        Raises:
            ValueError: If the daily performance or drift reports are not structured correctly.

        Note:
            This method mutates `past_report` in-place by adding the current day's
            entry. The returned dict is the same object as the input `past_report`.
        """

        validate_type(
            name_file=(name_file, str),
            daily_drift=(daily_drift, dict),
            daily_performance=(daily_performance, dict),
            past_report=(past_report, dict),
        )

        # Initialize the daily report dictionary
        daily_report: dict[str, dict[str, dict]] = {}
        daily_report[f'{name_file}'] = {}

        # DRIFT
        daily_drift_report = self._evaluate_drift_triggers(
            daily_drift=daily_drift,
        )
        daily_report[name_file]['drift'] = daily_drift_report

        # PERFORMANCE
        daily_performance_report = self._evaluate_performance_triggers(
            daily_performance=daily_performance,
            past_report=past_report,
        )
        daily_report[name_file]['performance'] = daily_performance_report

        # COOLDOWN: Check if the retraining policy is currently in a cooldown period.
        daily_check_cooldown = self._check_cooldown(
            daily_report=daily_report,
            past_report=past_report
        )

        # SYNTHESIZE DECISION: 
        # Synthesize the retraining decision based on the daily performance, drift, and cooldown reports.
        daily_synthesize_decision = self._synthesize_decision(
            daily_performance_report=daily_performance_report,
            daily_drift_report=daily_drift_report,
            daily_cooldown_report=daily_check_cooldown,
        )

        # Update the daily report with the cooldown and decision information
        past_report[name_file] = {
            "performance": daily_performance_report,
            "drift": daily_drift_report,
            "cooldown": daily_check_cooldown,
            "decision": daily_synthesize_decision,
        }

        return past_report

        


