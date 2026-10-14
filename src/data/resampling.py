from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from configs.pipeline_loader import load_random_seed_config


class ResamplerFactory:
    _registry = {
        "smote": SMOTE,
        "undersampling": RandomUnderSampler,
        "class_weight": None,  # <-- I'm setting it to 'None' because the default models have class_weight = 'balanced'
    }

    @classmethod
    def get_resampler(cls, strategy_name: str, random_state: int | None = None):
        strategy = strategy_name

        if strategy not in cls._registry:
            raise ValueError(f"Strategia applicata per il simile a '{strategy}' non valida!")

        resampler_class = cls._registry[strategy]

        if resampler_class is None:
            return None

        if random_state is None:
            random_state = load_random_seed_config().random_seed_resempling

        return resampler_class(random_state=random_state)
